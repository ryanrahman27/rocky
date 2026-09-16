"""Echolocation: the estimator, against a scene only the test is allowed to see."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocky import arm
from rocky.gait import WaveGait
from rocky.kinematics import limb_points
from rocky.sonar import PairTracker, Sonar

mujoco = pytest.importorskip("mujoco")

MODEL = "model/rocky_cubes.xml"
AWAY = (9.0, 9.0)


@pytest.fixture(scope="module")
def rig():
    m = mujoco.MjModel.from_xml_path(MODEL)
    son = Sonar(m)
    acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    qa = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in acts}
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "start")
    stance = WaveGait().neutral_joint_targets()
    return m, son, acts, qa, key, stance


def place(rig, red, blue, settle=400):
    m, son, acts, qa, key, stance = rig
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    for k in range(5):
        for n, v in zip((f"sweep_{k}", f"lift_{k}", f"elbow_{k}"), stance[k]):
            d.qpos[qa[n]] = float(v)
    d.qpos[qa["jaw_r"]] = d.qpos[qa["jaw_l"]] = arm.JAW_OPEN
    d.qpos[2] = 0.075
    for name, xy in (("cube_red_free", red), ("cube_blue_free", blue)):
        a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
        d.qpos[a:a + 7] = (xy[0], xy[1], arm.CUBE / 2, 1, 0, 0, 0)
    mujoco.mj_forward(m, d)
    d.ctrl[:] = [d.qpos[qa[n]] for n in acts]
    for _ in range(settle):
        mujoco.mj_step(m, d)
    return d


def test_the_ring_can_see_past_the_robots_own_legs(rig):
    """The emitters live in the gaps between the limbs for exactly this reason."""
    m, son, *_ = rig
    d = place(rig, AWAY, AWAY)
    ranges = son.ranges(d)
    res = son.residual(ranges, son.up(d))
    assert res.max() < 0.02, "an empty floor should look empty from every ray"
    assert son.detect(d) == []


def test_ride_height_comes_off_the_floor_itself(rig):
    m, son, *_ = rig
    d = place(rig, AWAY, AWAY)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert son.floor_plane(son.ranges(d), son.up(d)) == pytest.approx(d.xpos[base][2], abs=2e-3)


@pytest.mark.parametrize("R", [0.90, 0.70, 0.55, 0.45, 0.34])
def test_a_cube_is_heard_from_a_metre_in_to_arms_length(rig, R):
    th = math.radians(36.0)
    d = place(rig, (R * math.cos(th), R * math.sin(th)), AWAY)
    found = rig[1].detect(d)
    assert found, f"nothing heard at {R} m"
    best = min(found, key=lambda e: abs(e.range - R))
    assert abs(best.range - R) < 0.025
    assert abs(math.degrees(best.bearing) - 36.0) < 3.0


def test_both_cubes_resolve_as_a_pair_across_the_approach(rig):
    for R in (0.55, 0.45, 0.38, 0.34):
        th, half = math.radians(36.0), 0.115 / (2 * R)
        p1 = (R * math.cos(th + half), R * math.sin(th + half))
        p2 = (R * math.cos(th - half), R * math.sin(th - half))
        d = place(rig, p1, p2)
        found = rig[1].detect(d)
        assert len(found) == 2, f"{len(found)} objects at {R} m, expected 2"
        mid = (found[0].xy + found[1].xy) / 2.0
        assert np.hypot(*(mid - (np.array(p1) + np.array(p2)) / 2.0)) < 0.025


def test_swinging_legs_are_not_mistaken_for_objects(rig):
    """A leg in swing travels straight through the gap the emitters fire down."""
    m, son, acts, qa, key, stance = rig
    gait = WaveGait()
    d = place(rig, AWAY, AWAY)
    phantoms = 0
    for i in range(int(gait.p.period / m.opt.timestep)):
        t = i * m.opt.timestep
        q = gait.joint_targets(t, 0.03, 0.0, 0.0)
        ctrl = {"jaw_r": arm.JAW_OPEN, "jaw_l": arm.JAW_OPEN, "wrist_0": 0.0}
        for k in range(5):
            ctrl[f"sweep_{k}"], ctrl[f"lift_{k}"], ctrl[f"elbow_{k}"] = (float(v) for v in q[k])
        d.ctrl[:] = [ctrl[n] for n in acts]
        mujoco.mj_step(m, d)
        if i % 10 == 0:
            phantoms += len(son.detect(d, limbs=limb_points(q)))
    assert phantoms == 0, f"{phantoms} phantom objects over one gait cycle"


def test_touch_is_the_only_thing_that_knows_the_grasp_worked(rig):
    d = place(rig, (0.34, 0.06), (0.34, -0.06))
    assert not rig[1].gripping(d), "nothing in the hand, but the pads say otherwise"
    assert rig[1].touch(d).shape == (2,)


def test_the_tracker_needs_more_than_one_ping_to_believe(rig):
    """Track before detect: a single ray is a rumour, not a cube."""
    m, son, *_ = rig
    th, R = math.radians(36.0), 0.45
    half = 0.115 / (2 * R)
    d = place(rig, (R * math.cos(th + half), R * math.sin(th + half)),
              (R * math.cos(th - half), R * math.sin(th - half)))
    tr = PairTracker(son)
    assert tr.update(d, 0.05, 0.0, (0.0, 0.0)) is None, "believed the very first ping"
    pair = None
    for _ in range(6):
        pair = tr.update(d, 0.05, 0.0, (0.0, 0.0))
    assert pair is not None
    assert abs(np.hypot(*pair.midpoint) - R) < 0.03
    assert math.degrees(math.atan2(*pair.midpoint[::-1])) == pytest.approx(36.0, abs=4.0)


def test_the_belief_coasts_when_the_sonar_goes_quiet(rig):
    """Squaring up puts the cubes behind the front limb; the IMU carries them."""
    m, son, *_ = rig
    th, R, half = math.radians(36.0), 0.45, 0.115 / (2 * 0.45)
    d = place(rig, (R * math.cos(th + half), R * math.sin(th + half)),
              (R * math.cos(th - half), R * math.sin(th - half)))
    tr = PairTracker(son)
    for _ in range(6):
        pair = tr.update(d, 0.05, 0.0, (0.0, 0.0))
    assert pair is not None
    before = pair.midpoint.copy()
    empty = place(rig, AWAY, AWAY)
    # half a second of turning at 0.2 rad/s with nothing to hear
    for _ in range(10):
        pair = tr.update(empty, 0.05, 0.2, (0.0, 0.0))
    assert pair is not None, "dropped the cubes the moment it stopped hearing them"
    turned = -0.2 * 0.5
    c, s = math.cos(turned), math.sin(turned)
    expect = np.array([c * before[0] - s * before[1], s * before[0] + c * before[1]])
    assert np.hypot(*(pair.midpoint - expect)) < 0.02
    assert pair.age == pytest.approx(0.5, abs=1e-6)


def test_the_observation_is_the_size_it_claims_and_holds_no_secrets(rig):
    """A policy trained on this has to be runnable on hardware."""
    from rocky.sonar import observation
    m, son, acts, qa, key, stance = rig
    d = place(rig, (0.34, 0.06), (0.34, -0.06))
    obs = observation(son, d, qa, acts, belief=np.array([0.34, 0.06, 0.34, -0.06]))
    assert obs.dtype == np.float32
    assert obs.shape == (len(son.ring) + len(son.fan) + 2 + 2 * len(acts) + 6 + 4,)
    assert np.isfinite(obs).all()
    # every element is a sensor reading, a joint, or the robot's own belief
    assert np.allclose(obs[:len(son.ring)], son.residual(son.ranges(d), son.up(d)))
    assert np.allclose(obs[-4:], [0.34, 0.06, 0.34, -0.06])


def test_the_belief_is_the_only_derived_quantity(rig):
    """Without it the observation is the same however far away the cubes are."""
    from rocky.sonar import observation
    m, son, acts, qa, key, stance = rig
    d = place(rig, (0.34, 0.06), (0.34, -0.06))
    a = observation(son, d, qa, acts, belief=np.array([0.34, 0.06, 0.34, -0.06]))
    b = observation(son, d, qa, acts, belief=np.array([0.36, 0.05, 0.33, -0.07]))
    assert np.allclose(a[:-4], b[:-4])
    assert not np.allclose(a[-4:], b[-4:])
