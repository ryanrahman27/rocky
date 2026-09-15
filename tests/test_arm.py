"""Manipulator kinematics: against the compiled model, and round-tripped."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocky import arm
from rocky import model_params as P

mujoco = pytest.importorskip("mujoco")

MODEL = "model/rocky_manip.xml"
ARM = ("sweep_0", "lift_0", "elbow_0", "wrist_0")


@pytest.fixture(scope="module")
def sim():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    adr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM}
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp")
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    return m, d, adr, site, base


def grasp_in_body(sim, q):
    m, d, adr, site, base = sim
    d.qpos[:] = m.key_qpos[0]
    for n, v in zip(ARM, q):
        d.qpos[adr[n]] = float(v)
    mujoco.mj_kinematics(m, d)
    R = d.xmat[base].reshape(3, 3)
    return R.T @ (d.site_xpos[site] - d.xpos[base])


def random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.uniform(*P.SWEEP_RANGE, n),
        rng.uniform(*P.LIFT_RANGE, n),
        rng.uniform(*P.elbow_range(arm.ARM), n),
        rng.uniform(*P.WRIST_RANGE, n),
    ], axis=1)


def test_forward_matches_the_compiled_grasp_site(sim):
    """The analytic chain has to agree with the MJCF, not just with itself."""
    for q in random_poses(300):
        got = arm.forward(*q) + arm.grasp_offset(q[0])
        assert np.allclose(got, grasp_in_body(sim, q), atol=1e-9)


def test_grasp_offset_is_the_jaw_centreline():
    """The offset is perpendicular to the swept plane and 7.6 mm long."""
    for sweep in np.linspace(*P.SWEEP_RANGE, 9):
        off = arm.grasp_offset(sweep)
        assert off[2] == pytest.approx(0.0)
        assert np.hypot(*off[:2]) == pytest.approx(arm.GRASP_Y, abs=1e-12)
        radial = np.array([math.cos(sweep), math.sin(sweep)])
        assert off[:2] @ radial == pytest.approx(0.0, abs=1e-12)


def test_inverse_round_trips_at_a_fixed_approach():
    for q in random_poses(200, seed=1):
        target = arm.forward(*q)
        back = arm.inverse(target, arm.approach_angle(q))
        assert np.allclose(arm.forward(*back), target, atol=1e-9)


def test_reach_for_hits_the_jaw_centre_not_the_swept_plane():
    """Aiming the swept plane at a cube puts it against one finger."""
    for target in ([0.34, 0.06, -0.030], [0.36, -0.08, -0.018], [0.30, 0.0, 0.02]):
        q = arm.reach_for(np.array(target))
        got = arm.forward(*q) + arm.grasp_offset(q[0])
        assert np.allclose(got, target, atol=1e-6)


def test_auto_approach_prefers_straight_down_but_gives_it_up_when_it_must():
    """The wrist runs out before the reach does, so the tilt has to slacken."""
    near = arm.reach_for(np.array([0.32, 0.0, -0.038]))
    far = arm.reach_for(np.array([0.40, 0.0, -0.038]))
    assert arm.approach_angle(near) == pytest.approx(-math.pi / 2, abs=1e-6)
    assert arm.approach_angle(far) > -math.pi / 2 + math.radians(20.0)
    for q in (near, far):
        assert np.all(q >= arm._LO - 1e-9) and np.all(q <= arm._HI + 1e-9)


def test_every_solution_respects_the_joint_limits():
    rng = np.random.default_rng(5)
    solved = 0
    for _ in range(300):
        R = rng.uniform(0.26, 0.40)
        th = rng.uniform(-0.4, 0.4)
        z = rng.uniform(-0.05, 0.10)
        try:
            q = arm.reach_for(np.array([R * math.cos(th), R * math.sin(th), z]))
        except arm.Unreachable:
            continue
        solved += 1
        assert np.all(q >= arm._LO - 1e-9) and np.all(q <= arm._HI + 1e-9)
    assert solved > 200


def test_unreachable_is_raised_not_silently_clipped():
    with pytest.raises(arm.Unreachable):
        arm.inverse(np.array([1.0, 0.0, 0.0]))
    clipped = arm.inverse(np.array([1.0, 0.0, 0.0]), clip=True)
    assert np.all(clipped >= arm._LO - 1e-9) and np.all(clipped <= arm._HI + 1e-9)


def test_the_brace_draws_leg_zero_in_and_the_others_out():
    """Widening the arm with the rest walks the open gripper into the cubes."""
    assert arm.BRACE.spread > 1.0
    assert arm.BRACE.arm_tuck < 1.0
