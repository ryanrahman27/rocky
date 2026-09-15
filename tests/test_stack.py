"""The scripted stacking sequence, checked without running physics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocky import arm
from rocky import model_params as P
from rocky.stack import StackController, StackPlan

RED = (0.345, 0.058)
BLUE = (0.345, -0.058)
UPRIGHT = (1.0, 0.0, 0.0, 0.0)
ORIGIN = (0.0, 0.0, arm.BRACE.height)
ARM = ("sweep_0", "lift_0", "elbow_0", "wrist_0")


@pytest.fixture
def ctl():
    c = StackController()
    c.reset(RED, BLUE)
    return c


def grasp_point(step):
    q = np.array([step.joints[n] for n in ARM])
    return arm.forward(*q) + arm.grasp_offset(q[0])


def at_end_of(ctl, name):
    """The step at the last instant of a named phase."""
    t = 0.0
    for phase, dur, _, _ in ctl._segments:
        t += dur
        if phase == name:
            return ctl.step(t - 1e-6, ORIGIN, UPRIGHT)
    raise KeyError(name)


def roll(ctl, dt=0.01):
    t = 0.0
    while t <= ctl.duration:
        yield t, ctl.step(t, ORIGIN, UPRIGHT)
        t += dt


def test_every_actuator_gets_a_target_every_tick(ctl):
    names = {f"{j}_{k}" for k in range(P.N_LEGS) for j in ("sweep", "lift", "elbow")}
    names |= {"wrist_0", "jaw_r", "jaw_l"}
    for _, step in roll(ctl, dt=0.05):
        assert set(step.joints) == names
        assert all(np.isfinite(v) for v in step.joints.values())


def test_targets_stay_inside_the_joint_limits(ctl):
    for _, step in roll(ctl, dt=0.02):
        for k in range(P.N_LEGS):
            for name, rng in ((f"sweep_{k}", P.SWEEP_RANGE), (f"lift_{k}", P.LIFT_RANGE),
                              (f"elbow_{k}", P.elbow_range(k))):
                assert rng[0] - 1e-9 <= step.joints[name] <= rng[1] + 1e-9, name
        assert P.WRIST_RANGE[0] - 1e-9 <= step.joints["wrist_0"] <= P.WRIST_RANGE[1] + 1e-9
        for jaw in ("jaw_r", "jaw_l"):
            assert 0.0 <= step.joints[jaw] <= arm.JAW_OPEN + 1e-12


def test_the_trajectory_is_continuous(ctl):
    """Every seam included -- a jump at a phase boundary is a servo slamming."""
    prev = None
    for _, step in roll(ctl, dt=0.005):
        q = np.array([step.joints[n] for n in step.joints])
        if prev is not None:
            assert np.abs(q - prev).max() < math.radians(2.0)
        prev = q


def test_the_gripper_reaches_the_cubes_it_is_aimed_at(ctl):
    """End of `descend` on the red cube, end of `place` on top of the blue."""
    rise = ctl.p.grasp_rise
    want_pick = np.array([RED[0], RED[1], arm.CUBE / 2 + rise - arm.BRACE.height])
    want_place = np.array([BLUE[0], BLUE[1],
                           arm.CUBE / 2 + arm.CUBE + ctl.p.stack_gap + rise - arm.BRACE.height])
    assert np.allclose(grasp_point(at_end_of(ctl, "descend")), want_pick, atol=1e-3)
    assert np.allclose(grasp_point(at_end_of(ctl, "place")), want_place, atol=1e-3)


def test_the_grasp_sits_above_the_cube_centre_to_clear_the_gripper_body(ctl):
    """Aimed at the centre, the hull buries itself in the cube's top face."""
    assert ctl.p.grasp_rise > 0.005


def test_jaws_open_before_the_descent_and_shut_before_the_lift(ctl):
    seen = {}
    for _, step in roll(ctl, dt=0.02):
        seen.setdefault(step.phase, []).append(step.joints["jaw_r"])
    assert seen["descend"][-1] == pytest.approx(arm.JAW_OPEN)
    assert seen["close"][-1] < arm.CUBE / 2
    assert seen["lift"][-1] < arm.CUBE / 2
    assert seen["release"][-1] == pytest.approx(arm.JAW_OPEN)
    # The gripper is a foot again by the end, and a 60 mm-wide one is a hazard.
    assert seen["stow"][-1] == pytest.approx(ctl.p.stow_gap)


def test_the_brace_widens_four_legs_and_tucks_the_fifth(ctl):
    from rocky import kinematics as K
    walk = np.stack([K.forward_kinematics(k, ctl.walk_stance[k]) for k in range(P.N_LEGS)])
    brace = np.stack([K.forward_kinematics(k, ctl.brace_stance[k]) for k in range(P.N_LEGS)])
    for k in range(P.N_LEGS):
        r_walk, r_brace = np.hypot(*walk[k, :2]), np.hypot(*brace[k, :2])
        if k == P.MANIP_LEG:
            assert r_brace < r_walk - 0.01
        else:
            assert r_brace > r_walk + 0.01


def test_the_trim_converges_on_a_constant_sag_and_refuses_to_chase_a_block(ctl):
    """Integrate a servo that is 3 deg behind, then one that is jammed 30 deg."""
    sag = math.radians(3.0)
    t = 0.0
    while t < 6.0:                      # inside the Cartesian phases
        step = ctl.step(t, ORIGIN, UPRIGHT,
                        np.array([step.joints[n] for n in ARM]) - sag if t else np.zeros(4))
        t += 0.01
    assert ctl._trim[0] == pytest.approx(sag, abs=math.radians(0.5))

    blocked = StackController()
    blocked.reset(RED, BLUE)
    t = 0.0
    step = None
    while t < 6.0:
        meas = (np.array([step.joints[n] for n in ARM]) - math.radians(30.0)
                if step else np.zeros(4))
        step = blocked.step(t, ORIGIN, UPRIGHT, meas)
        t += 0.01
    assert np.abs(blocked._trim).max() < 1e-9


def test_the_plan_is_all_positive_durations():
    p = StackPlan()
    ctl = StackController(p)
    assert all(dur > 0 for _, dur, _, _ in ctl._segments)
    assert ctl.duration == pytest.approx(sum(dur for _, dur, _, _ in ctl._segments))
