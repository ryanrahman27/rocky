"""Wave gait invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocky import kinematics as K
from rocky import model_params as P
from rocky.gait import CRUISE_SPEED, GaitParams, WaveGait, support_margin

GAIT = WaveGait()
TIMES = np.linspace(0.0, GAIT.p.period, 241)


def test_phase_offsets_are_a_permutation():
    off = GAIT.p.phase_offsets
    assert sorted(np.round(off * P.N_LEGS).astype(int)) == list(range(P.N_LEGS))


def test_exactly_one_leg_airborne():
    for t in TIMES:
        assert int(GAIT.in_swing(t).sum()) == 1, f"t={t}"


def test_swing_order_is_the_star_sequence():
    seen = []
    for t in np.linspace(0.0, GAIT.p.period, 2000, endpoint=False):
        leg = int(np.argmax(GAIT.in_swing(t)))
        if not seen or seen[-1] != leg:
            seen.append(leg)
    assert seen == list(GAIT.p.swing_order)


def test_bad_swing_order_rejected():
    with pytest.raises(ValueError):
        GaitParams(swing_order=(0, 1, 2, 3, 3))
    with pytest.raises(ValueError):
        GaitParams(duty=1.5)


def test_foot_targets_are_continuous():
    """No jump at lift-off or touch-down: a discontinuity would be a step command
    the servos cannot follow."""
    dt = 1e-4
    for t in np.linspace(0.0, GAIT.p.period, 400):
        a = GAIT.foot_targets(t, CRUISE_SPEED, 0.0, 0.1)
        b = GAIT.foot_targets(t + dt, CRUISE_SPEED, 0.0, 0.1)
        assert np.abs(b - a).max() < 1e-3, f"jump at t={t}"


def test_planted_feet_move_backwards_at_body_speed():
    """The kinematic contract of a stance phase."""
    v, dt = CRUISE_SPEED, 1e-4
    for t in np.linspace(0.0, GAIT.p.period, 200):
        sw = GAIT.in_swing(t)
        a = GAIT.foot_targets(t, v, 0.0, 0.0)
        b = GAIT.foot_targets(t + dt, v, 0.0, 0.0)
        vel = (b - a) / dt
        for k in range(P.N_LEGS):
            if not sw[k] and not GAIT.in_swing(t + dt)[k]:
                assert vel[k, 0] == pytest.approx(-v, abs=1e-4)
                assert abs(vel[k, 2]) < 1e-6


def test_swing_foot_clears_the_ground():
    peak = 0.0
    for t in TIMES:
        sw = GAIT.in_swing(t)
        z = GAIT.foot_targets(t, CRUISE_SPEED)[sw, 2][0] + GAIT.p.stance_height
        peak = max(peak, z)
        assert z >= -1e-9
    assert peak == pytest.approx(GAIT.p.step_height, abs=1e-3)


def test_support_polygon_always_contains_the_body_axis():
    for t in TIMES:
        m = support_margin(GAIT.foot_targets(t, CRUISE_SPEED), GAIT.in_swing(t))
        assert m > 0.05, f"margin {m * 1000:.0f} mm at t={t}"


def test_cruise_is_within_joint_limits_and_servo_speed():
    r = GAIT.check_feasible(CRUISE_SPEED, 0.0, 0.0)
    assert r["within_limits"]
    assert r["max_joint_speed"] < P.VELOCITY_LIMIT


def test_pure_cruise_needs_no_clipping():
    for t in TIMES:
        GAIT.joint_targets(t, CRUISE_SPEED, 0.0, 0.0, clip=False)


def test_translation_and_yaw_compete_for_workspace():
    """Documented coupling: near top speed there is almost no turning left."""
    assert GAIT.check_feasible(0.0, 0.0, 0.3)["within_limits"]
    assert not GAIT.check_feasible(CRUISE_SPEED, 0.0, 0.3)["within_limits"]


def test_clamp_command_makes_any_request_feasible():
    for cmd in [(CRUISE_SPEED, 0.0, 0.3), (0.2, 0.0, 0.0), (0.05, 0.05, 0.5)]:
        c = GAIT.clamp_command(*cmd)
        r = GAIT.check_feasible(*c)
        assert r["within_limits"], f"{cmd} -> {c}"
        assert np.dot(c, cmd) >= 0.0        # same direction, only scaled down


def test_joint_targets_respect_limits_over_a_cycle():
    cmd = GAIT.clamp_command(CRUISE_SPEED, 0.0, 0.2)
    for t in TIMES:
        q = GAIT.joint_targets(t, *cmd, clip=False)
        for k in range(P.N_LEGS):
            assert P.SWEEP_RANGE[0] <= q[k, 0] <= P.SWEEP_RANGE[1]
            assert P.LIFT_RANGE[0] <= q[k, 1] <= P.LIFT_RANGE[1]
            assert P.elbow_range(k)[0] <= q[k, 2] <= P.elbow_range(k)[1]


def test_neutral_stance_is_symmetric_and_complete():
    d = GAIT.neutral_joint_dict()
    assert set(d) == set(P.JOINT_NAMES)
    lifts = [d[f"lift_{k}"] for k in range(P.N_LEGS)]
    elbows = [d[f"elbow_{k}"] for k in range(P.N_LEGS)]
    assert max(lifts) - min(lifts) < math.radians(1.0)
    assert max(elbows) - min(elbows) < math.radians(1.0)
    assert all(abs(d[f"sweep_{k}"]) < 1e-9 for k in range(P.N_LEGS))


def test_action_vector_matches_model_joint_order():
    v = GAIT.action_vector(0.3, CRUISE_SPEED)
    assert v.shape == (16,)
    q = GAIT.joint_targets(0.3, CRUISE_SPEED)
    assert v[P.JOINT_NAMES.index("elbow_2")] == pytest.approx(q[2, 2])


def test_zero_command_holds_the_neutral_footprint():
    for t in TIMES:
        sw = GAIT.in_swing(t)
        planted = GAIT.foot_targets(t, 0.0, 0.0, 0.0)[~sw]
        assert np.allclose(planted, GAIT.neutral[~sw], atol=1e-12)


def test_max_speed_is_positive_and_feasible():
    v = GAIT.max_speed(servo_margin=1.2)
    assert v > 0.03
    assert GAIT.check_feasible(v, 0.0, 0.0)["within_limits"]
