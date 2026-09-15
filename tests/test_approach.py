"""Walking up to the cubes: where it aims, and that it gets there."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocky.approach import Approach, ApproachParams, Twist, wrap_pi, yaw_of

RED = np.array([0.345, 0.058])
BLUE = np.array([0.345, -0.058])


def quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


@pytest.fixture(scope="module")
def appr():
    return Approach()


def test_yaw_of_inverts_the_quaternion():
    for yaw in np.linspace(-3.0, 3.0, 13):
        assert yaw_of(quat(yaw)) == pytest.approx(yaw, abs=1e-9)


def test_the_goal_sits_on_the_perpendicular_bisector(appr):
    """Equal ranges to both cubes is the whole reason for taking the axis."""
    mid = (RED + BLUE) / 2.0
    for start in ([-1.0, 0.0], [-0.8, 0.5], [-0.6, -0.7]):
        goal, yaw = appr.goal(start, mid, RED - BLUE)
        assert np.hypot(*(RED - goal)) == pytest.approx(np.hypot(*(BLUE - goal)), abs=1e-9)
        assert np.hypot(*(mid - goal)) == pytest.approx(appr.p.stand_off, abs=1e-9)
        # and it faces the cubes from the side the robot is already on
        assert (mid - goal) @ (mid - np.asarray(start, float)) > 0.0
        assert wrap_pi(math.atan2(*(mid - goal)[::-1]) - yaw) == pytest.approx(0.0, abs=1e-9)


def test_the_stand_off_is_inside_the_arms_working_band(appr):
    assert 0.340 <= appr.p.stand_off <= 0.380


def test_the_speed_envelope_is_one_the_gait_can_actually_hold(appr):
    """Sampled once at construction, so it had better match the real check."""
    for heading in np.linspace(-math.pi, math.pi, 9):
        v = appr.envelope(float(heading))
        assert v > 0.0
        report = appr.gait.check_feasible(v * math.cos(heading), v * math.sin(heading), 0.0,
                                          samples=96)
        assert report["within_limits"], heading
        assert report["speed_margin"] > 1.0


def test_arrival_needs_the_hold_time(appr):
    a = Approach(ApproachParams(hold_s=0.5), appr.gait)
    mid = (RED + BLUE) / 2.0
    goal, yaw = a.goal([-1.0, 0.0], mid, RED - BLUE)
    parked = np.array([goal[0], goal[1], 0.058])
    assert not a.command(0.0, parked, quat(yaw), mid, RED - BLUE).arrived
    assert not a.command(0.3, parked, quat(yaw), mid, RED - BLUE).arrived
    assert a.command(0.6, parked, quat(yaw), mid, RED - BLUE).arrived
    # one step off target and the clock restarts
    assert not a.command(0.7, parked + np.array([0.2, 0, 0]), quat(yaw), mid, RED - BLUE).arrived
    assert not a.command(0.9, parked, quat(yaw), mid, RED - BLUE).arrived


def test_a_kinematic_robot_driven_by_the_twist_parks_where_it_was_told(appr):
    """No physics -- just integrate the commanded twist and see where it lands."""
    a = Approach(ApproachParams(hold_s=0.4), appr.gait)
    mid = (RED + BLUE) / 2.0
    for start, yaw0 in (([-0.9, 0.1], 0.2), ([-0.7, -0.5], -1.0), ([-1.2, 0.4], 2.5)):
        pos = np.array(start, float)
        yaw = yaw0
        a.reset()
        dt, t, tw = 0.02, 0.0, Twist(0, 0, 0, 9, 9, False)
        while t < 120.0 and not tw.arrived:
            tw = a.command(t, np.r_[pos, 0.058], quat(yaw), mid, RED - BLUE)
            c, s = math.cos(yaw), math.sin(yaw)
            pos += dt * np.array([c * tw.vx - s * tw.vy, s * tw.vx + c * tw.vy])
            yaw = wrap_pi(yaw + dt * tw.wz)
            t += dt
        assert tw.arrived, f"never parked from {start}"
        goal, want = a.goal(pos, mid, RED - BLUE)
        assert np.hypot(*(goal - pos)) <= a.p.position_tol + 1e-9
        assert abs(wrap_pi(want - yaw)) <= a.p.heading_tol + 1e-9
