"""Gait-phase term maths, exercised against a stub env (no mjlab needed)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch", reason="gait MDP terms need torch")

from rocky.gait import GaitParams  # noqa: E402
from rocky import gait_mdp as G  # noqa: E402


class _Sensor:
    def __init__(self, found):
        self.data = types.SimpleNamespace(found=found, heights=None)


class _HeightSensor:
    def __init__(self, heights):
        self.data = types.SimpleNamespace(heights=heights)


def make_env(step, contacts=None, heights=None, command=(0.05, 0.0, 0.0), num_envs=1, dt=0.02):
    b = num_envs
    cmd = torch.tensor([list(command)] * b, dtype=torch.float32)
    env = types.SimpleNamespace(
        num_envs=b,
        device=torch.device("cpu"),
        step_dt=dt,
        episode_length_buf=torch.full((b,), step, dtype=torch.long),
        extras={"log": {}},
        command_manager=types.SimpleNamespace(get_command=lambda name: cmd),
        scene={
            "feet": _Sensor(contacts if contacts is not None else torch.ones(b, 5)),
            "foot_height_scan": _HeightSensor(heights if heights is not None else torch.full((b, 5), 0.012)),
        },
    )
    return env


P = GaitParams()


def test_phase_advances_one_cycle_per_period():
    dt = 0.02
    steps = round(P.period / dt)
    a = G.gait_phase(make_env(0, dt=dt), P.period)
    b = G.gait_phase(make_env(steps, dt=dt), P.period)
    wrapped = torch.minimum((a - b).abs(), 1.0 - (a - b).abs())   # phase is circular
    assert float(wrapped.max()) < 1e-5


def test_phase_offsets_spread_environments():
    ph = G.gait_phase(make_env(0, num_envs=5), P.period)
    assert torch.allclose(ph, torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8]), atol=1e-6)


def test_exactly_one_leg_swings_at_a_time():
    """The defining property of the wave gait."""
    dt = 0.01
    for step in range(round(P.period / dt)):
        env = make_env(step, dt=dt)
        ph = G._leg_phase(env, P.period, P.swing_order)
        n_swing = int((ph < P.swing_fraction - G._BOUNDARY_EPS).sum())
        assert n_swing == 1, f"step {step}: {n_swing} legs airborne"


def test_schedule_order_matches_gait_module():
    """Leg lift-off order in the reward must equal the scripted gait's."""
    dt = 0.01
    seen = []
    for step in range(round(P.period / dt)):
        env = make_env(step, dt=dt)
        ph = G._leg_phase(env, P.period, P.swing_order)
        leg = int(torch.argmin(torch.where(ph < P.swing_fraction - G._BOUNDARY_EPS, ph, torch.ones_like(ph))))
        if not seen or seen[-1] != leg:
            seen.append(leg)
    assert seen == list(P.swing_order), seen


def test_perfect_contact_tracking_scores_one():
    dt = 0.01
    for step in (0, 37, 91, 150):
        env = make_env(step, dt=dt)
        ph = G._leg_phase(env, P.period, P.swing_order)
        target = G._stance_target(env, P.period, P.swing_order, P.swing_fraction, blend=0.0)
        env = make_env(step, contacts=target, dt=dt)
        r = G.gait_contact_schedule(env, "feet", "twist", P.period, P.swing_order, P.swing_fraction, blend=0.0)
        assert torch.allclose(r, torch.ones(1), atol=1e-6)


def test_inverted_contact_scores_zero():
    env0 = make_env(50)
    target = G._stance_target(env0, P.period, P.swing_order, P.swing_fraction, blend=0.0)
    env = make_env(50, contacts=1.0 - target)
    r = G.gait_contact_schedule(env, "feet", "twist", P.period, P.swing_order, P.swing_fraction, blend=0.0)
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


def test_standing_command_wants_all_feet_down():
    env = make_env(50, contacts=torch.ones(1, 5), command=(0.0, 0.0, 0.0))
    r = G.gait_contact_schedule(env, "feet", "twist")
    assert torch.allclose(r, torch.ones(1), atol=1e-6)
    env = make_env(50, contacts=torch.zeros(1, 5), command=(0.0, 0.0, 0.0))
    r = G.gait_contact_schedule(env, "feet", "twist")
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


def test_clock_is_unit_circle():
    c = G.gait_clock(make_env(13), P.period)
    assert c.shape == (1, 2)
    assert torch.allclose(torch.linalg.norm(c, dim=-1), torch.ones(1), atol=1e-6)


def test_swing_clearance_zero_on_perfect_profile():
    dt = 0.01
    step = 12
    env0 = make_env(step, dt=dt)
    ph = G._leg_phase(env0, P.period, P.swing_order)
    s = torch.clamp(ph / P.swing_fraction, 0.0, 1.0)
    swinging = (ph < P.swing_fraction).float()
    perfect = 0.012 + swinging * P.step_height * torch.sin(torch.pi * s)
    env = make_env(step, heights=perfect, dt=dt, command=(0.05, 0.0, 0.0))
    cost = G.gait_swing_clearance(env, "foot_height_scan", "twist", P.period, P.swing_order, P.swing_fraction, P.step_height)
    assert float(cost) == pytest.approx(0.0, abs=1e-9)
