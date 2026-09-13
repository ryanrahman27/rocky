"""Gait-phase observation and reward terms for Rocky (mjlab MDP terms).

The scripted wave gait in `rocky.gait` defines *when* each foot should be on the
ground: leg k lifts at cycle fraction `phase_offsets[k]` and stays airborne for
`swing_fraction` of the cycle. These terms hand the policy a clock and reward it
for honouring that contact schedule, so PPO inherits the gait's rhythm while
still choosing its own joint motion.

Standing is handled by an activity gate: when the commanded velocity is near
zero the schedule collapses to "all five feet planted", so the policy is not
pushed to march on the spot.

These are plain functions over torch tensors and an mjlab env; nothing here
imports mjlab, so they can be unit-tested against a stub env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rocky.gait import GaitParams

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor

_DEFAULT = GaitParams()

#: Tie-break at the swing/stance boundary. Phases land exactly on a boundary at
#: integer step counts, where float error alone decides whether a foot counts as
#: airborne -- which can make two legs "swing" at once in a gait defined to have
#: one. A hair under one part in 10,000 of a cycle (0.24 ms at 2.4 s) resolves it
#: toward stance without materially changing the schedule.
_BOUNDARY_EPS = 1e-4


def _offsets(env: ManagerBasedRlEnv, swing_order: tuple[int, ...]) -> torch.Tensor:
    off = torch.zeros(len(swing_order), device=env.device)
    for slot, leg in enumerate(swing_order):
        off[leg] = slot / len(swing_order)
    return off


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    """Global gait phase in [0, 1), shape [B].

    Time comes from the episode step counter, so the clock is deterministic and
    stateless. Each environment carries a fixed offset derived from its index,
    which spreads the batch over the whole cycle instead of locking every robot
    into the same phase.
    """
    t = env.episode_length_buf.float() * env.step_dt
    idx = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
    return torch.remainder(t / period + idx / max(env.num_envs, 1), 1.0)


def _leg_phase(env, period: float, swing_order: tuple[int, ...]) -> torch.Tensor:
    """Per-leg phase in [-eps, 1), shape [B, 5]. 0 is lift-off.

    A phase that should be exactly 0 can come back from `remainder` as something
    a hair under 1, which would read as "still in stance" at the very instant the
    foot is due to lift. Values within `_BOUNDARY_EPS` of a full cycle are folded
    back to just below zero so lift-off lands on the swing side.
    """
    ph = torch.remainder(
        gait_phase(env, period)[:, None] - _offsets(env, swing_order)[None, :], 1.0
    )
    return torch.where(ph > 1.0 - _BOUNDARY_EPS, ph - 1.0, ph)


def _activity(env, command_name: str, command_threshold: float) -> torch.Tensor:
    """0 when standing still, ramping to 1 once the command asks for motion."""
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    mag = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return torch.clamp(mag / max(command_threshold, 1e-6), 0.0, 1.0)


def _stance_target(
    env, period: float, swing_order: tuple[int, ...], swing_fraction: float, blend: float
) -> torch.Tensor:
    """Smoothed "this foot should be planted" target in [0, 1], shape [B, 5].

    A hard indicator makes the reward discontinuous at touch-down, which the
    policy games by hovering; `blend` (in cycle fractions) ramps it instead.
    """
    ph = _leg_phase(env, period, swing_order)
    b = max(blend, 1e-6)
    sf = swing_fraction - _BOUNDARY_EPS
    rise = torch.clamp((ph - (sf - b)) / b, 0.0, 1.0)                 # 0 -> 1 at touch-down
    fall = torch.clamp((ph - (1.0 - b)) / b, 0.0, 1.0)                # 1 -> 0 at lift-off
    return rise - fall


def gait_clock(env: ManagerBasedRlEnv, period: float = _DEFAULT.period) -> torch.Tensor:
    """Observation: (sin, cos) of the gait phase, shape [B, 2]."""
    ph = gait_phase(env, period) * 2.0 * torch.pi
    return torch.stack([torch.sin(ph), torch.cos(ph)], dim=-1)


def gait_contact_schedule(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    period: float = _DEFAULT.period,
    swing_order: tuple[int, ...] = _DEFAULT.swing_order,
    swing_fraction: float = _DEFAULT.swing_fraction,
    blend: float = 0.04,
    command_threshold: float = 0.02,
) -> torch.Tensor:
    """Reward matching the wave gait's contact schedule, shape [B].

    1.0 when every foot's contact state agrees with the schedule, 0.0 when all
    five disagree. Gated by command magnitude so standing still is rewarded for
    keeping all feet down.
    """
    sensor: "ContactSensor" = env.scene[sensor_name]
    assert sensor.data.found is not None
    in_contact = (sensor.data.found > 0).float()                        # [B, 5]

    target = _stance_target(env, period, swing_order, swing_fraction, blend)
    act = _activity(env, command_name, command_threshold)[:, None]
    target = 1.0 - act * (1.0 - target)                                 # stand still -> all planted

    match = target * in_contact + (1.0 - target) * (1.0 - in_contact)
    reward = match.mean(dim=1)

    env.extras["log"]["Metrics/gait_schedule_match"] = reward.mean()
    env.extras["log"]["Metrics/duty_factor"] = in_contact.mean()
    return reward


def gait_swing_clearance(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    command_name: str,
    period: float = _DEFAULT.period,
    swing_order: tuple[int, ...] = _DEFAULT.swing_order,
    swing_fraction: float = _DEFAULT.swing_fraction,
    step_height: float = _DEFAULT.step_height,
    foot_radius: float = 0.012,
    command_threshold: float = 0.02,
) -> torch.Tensor:
    """Cost on departing from the scheduled foot height profile, shape [B].

    The reference lifts a swinging foot on a raised sine of amplitude
    `step_height` and keeps planted feet on the ground. Use with a negative
    weight.
    """
    sensor = env.scene[height_sensor_name]
    heights = sensor.data.heights                                       # [B, 5]

    ph = _leg_phase(env, period, swing_order)
    s = torch.clamp(ph / swing_fraction, 0.0, 1.0)
    swinging = (ph < swing_fraction - _BOUNDARY_EPS).float()
    act = _activity(env, command_name, command_threshold)[:, None]

    target = foot_radius + act * swinging * step_height * torch.sin(torch.pi * s)
    cost = torch.sum(torch.square(heights - target) * swinging, dim=1)

    env.extras["log"]["Metrics/swing_height_err"] = torch.sqrt(
        torch.clamp((cost / torch.clamp(swinging.sum(dim=1), min=1.0)).mean(), min=0.0)
    )
    return cost
