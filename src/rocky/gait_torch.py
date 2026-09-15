"""The wave gait, batched in torch, for use inside RL reward terms.

`rocky.gait` is the readable numpy reference and the thing the open-loop demo
runs; this is the same maths vectorised over environments so a reward term can
ask "what would the scripted gait be doing right now, at this env's commanded
velocity?" every step.

The two implementations are checked against each other in
`tests/test_gait_torch.py` -- if they ever disagree by more than a micro-radian
that test fails, because a reference the policy is rewarded for matching is
worthless if it has drifted from the gait it claims to be.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import model_params as P
from .gait import GaitParams, WaveGait


class TorchGait:
    """Batched reference joint targets for the wave gait."""

    def __init__(self, params: GaitParams | None = None, device: torch.device | str = "cpu"):
        self.p = params or GaitParams()
        ref = WaveGait(self.p)
        d = torch.device(device)
        f = torch.float32

        self.neutral = torch.tensor(ref.neutral, dtype=f, device=d)                 # [5, 3]
        self.offsets = torch.tensor(ref.p.phase_offsets, dtype=f, device=d)         # [5]
        self.mount = torch.tensor(
            np.stack([P.mount_position(k) for k in range(P.N_LEGS)]), dtype=f, device=d
        )                                                                            # [5, 3]
        self.mount_ang = torch.tensor(P.MOUNT_ANGLES, dtype=f, device=d)             # [5]
        self.shin = torch.tensor(P.SHIN_LEN, dtype=f, device=d)                      # [5]
        self.foot_r = torch.tensor(P.FOOT_RADIUS, dtype=f, device=d)                 # [5]
        self.lo = torch.tensor(
            [[P.SWEEP_RANGE[0], P.LIFT_RANGE[0], P.elbow_range(k)[0]] for k in range(P.N_LEGS)],
            dtype=f, device=d,
        )
        self.hi = torch.tensor(
            [[P.SWEEP_RANGE[1], P.LIFT_RANGE[1], P.elbow_range(k)[1]] for k in range(P.N_LEGS)],
            dtype=f, device=d,
        )
        self.device = d
        # index of each leg's (sweep, lift, elbow) in the 16-joint model order
        self.joint_index = torch.tensor(
            [[P.JOINT_NAMES.index(f"{g}_{k}") for g in ("sweep", "lift", "elbow")]
             for k in range(P.N_LEGS)],
            dtype=torch.long, device=d,
        )
        self.wrist_index = P.JOINT_NAMES.index("wrist_0")

    def leg_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """[B] global phase -> [B, 5] per-leg phase in [0, 1)."""
        return torch.remainder(phase[:, None] - self.offsets[None, :], 1.0)

    def foot_targets(self, phase: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """Contact-point targets in the body frame, [B, 5, 3]."""
        ph = self.leg_phase(phase)
        sf = self.p.swing_fraction
        tst = self.p.stance_time
        vx, vy, wz = cmd[:, 0:1], cmd[:, 1:2], cmd[:, 2:3]                           # [B, 1]

        def stance_point(s: torch.Tensor) -> torch.Tensor:
            ang = -wz * tst * (s - 0.5)                                              # [B, 5]
            c, sn = torch.cos(ang), torch.sin(ang)
            nx, ny = self.neutral[None, :, 0], self.neutral[None, :, 1]
            x = c * nx - sn * ny - vx * tst * (s - 0.5)
            y = sn * nx + c * ny - vy * tst * (s - 0.5)
            return torch.stack([x, y], dim=-1)                                       # [B, 5, 2]

        ones = torch.ones_like(ph)
        lift_xy = stance_point(ones)          # where it left the ground
        land_xy = stance_point(ones * 0.0)    # where it will touch down
        st_xy = stance_point(torch.clamp((ph - sf) / self.p.duty, 0.0, 1.0))

        s = torch.clamp(ph / sf, 0.0, 1.0)
        blend = (s * s * (3.0 - 2.0 * s))[..., None]
        sw_xy = lift_xy + (land_xy - lift_xy) * blend

        swinging = (ph < sf)[..., None]
        xy = torch.where(swinging, sw_xy, st_xy)
        z = self.neutral[None, :, 2] + torch.where(
            swinging[..., 0], self.p.step_height * torch.sin(math.pi * s), torch.zeros_like(s)
        )
        return torch.cat([xy, z[..., None]], dim=-1)

    def joint_targets(self, phase: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """Reference joint angles in model order, [B, 16]."""
        contacts = self.foot_targets(phase, cmd)
        centre = contacts + torch.stack(
            [torch.zeros_like(self.foot_r), torch.zeros_like(self.foot_r), self.foot_r], dim=-1
        )[None]

        # body frame -> leg frame
        rel = centre - self.mount[None]
        ca, sa = torch.cos(-self.mount_ang), torch.sin(-self.mount_ang)
        x = ca[None, :] * rel[..., 0] - sa[None, :] * rel[..., 1]
        y = sa[None, :] * rel[..., 0] + ca[None, :] * rel[..., 1]
        z = rel[..., 2]

        # two-link IK, knee-up branch
        q1 = torch.atan2(y, x)
        rr = torch.hypot(x, y) - P.COXA_LEN
        a = P.FEMUR_LEN
        b = self.shin[None, :]
        d = torch.hypot(rr, z)
        cos_q3 = torch.clamp((d * d - a * a - b * b) / (2.0 * a * b), -1.0, 1.0)
        q3 = torch.acos(cos_q3)
        alpha = torch.atan2(z, rr) + torch.atan2(b * torch.sin(q3), a + b * torch.cos(q3))
        q2 = -alpha

        q = torch.stack([q1, q2, q3], dim=-1)                                        # [B, 5, 3]
        q = torch.clamp(q, self.lo[None], self.hi[None])

        out = torch.zeros(q.shape[0], len(P.JOINT_NAMES), device=q.device, dtype=q.dtype)
        out[:, self.joint_index.reshape(-1)] = q.reshape(q.shape[0], -1)
        out[:, self.wrist_index] = P.HOME_WRIST
        return out
