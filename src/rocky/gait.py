"""Wave gait for a five-legged robot.

One foot swings at a time: the other four stay planted, so the support polygon
is always a quadrilateral containing the body axis with ~85 mm of margin. That
is the slow, statically stable crawl a five-legged machine does naturally, and
the first rung of Rocky's development ladder.

Swing order is the "star" sequence 0-2-4-1-3 rather than going round the body
in order. Consecutive swing legs then sit 144 deg apart instead of 72 deg, so
the support polygon shifts symmetrically about the body instead of walking
around it, which keeps the static margin much more even through the cycle.

Foot trajectories are planned as *ground contact points* in the body frame.
During stance a planted foot must travel backwards through the body frame at
exactly the commanded body velocity; during swing it lifts on a raised sine and
returns to the front of its stroke.

Everything here is numpy and has no mjlab or torch dependency, so the same gait
drives the open-loop MuJoCo demo, the unit tests, and the reference schedule the
RL reward is shaped against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import kinematics as K
from . import model_params as P


@dataclass(frozen=True)
class GaitParams:
    """Tunables for the wave gait. Defaults are the tuned values from scripts/play_gait.py."""

    period: float = 2.4
    """Seconds for one full cycle: every leg swings exactly once."""
    duty: float = 0.8
    """Fraction of the cycle a foot spends planted. 0.8 = one leg airborne at a time."""
    swing_order: tuple[int, ...] = (0, 2, 4, 1, 3)
    """Order legs take their swing slot."""
    step_height: float = 0.030
    """Peak foot lift above the contact plane, m."""
    stance_height: float = 0.065
    """Body origin above the contact plane, m. The CAD pose is 0.0447; the gait
    stands slightly taller and wider to free up elbow fold for the stride."""
    footprint_scale: float = 1.05
    """Scales the neutral footprint radius about the body axis. 1.05 puts the
    footprint at 582 mm, inside the 550-650 mm design envelope."""
    swing_lead: float = 0.0
    """Shifts every leg's phase; only useful for lining up against a recording."""

    def __post_init__(self) -> None:
        if sorted(self.swing_order) != list(range(P.N_LEGS)):
            raise ValueError(f"swing_order must be a permutation of 0..4, got {self.swing_order}")
        if not 0.0 < self.duty < 1.0:
            raise ValueError(f"duty must be in (0, 1), got {self.duty}")
        if self.period <= 0.0:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def swing_fraction(self) -> float:
        return 1.0 - self.duty

    @property
    def stance_time(self) -> float:
        return self.duty * self.period

    @property
    def swing_time(self) -> float:
        return self.swing_fraction * self.period

    @property
    def phase_offsets(self) -> np.ndarray:
        """Cycle fraction at which each leg starts its swing, indexed by leg."""
        off = np.zeros(P.N_LEGS)
        for slot, leg in enumerate(self.swing_order):
            off[leg] = slot / P.N_LEGS
        return (off + self.swing_lead) % 1.0


#: Safety back-off applied by `WaveGait.clamp_command`.
_CLAMP_SAFETY = 0.04


def _rotz2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class WaveGait:
    """Generates foot targets and joint angles for the wave gait."""

    def __init__(self, params: GaitParams | None = None):
        self.p = params or GaitParams()
        # Neutral contact points: the CAD stance footprint, re-scaled and set to
        # the commanded body height.
        neutral = np.stack([K.home_contact_position(k) for k in range(P.N_LEGS)])
        neutral[:, :2] *= self.p.footprint_scale
        neutral[:, 2] = -self.p.stance_height
        self.neutral = neutral

    # --- phase bookkeeping ------------------------------------------------
    def leg_phases(self, t: float) -> np.ndarray:
        """Per-leg phase in [0, 1): 0 = lift-off, `swing_fraction` = touch-down."""
        return ((t / self.p.period) - self.p.phase_offsets) % 1.0

    def in_swing(self, t: float) -> np.ndarray:
        return self.leg_phases(t) < self.p.swing_fraction

    def swing_progress(self, t: float) -> np.ndarray:
        """0..1 through the swing, 0 outside it."""
        ph = self.leg_phases(t)
        s = ph / self.p.swing_fraction
        return np.where(ph < self.p.swing_fraction, s, 0.0)

    # --- foot targets -----------------------------------------------------
    def _stance_point(self, leg: int, s: float, vx: float, vy: float, wz: float) -> np.ndarray:
        """Contact point for a planted foot, `s` in [0, 1] through its stance."""
        tst = self.p.stance_time
        ang = -wz * tst * (s - 0.5)
        xy = _rotz2(ang) @ self.neutral[leg, :2] - np.array([vx, vy]) * tst * (s - 0.5)
        return np.array([xy[0], xy[1], self.neutral[leg, 2]])

    def foot_targets(self, t: float, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> np.ndarray:
        """Contact-point targets for all five feet in the body frame, shape (5, 3)."""
        ph = self.leg_phases(t)
        sf = self.p.swing_fraction
        out = np.zeros((P.N_LEGS, 3))
        for k in range(P.N_LEGS):
            if ph[k] < sf:
                s = ph[k] / sf
                lift = self._stance_point(k, 1.0, vx, vy, wz)     # where it left the ground
                land = self._stance_point(k, 0.0, vx, vy, wz)     # where it will touch down
                xy = lift[:2] + (land[:2] - lift[:2]) * _smoothstep(s)
                z = self.neutral[k, 2] + self.p.step_height * math.sin(math.pi * s)
                out[k] = (xy[0], xy[1], z)
            else:
                out[k] = self._stance_point(k, (ph[k] - sf) / self.p.duty, vx, vy, wz)
        return out

    def joint_targets(
        self, t: float, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, clip: bool = True
    ) -> np.ndarray:
        """Joint angles for all legs, shape (5, 3): sweep, lift, elbow."""
        contacts = self.foot_targets(t, vx, vy, wz)
        q = np.zeros((P.N_LEGS, 3))
        for k in range(P.N_LEGS):
            centre = contacts[k] + np.array([0.0, 0.0, P.FOOT_RADIUS[k]])
            q[k] = K.inverse_kinematics(k, K.body_to_leg(k, centre), clip=clip)
        return q

    def action_vector(self, t: float, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> np.ndarray:
        """The 16 joint targets in model order, wrist held at zero."""
        return K.joint_vector(self.joint_targets(t, vx, vy, wz), wrist=P.HOME_WRIST)

    def neutral_joint_targets(self) -> np.ndarray:
        """Joint angles with all five feet planted at their neutral stance, (5, 3)."""
        q = np.zeros((P.N_LEGS, 3))
        for k in range(P.N_LEGS):
            centre = self.neutral[k] + np.array([0.0, 0.0, P.FOOT_RADIUS[k]])
            q[k] = K.inverse_kinematics(k, K.body_to_leg(k, centre))
        return q

    def neutral_joint_dict(self, wrist: float = P.HOME_WRIST) -> dict[str, float]:
        """Neutral stance as {joint_name: angle}, for an RL init state."""
        q = self.neutral_joint_targets()
        out: dict[str, float] = {}
        for k in range(P.N_LEGS):
            out[f"sweep_{k}"], out[f"lift_{k}"], out[f"elbow_{k}"] = (float(v) for v in q[k])
            if k == P.MANIP_LEG:
                out[f"wrist_{k}"] = float(wrist)
        return out

    # --- feasibility ------------------------------------------------------
    def stride_length(self, speed: float) -> float:
        return abs(speed) * self.p.stance_time

    def check_feasible(self, vx: float, vy: float, wz: float, samples: int = 256) -> dict:
        """Walk one cycle and report reach, joint-limit and servo-speed headroom.

        Returns the worst-case joint excursions and speeds over the cycle. A gait
        is feasible when `within_limits` is True and `max_joint_speed` is
        comfortably under the servo's no-load speed.
        """
        ts = np.linspace(0.0, self.p.period, samples, endpoint=False)
        qs, ok = [], True
        for t in ts:
            try:
                qs.append(self.joint_targets(t, vx, vy, wz, clip=False))
            except K.Unreachable:
                ok = False
                qs.append(self.joint_targets(t, vx, vy, wz, clip=True))
        qs = np.stack(qs)
        dt = self.p.period / samples
        dq = np.abs(np.diff(np.concatenate([qs, qs[:1]]), axis=0)) / dt
        return {
            "within_limits": ok,
            "max_joint_speed": float(dq.max()),
            "speed_margin": float(P.VELOCITY_LIMIT / max(dq.max(), 1e-9)),
            "sweep_range_deg": float(np.degrees(np.ptp(qs[:, :, 0]))),
            "lift_range_deg": float(np.degrees(np.ptp(qs[:, :, 1]))),
            "elbow_range_deg": float(np.degrees(np.ptp(qs[:, :, 2]))),
            "stride": self.stride_length(math.hypot(vx, vy)),
        }

    def clamp_command(
        self, vx: float, vy: float, wz: float, servo_margin: float = 1.2
    ) -> tuple[float, float, float]:
        """Scale a velocity command down until the gait can actually execute it.

        Translation and rotation compete for the same foot workspace: at cruise
        the gait has almost no yaw authority left, while on the spot it can turn
        at ~0.4 rad/s. Rather than clipping joints (which distorts the foot path
        and makes the robot fight itself), scale the whole command uniformly so
        the gait stays geometrically consistent.
        """
        def ok(s: float, samples: int = 96) -> bool:
            r = self.check_feasible(vx * s, vy * s, wz * s, samples=samples)
            return bool(r["within_limits"]) and r["speed_margin"] >= servo_margin

        if ok(1.0):
            return vx, vy, wz
        lo, hi = 0.0, 1.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        # Bisection lands exactly on the boundary, where a different time sample
        # can still tip over a joint limit. Back off a little.
        lo *= 1.0 - _CLAMP_SAFETY
        return vx * lo, vy * lo, wz * lo

    def max_speed(self, direction: tuple[float, float] = (1.0, 0.0), servo_margin: float = 2.0) -> float:
        """Largest speed along `direction` that stays reachable and within
        `servo_margin`x of the servo's no-load speed. Bisection, m/s."""
        d = np.array(direction, float)
        d /= max(np.linalg.norm(d), 1e-9)

        def feasible(v: float) -> bool:
            r = self.check_feasible(v * d[0], v * d[1], 0.0, samples=96)
            return bool(r["within_limits"]) and r["speed_margin"] >= servo_margin

        lo, hi = 0.0, 1.0
        if feasible(hi):
            return hi
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        return lo


#: Nominal cruise: a 100 mm stride over the stance phase. Open loop the robot
#: reaches about 78% of this, the rest being absorbed by servo compliance; see
#: docs/gait.md.
CRUISE_STRIDE = 0.10
CRUISE_SPEED = CRUISE_STRIDE / (GaitParams().duty * GaitParams().period)


def support_margin(contacts: np.ndarray, in_swing: np.ndarray, com_xy=(0.0, 0.0)) -> float:
    """Distance from `com_xy` to the nearest edge of the planted feet's convex
    hull, in metres. Negative means the point is outside the support polygon."""
    pts = contacts[~in_swing][:, :2]
    if len(pts) < 3:
        return -np.inf
    c = np.asarray(com_xy, float)
    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    hull = pts[order]
    best = np.inf
    inside = True
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        n = np.array([-e[1], e[0]])
        n /= max(np.linalg.norm(n), 1e-12)
        d = float(n @ (c - a))
        if d < 0:
            inside = False
        best = min(best, abs(d))
    return best if inside else -best
