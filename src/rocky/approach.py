"""Walking up to the cubes and knowing when to stop.

The locomotion controller -- the trained policy, or the analytic wave gait as a
stand-in -- takes a (vx, vy, wz) twist. This turns "the cubes are over there"
into that twist, and says when the robot is parked well enough to hand over to
the arm.

The stand-off it drives to is not arbitrary. Sweeping the scripted stack across
the workspace, the pick-and-place succeeds from 340 to 380 mm with the cube pair
within about 15 deg of straight ahead, and fails inside 320 mm, where the arm
folds far enough to bring the gripper body back over the cubes. So the target is
the middle of that band.

It drives to a POSE, not just a range and bearing. Parking anywhere on a circle
around the pair's midpoint satisfies "355 mm away, facing them", and some of
those parks leave one cube at 357 mm and the other at 385 mm -- the far one right
on the edge of what the arm can place at. Standing on the perpendicular bisector
of the two cubes makes the two ranges equal by construction, and turned the
scripted stack from a coin flip into something that mostly works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gait import WaveGait


@dataclass(frozen=True)
class ApproachParams:
    stand_off: float = 0.355
    """Body axis to the midpoint of the two cubes at handoff, m."""
    position_tol: float = 0.012
    """Close enough in position, m."""
    heading_tol: float = math.radians(4.0)
    """Close enough in bearing, rad."""
    turn_first: float = math.radians(25.0)
    """Past this bearing error, turn in place instead of walking."""
    k_range: float = 0.45
    """Position error -> body-frame speed, 1/s."""
    k_yaw: float = 1.5
    """Bearing error -> yaw rate, 1/s."""
    hold_s: float = 1.2
    """How long the robot has to stay inside both tolerances before handing over.

    A pentapod on a wave gait is never quite still: it arrives, then keeps
    shuffling for most of a cycle. Handing the arm a moving base is how the
    first grasp attempt ended up 20 mm off.
    """


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class Twist:
    vx: float
    vy: float
    wz: float
    pos_err: float
    yaw_err: float
    arrived: bool


class Approach:
    """Drives the base to a stand-off pose in front of a target point."""

    def __init__(self, params: ApproachParams | None = None, gait: WaveGait | None = None,
                 envelope_samples: int = 7):
        self.p = params or ApproachParams()
        self.gait = gait or WaveGait()
        self._inside_since: float | None = None
        # `WaveGait.clamp_command` bisects over a few hundred IK solves, which is
        # 15 ms -- fine once, ruinous at 200 Hz. Sample the envelope here instead
        # and interpolate it at run time.
        self.yaw_max = abs(self.gait.clamp_command(0.0, 0.0, 10.0)[2])
        self._dir_grid = np.linspace(-math.pi, math.pi, 2 * envelope_samples + 1)
        self._dir_max = np.array([
            float(np.hypot(*self.gait.clamp_command(10.0 * math.cos(a), 10.0 * math.sin(a), 0.0)[:2]))
            for a in self._dir_grid])

    def envelope(self, heading: float) -> float:
        """Largest speed the gait can hold travelling along a body-frame heading."""
        return float(np.interp(wrap_pi(heading), self._dir_grid, self._dir_max))

    def reset(self) -> None:
        self._inside_since = None

    def goal(self, base_xy, target_xy, axis=None) -> tuple[np.ndarray, float]:
        """Where to stand and which way to face.

        `axis` is the vector between the two cubes. Given it, the goal is on
        their perpendicular bisector; without it, straight back along the line
        from the robot to the target.
        """
        pos = np.asarray(base_xy, float)[:2]
        tgt = np.asarray(target_xy, float)[:2]
        if axis is None:
            n = tgt - pos
        else:
            a = np.asarray(axis, float)[:2]
            n = np.array([a[1], -a[0]])
            if float(n @ (tgt - pos)) < 0.0:
                n = -n
        norm = float(np.hypot(*n))
        if norm < 1e-9:
            n, norm = np.array([1.0, 0.0]), 1.0
        n = n / norm
        return tgt - self.p.stand_off * n, math.atan2(n[1], n[0])

    def command(self, t: float, base_pos, base_quat, target_xy, axis=None) -> Twist:
        pos = np.asarray(base_pos, float)[:2]
        yaw = yaw_of(base_quat)
        goal, want_yaw = self.goal(pos, target_xy, axis)

        world_err = goal - pos
        pos_err = float(np.hypot(*world_err))
        yaw_err = wrap_pi(want_yaw - yaw)
        # into the body frame: the gait is commanded in its own axes
        c, s = math.cos(yaw), math.sin(yaw)
        bx = c * world_err[0] + s * world_err[1]
        by = -s * world_err[0] + c * world_err[1]

        wz = float(np.clip(self.p.k_yaw * yaw_err, -self.yaw_max, self.yaw_max))
        v = self.p.k_range * np.array([bx, by])
        # A big heading error is worth fixing before travelling: walking while
        # badly misaligned just carves an arc.
        if abs(yaw_err) > self.p.turn_first:
            v[:] = 0.0
        speed = float(np.hypot(*v))
        if speed > 1e-9:
            v *= min(1.0, self.envelope(math.atan2(v[1], v[0])) / speed)

        inside = pos_err <= self.p.position_tol and abs(yaw_err) <= self.p.heading_tol
        if inside:
            if self._inside_since is None:
                self._inside_since = float(t)
        else:
            self._inside_since = None
        arrived = bool(inside and (float(t) - self._inside_since) >= self.p.hold_s)
        if arrived:
            return Twist(0.0, 0.0, 0.0, pos_err, yaw_err, True)
        return Twist(float(v[0]), float(v[1]), wz, pos_err, yaw_err, False)


def spawn_pose(rng, cubes_at: float = 0.355) -> tuple[np.ndarray, float]:
    """A random starting pose for the robot, facing roughly at the cubes.

    Matches the episode randomisation the stacking task uses: 0.6 to 1.2 m back
    along a bearing within 30 deg of the cubes, pointing within 25 deg of them.
    """
    dist = rng.uniform(0.6, 1.2)
    bearing = rng.uniform(-math.pi / 6.0, math.pi / 6.0)
    pos = np.array([-dist * math.cos(bearing), -dist * math.sin(bearing)])
    yaw = math.atan2(-pos[1], -pos[0]) + rng.uniform(-math.pi / 7.2, math.pi / 7.2)
    return pos, yaw
