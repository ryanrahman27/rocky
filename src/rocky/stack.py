"""Scripted pick-and-place: brace on four legs, stack the red cube on the blue.

This is the manipulation half of the handoff the locomotion policy sets up. The
policy walks Rocky to a stand-off pose in front of the cubes and then stops
issuing commands; this controller takes over, widens the four stance legs into a
brace, lifts leg 0 off the ground, and drives it through a waypoint sequence.

It is deliberately a plain function of time and the measured base pose. Every
waypoint is expressed in WORLD coordinates and converted into the body frame at
each control step using the base pose the simulator (or, on hardware, the state
estimator) reports, so sag and the small forward pitch the brace settles into are
corrected for rather than accumulated. The joint targets it emits are exactly
what a VLA policy will later have to produce, which is the point: this script is
the demonstrator that generates that policy's training data.

Phases, in order::

    brace     four stance legs widen and the body drops, leg 0 still planted
    lift_off  leg 0 leaves the ground and folds into the ready pose
    over_red  gripper travels to a hover above the red cube, jaws open
    descend   straight down onto the cube
    close     jaws squeeze
    lift      straight up to transit height
    transit   across to a hover above the blue cube
    place     down until the red cube's underside meets the blue cube's top
    release   jaws open
    retract   straight up, clear of the stack
    stow      arm folds back and the stance legs return to the walking footprint
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import arm
from . import kinematics as K
from . import model_params as P
from .gait import GaitParams, WaveGait


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class StackPlan:
    """Timing, clearances and the ready posture for the scripted sequence."""

    hover: float = 0.105
    """Gripper height above a cube's centre while travelling, m. Has to clear the
    40 mm cube the jaws are holding plus the finger pads below the wrist."""
    ready_radius: float = 0.30
    """Ready posture: how far the grasp point sits from the body axis, m."""
    ready_height: float = 0.140
    """Ready posture: grasp-point height above the floor, m. High enough that the
    Cartesian moves in and out of the workspace pass over the cubes rather than
    through them -- which is exactly how the first run punted the red cube a
    metre downrange."""
    grasp_rise: float = 0.008
    """How far above a cube's centre the grasp point sits, m.

    The gripper's body hull stops 17 mm short of the point between the pads, so
    a top-down grasp aimed at the centre of a 40 mm cube buries the hull 3 mm
    into the cube's top face and the descent stalls there. Grabbing 8 mm high
    puts the pads across the upper half of the cube and leaves the hull 5 mm
    clear. The same offset is added at the place, so the cube still lands where
    it is asked to."""
    squeeze: float = 0.002
    """How far past contact each jaw is commanded, m. The servo is a position
    source with a force limit, so the overshoot is what becomes grip force."""
    open_gap: float = arm.JAW_OPEN
    """Per-jaw travel when open; twice this is the opening, 60 mm."""
    stow_gap: float = 0.010
    """Per-jaw travel while the gripper is still a foot. A 60 mm-wide open
    gripper swinging along the floor is a hazard to anything it passes."""
    stack_gap: float = 0.001
    """Clearance left under the carried cube at the end of `place`, m."""
    trim_gain: float = 6.0
    """Integral trim on the four arm joints, 1/s.

    The servos are position sources with finite stiffness, so holding the arm
    out at a third of a metre leaves a standing error of tau/kp -- about 2.6 deg
    of sweep under the load of the arm plus a cube, which is 16 mm of placement
    error at that radius, enough to leave the red cube teetering on the corner of
    the blue one. Integrating the encoder error and adding it back is what the
    real servo bus would do for us; here it has to be done explicitly."""
    trim_limit: float = 0.12
    """Cap on that trim, rad."""
    trim_band: float = 0.09
    """Only integrate while the tracking error is under this, rad.

    A servo that is merely sagging is a few degrees behind. A servo pressing the
    cube it is carrying into the cube it is stacking on is tens of degrees
    behind and going nowhere, and integrating that just winds the command off
    into the scenery -- which is exactly what the first version did, driving the
    gripper 26 mm past the target and toppling the stack it had just built."""

    t_brace: float = 1.4
    t_lift_off: float = 1.0
    t_to_red: float = 1.3
    t_descend: float = 1.0
    t_close: float = 0.7
    t_lift: float = 1.0
    t_transit: float = 1.6
    t_place: float = 1.2
    t_settle: float = 0.6
    t_release: float = 0.5
    t_retract: float = 0.9
    t_to_ready: float = 1.3
    t_stow: float = 1.8

    walk: GaitParams = field(default_factory=GaitParams)
    brace: arm.BracePose = field(default_factory=arm.BracePose)


@dataclass
class Step:
    """One control step's output."""

    joints: dict[str, float]
    phase: str
    progress: float
    done: bool


#: Marker for a waypoint that is fixed in the body frame rather than the world.
BODY = "body"


class StackController:
    """Time-indexed joint targets for the brace-and-stack sequence.

    Construct once, `reset` per episode with the cubes' world positions, then
    call `step` every control tick with the measured base pose.
    """

    def __init__(self, plan: StackPlan | None = None):
        self.p = plan or StackPlan()
        self._walk = WaveGait(self.p.walk)
        self._brace_gait = WaveGait(GaitParams(
            period=self.p.walk.period,
            duty=self.p.walk.duty,
            swing_order=self.p.walk.swing_order,
            step_height=self.p.walk.step_height,
            stance_height=self.p.brace.height,
            footprint_scale=self.p.brace.spread,
        ))
        self.walk_stance = self._walk.neutral_joint_targets()      # (5, 3)
        self.brace_stance = self._brace_gait.neutral_joint_targets()
        # Leg 0 draws in rather than out: it is the arm, and the cubes are
        # sitting exactly where a widened gripper would sweep.
        self.brace_stance[P.MANIP_LEG] = self._tuck_leg0()
        #: Ready posture, in the body frame: gripper parked high and inboard.
        self.ready = np.array([self.p.ready_radius, 0.0,
                               self.p.ready_height - self.p.brace.height])
        self.reset((0.34, 0.06), (0.34, -0.06))

    def _tuck_leg0(self) -> np.ndarray:
        """Leg 0's planted pose at the end of the brace: same height, drawn in."""
        contact = K.home_contact_position(P.MANIP_LEG).copy()
        contact[:2] *= self.p.brace.arm_tuck
        contact[2] = -self.p.brace.height
        centre = contact + np.array([0.0, 0.0, P.FOOT_RADIUS[P.MANIP_LEG]])
        return K.inverse_kinematics(P.MANIP_LEG, K.body_to_leg(P.MANIP_LEG, centre))

    # -- episode setup -----------------------------------------------------
    def reset(self, red_xy, blue_xy, floor_z: float = 0.0) -> None:
        """Latch the cube positions for this episode. World frame, metres."""
        self.red = np.array([float(red_xy[0]), float(red_xy[1]), floor_z + arm.CUBE / 2])
        self.blue = np.array([float(blue_xy[0]), float(blue_xy[1]), floor_z + arm.CUBE / 2])
        self.floor_z = float(floor_z)
        self._approach_cache: dict[tuple, float] = {}
        self._segments = self._build()
        self._total = sum(seg[1] for seg in self._segments)
        self._trim = np.zeros(4)
        self._t_prev: float | None = None
        self._q_prev: np.ndarray | None = None

    def _build(self):
        """(name, duration, mode, payload) tuples. Waypoints are (frame, point)."""
        p = self.p
        W = "world"
        up = lambda c, h: (W, c + np.array([0.0, 0.0, h]))          # noqa: E731
        # Targets are grasp-point positions, which ride `grasp_rise` above the
        # cube centre they correspond to.
        red = up(self.red, p.grasp_rise)
        hover_red = up(self.red, p.hover)
        # The carried cube's centre once stacked: one cube edge above the blue
        # cube's centre, less the clearance we stop short by and let it drop.
        stacked_pt = self.blue + np.array([0.0, 0.0, arm.CUBE + p.stack_gap + p.grasp_rise])
        stacked = (W, stacked_pt)
        hover_stack = (W, self.blue + np.array([0.0, 0.0, arm.CUBE + p.hover]))
        ready = (BODY, self.ready)
        grip = p.open_gap - arm.CUBE / 2 - p.squeeze     # per-jaw travel when gripping

        return [
            ("brace",    p.t_brace,    "legs",  (self.walk_stance, self.brace_stance, p.stow_gap)),
            ("lift_off", p.t_lift_off, "unfold", (ready, p.stow_gap)),
            ("to_red",   p.t_to_red,   "cart",  (ready, hover_red, p.open_gap)),
            ("descend",  p.t_descend,  "cart",  (hover_red, red, p.open_gap)),
            ("close",    p.t_close,    "cart",  (red, red, grip)),
            ("lift",     p.t_lift,     "cart",  (red, hover_red, grip)),
            ("transit",  p.t_transit,  "cart",  (hover_red, hover_stack, grip)),
            ("place",    p.t_place,    "cart",  (hover_stack, stacked, grip)),
            ("settle",   p.t_settle,   "cart",  (stacked, stacked, grip)),
            ("release",  p.t_release,  "cart",  (stacked, stacked, p.open_gap)),
            ("retract",  p.t_retract,  "cart",  (stacked, hover_stack, p.open_gap)),
            ("to_ready", p.t_to_ready, "cart",  (hover_stack, ready, p.open_gap)),
            ("stow",     p.t_stow,     "fold",  (ready, p.stow_gap)),
        ]

    @property
    def duration(self) -> float:
        return self._total

    # -- the sequence ------------------------------------------------------
    def step(self, t: float, base_pos, base_quat, arm_q=None) -> Step:
        """One control tick.

        `arm_q` is the measured (sweep, lift, elbow, wrist) of leg 0. Supply it
        and the controller trims out the servos' standing position error; leave
        it out and the sequence runs purely open-loop on the joints.
        """
        name, s, mode, payload = self._locate(t)
        R = _quat_to_mat(base_quat)
        origin = np.asarray(base_pos, float)

        def body(wp):
            frame, pt = wp
            return np.asarray(pt, float) if frame == BODY else R.T @ (np.asarray(pt, float) - origin)

        legs = self.brace_stance.copy()
        stance_q = np.append(self.brace_stance[P.MANIP_LEG], P.HOME_WRIST)

        if mode == "legs":
            a, b, jaw = payload
            legs = a + (b - a) * _smoothstep(s)
            q_arm = np.append(legs[P.MANIP_LEG], P.HOME_WRIST)
        elif mode == "unfold":
            ready, jaw = payload
            q_ready = self._solve(body(ready))
            q_arm = stance_q + (q_ready - stance_q) * _smoothstep(s)
        elif mode == "cart":
            a, b, jaw = payload
            pa, pb = body(a), body(b)
            ap = self._approach(pa, pb, s)
            q_arm = arm.reach_for(pa + (pb - pa) * _smoothstep(s), ap, clip=True)
        else:                                   # fold: arm home, then legs narrow
            ready, jaw = payload
            q_ready = self._solve(body(ready))
            q_arm = q_ready + (stance_q - q_ready) * _smoothstep(min(1.0, s / 0.55))
            k = _smoothstep(max(0.0, s - 0.5) / 0.5)
            legs = self.brace_stance + (self.walk_stance - self.brace_stance) * k

        q_arm = self._trimmed(t, q_arm, arm_q, mode == "cart")
        return Step(self._joint_dict(legs, q_arm, jaw), name, s, t >= self._total)

    def _trimmed(self, t: float, q_cmd: np.ndarray, measured, active: bool) -> np.ndarray:
        """Add the integral trim, and update it from this tick's error.

        The trim is only for the Cartesian phases, where the gripper has to be
        somewhere specific. During the joint-space fold and unfold it bleeds
        away: several degrees of standing sweep offset carried into the stow
        walks the gripper sideways into whatever it just built.
        """
        dt = 0.0 if self._t_prev is None else max(0.0, float(t) - self._t_prev)
        self._t_prev = float(t)
        if not active:
            self._trim *= math.exp(-dt / 0.25) if dt > 0.0 else 1.0
            self._q_prev = q_cmd
            return np.clip(q_cmd + self._trim, arm._LO, arm._HI)
        if measured is not None and self._q_prev is not None and dt > 0.0:
            err = self._q_prev - np.asarray(measured, float)
            err = np.where(np.abs(err) < self.p.trim_band, err, 0.0)
            self._trim = np.clip(self._trim + self.p.trim_gain * dt * err,
                                 -self.p.trim_limit, self.p.trim_limit)
        self._q_prev = q_cmd
        return np.clip(q_cmd + self._trim, arm._LO, arm._HI)

    # -- helpers -----------------------------------------------------------
    def _locate(self, t: float):
        t = max(0.0, float(t))
        acc = 0.0
        for i, (name, dur, mode, payload) in enumerate(self._segments):
            if t < acc + dur or i == len(self._segments) - 1:
                return name, min(1.0, (t - acc) / dur), mode, payload
            acc += dur
        raise AssertionError

    def _node_approach(self, pt: np.ndarray) -> float:
        """Steepest gripper tilt the joints comfortably hold at one waypoint.

        Cached on the waypoint, so a point that ends one segment and starts the
        next gets the same answer and the wrist does not jump at the seam.
        """
        key = tuple(np.round(pt, 5))
        hit = self._approach_cache.get(key)
        if hit is None:
            hit = arm.approach_angle(arm.reach_for(pt, None, clip=True))
            self._approach_cache[key] = hit
        return hit

    def _approach(self, a: np.ndarray, b: np.ndarray, s: float) -> float:
        """Approach angle part-way along a segment.

        The target and the tilt ride the same smoothstep, so the gripper's
        orientation is continuous everywhere -- including across the seams, where
        the two segments share a waypoint and therefore a cached angle.
        """
        ap_a, ap_b = self._node_approach(a), self._node_approach(b)
        return ap_a + (ap_b - ap_a) * _smoothstep(s)

    def _solve(self, pt: np.ndarray) -> np.ndarray:
        return arm.reach_for(pt, self._node_approach(pt), clip=True)

    def _joint_dict(self, legs: np.ndarray, q_arm: np.ndarray, jaw: float) -> dict[str, float]:
        out: dict[str, float] = {"jaw_r": float(jaw), "jaw_l": float(jaw)}
        for k in range(P.N_LEGS):
            if k == P.MANIP_LEG:
                out["sweep_0"], out["lift_0"], out["elbow_0"], out["wrist_0"] = (
                    float(v) for v in q_arm)
            else:
                out[f"sweep_{k}"], out[f"lift_{k}"], out[f"elbow_{k}"] = (
                    float(v) for v in legs[k])
        return out
