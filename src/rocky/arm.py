"""Manipulator-arm kinematics for leg 0.

Leg 0 is both a leg and an arm. Walking, it is one of five limbs; manipulating,
the other four brace and it lifts off to become a 4-DOF arm (sweep, lift, elbow,
wrist) with a 1-DOF parallel gripper.

The chain, in the leg-0 frame (origin on the sweep axis, +X radial, see
`rocky.kinematics.body_to_leg`), with every angle measured the way
`rocky.kinematics` measures it:

    sweep (about leg Z)  ->  coxa 42 mm
    lift  (about leg Y)  ->  femur 130 mm
    elbow (about leg Y)  ->  tibia 90 mm to the wrist axis
    wrist (about leg Y)  ->  67 mm to the point between the closed finger pads

lift, elbow and wrist all turn about the same axis, so the arm is planar inside
the swept vertical plane: two of those three place the grasp point and the third
sets the approach angle -- the direction the gripper's own +X points, measured in
that plane, positive up. -90 deg is straight down onto the floor.

The wrist's +-75 deg range is what makes the approach angle interesting. A
straight-down grasp is available out to about 360 mm from the body axis and then
runs out, so `inverse` picks the steepest approach the joints actually allow
instead of insisting on one that is only sometimes legal.

All target points are in the BODY frame, whose origin is the root body: a cube
sitting on the floor with the robot braced at 58 mm has its centre at
z = 0.020 - 0.058 = -0.038.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import model_params as P
from .kinematics import body_to_leg, leg_to_body

FINGER_X = 0.0675     # wrist axis -> point between the closed finger pads, m
GRASP_Y = 0.0076      # gripper centreline offset from the swept plane, m
JAW_OPEN = 0.030      # per-jaw travel; 60 mm total opening, m
JAW_CLOSED = 0.0
CUBE = 0.040          # cube edge, m

#: Where the gripper would rather point, and how far from a joint limit a
#: solution has to sit before `inverse` will call it comfortable.
PREFERRED_APPROACH = -math.pi / 2
APPROACH_MARGIN = math.radians(4.0)
_APPROACH_GRID = np.radians(np.arange(-90.0, 46.0, 1.0))

ARM = P.MANIP_LEG     # 0

_LO = np.array([P.SWEEP_RANGE[0], P.LIFT_RANGE[0], P.elbow_range(ARM)[0], P.WRIST_RANGE[0]])
_HI = np.array([P.SWEEP_RANGE[1], P.LIFT_RANGE[1], P.elbow_range(ARM)[1], P.WRIST_RANGE[1]])


class Unreachable(ValueError):
    pass


def forward(sweep: float, lift: float, elbow: float, wrist: float) -> np.ndarray:
    """(sweep, lift, elbow, wrist) -> grasp point in the body frame.

    The grasp point lies in the swept plane; the jaws straddle a line
    `GRASP_Y` to the side of it, so what the fingers close on is
    `forward(...) + grasp_offset(sweep)`.
    """
    a = -lift
    b = a - elbow
    c = b - wrist
    r = (P.COXA_LEN + P.FEMUR_LEN * math.cos(a)
         + P.WRIST_X * math.cos(b) + FINGER_X * math.cos(c))
    z = (P.FEMUR_LEN * math.sin(a) + P.WRIST_X * math.sin(b) + FINGER_X * math.sin(c))
    return leg_to_body(ARM, np.array([r * math.cos(sweep), r * math.sin(sweep), z]))


def grasp_offset(sweep: float) -> np.ndarray:
    """Body-frame offset from the swept plane to the jaws' closing centre.

    The jaws mirror about y = +7.6 mm in the gripper frame and slide along the
    gripper's own Y, which stays perpendicular to the swept plane whatever the
    lift/elbow/wrist do. Ignoring this offset puts the cube against one finger
    instead of between both.
    """
    return leg_to_body(ARM, np.array([-GRASP_Y * math.sin(sweep),
                                      GRASP_Y * math.cos(sweep), 0.0])) - P.mount_position(ARM)


def approach_angle(q: np.ndarray) -> float:
    """The approach angle a pose is actually holding."""
    return -float(q[1]) - float(q[2]) - float(q[3])


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _branch(radius: float, sweep: float, z: float, approach: float) -> np.ndarray | None:
    """Knee-up two-link solution for one signed radius, ignoring joint limits."""
    # Back the gripper out along the approach direction; what is left is the
    # familiar two-link reach to the wrist axis.
    wr = radius - P.COXA_LEN - FINGER_X * math.cos(approach)
    wz = z - FINGER_X * math.sin(approach)
    a, b = P.FEMUR_LEN, P.WRIST_X
    d = math.hypot(wr, wz)
    if not (abs(a - b) + 1e-9 < d < a + b - 1e-9):
        return None
    elbow = math.acos(float(np.clip((d * d - a * a - b * b) / (2.0 * a * b), -1.0, 1.0)))
    alpha = math.atan2(wz, wr) + math.atan2(b * math.sin(elbow), a + b * math.cos(elbow))
    return np.array([sweep, -alpha, elbow, alpha - elbow - approach])


def _solve_fixed(target: np.ndarray, approach: float) -> np.ndarray | None:
    """Joint vector for one approach angle. Prefers a solution inside the limits.

    Both sweep branches are tried. A folded arm can put the gripper behind its
    own sweep axis, which a single atan2 places 180 deg away and the limit check
    then rejects -- the same trap `rocky.kinematics` has to step around.
    """
    p = body_to_leg(ARM, target)
    radius = math.hypot(p[0], p[1])
    theta = math.atan2(p[1], p[0])
    outward = _branch(radius, theta, p[2], approach)
    inward = _branch(-radius, _wrap_pi(theta + math.pi), p[2], approach)
    for q in (outward, inward):
        if q is not None and np.all(q >= _LO - 1e-9) and np.all(q <= _HI + 1e-9):
            return q
    return outward if outward is not None else inward


def _margin(q: np.ndarray) -> float:
    return float(np.min(np.minimum(q - _LO, _HI - q)))


def inverse(target, approach: float | None = None, clip: bool = False) -> np.ndarray:
    """(sweep, lift, elbow, wrist) putting the grasp point at `target`.

    With `approach` given, that angle is held exactly and the pose is rejected if
    it breaks a joint limit. With `approach` None -- the default -- the solver
    scans the approach angles the arm can legally hold at this target and takes
    the steepest one that keeps every joint `APPROACH_MARGIN` clear of its stop,
    falling back to whichever legal angle has the most room.
    """
    t = np.asarray(target, float)

    if approach is not None:
        q = _solve_fixed(t, approach)
        if q is None:
            if not clip:
                raise Unreachable(f"target {t} out of the arm's reach at approach "
                                  f"{math.degrees(approach):.0f} deg")
            return np.clip(_nearest(t, approach), _LO, _HI)
        if clip:
            return np.clip(q, _LO, _HI)
        bad = [n for n, v, lo, hi in zip("sweep lift elbow wrist".split(), q, _LO, _HI)
               if v < lo - 1e-9 or v > hi + 1e-9]
        if bad:
            raise Unreachable(f"{', '.join(bad)} outside limits for target {t} "
                              f"at approach {math.degrees(approach):.0f} deg")
        return q

    best = None        # (margin, |approach - preferred|, q)
    roomiest = None
    for ap in _APPROACH_GRID:
        q = _solve_fixed(t, float(ap))
        if q is None:
            continue
        mrg = _margin(q)
        if roomiest is None or mrg > roomiest[0]:
            roomiest = (mrg, q)
        if mrg >= APPROACH_MARGIN:
            key = abs(float(ap) - PREFERRED_APPROACH)
            if best is None or key < best[0]:
                best = (key, q)
    if best is not None:
        return best[1]
    if roomiest is not None and roomiest[0] >= 0.0:
        return roomiest[1]
    if clip:
        return np.clip(roomiest[1] if roomiest else _nearest(t, PREFERRED_APPROACH), _LO, _HI)
    raise Unreachable(f"no approach angle reaches {t} within the joint limits")


def _nearest(target: np.ndarray, approach: float) -> np.ndarray:
    """Best effort when the target is out of reach: aim along the same bearing."""
    p = body_to_leg(ARM, np.asarray(target, float))
    sweep = math.atan2(p[1], p[0])
    return np.array([sweep, 0.0, P.elbow_range(ARM)[0], 0.0])


def reach_for(target, approach: float | None = None, clip: bool = False) -> np.ndarray:
    """Joints that close the jaws' centre -- not the swept plane -- on `target`.

    One correction pass is enough: the offset only moves the answer by the few
    tenths of a degree of sweep that 7.6 mm subtends at a third of a metre.
    """
    t = np.asarray(target, float)
    q = inverse(t, approach, clip=True)
    for _ in range(3):
        q_new = inverse(t - grasp_offset(q[0]), approach, clip=clip)
        if np.allclose(q_new, q, atol=1e-9):
            return q_new
        q = q_new
    return q


def reachable(target, approach: float | None = None) -> bool:
    try:
        reach_for(target, approach)
        return True
    except Unreachable:
        return False


@dataclass(frozen=True)
class BracePose:
    """The four stance legs widen and settle before leg 0 lifts off.

    With the manipulator airborne the support polygon loses a corner, and the arm
    then swings a cube around outside what is left of it. Widening the other four
    trades a little servo torque for stability margin.

    Leg 0 goes the other way. It is about to leave the ground, and widening it
    with the rest walks the open gripper straight into the cubes -- which, the
    first time this ran, sent the red one a metre downrange. So it draws inboard
    instead and lifts from there.
    """
    spread: float = 1.18        # footprint scale of the four stance legs
    arm_tuck: float = 0.88      # footprint scale of leg 0 while it is still planted
    height: float = 0.058       # body height, a little lower than walking
    settle_s: float = 1.2


BRACE = BracePose()
