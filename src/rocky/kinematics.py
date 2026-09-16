"""Analytic forward and inverse kinematics for one Rocky leg.

The end effector is the centre of the leg's foot contact sphere; the ground
contact point is FOOT_RADIUS below it in world Z, whatever the leg's pose,
because the contact geom is a sphere. Plan contact points, add the radius, and
solve for the sphere centre.

With the wrist held at zero the manipulator leg is kinematically a standard leg
with a slightly longer shin, so one solver covers all five.
"""

from __future__ import annotations

import math

import numpy as np

from . import model_params as P


class Unreachable(ValueError):
    """Raised when a foot target lies outside the leg's reachable workspace."""


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_leg(leg: int, p_body: np.ndarray) -> np.ndarray:
    """Body-frame point -> leg frame (origin on the sweep axis, +X radial)."""
    return _rotz(-P.MOUNT_ANGLES[leg]) @ (np.asarray(p_body, float) - P.mount_position(leg))


def leg_to_body(leg: int, p_leg: np.ndarray) -> np.ndarray:
    return _rotz(P.MOUNT_ANGLES[leg]) @ np.asarray(p_leg, float) + P.mount_position(leg)


def forward_kinematics(leg: int, q: np.ndarray) -> np.ndarray:
    """(sweep, lift, elbow) -> foot sphere centre in the leg frame."""
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    a = P.FEMUR_LEN
    b = float(P.SHIN_LEN[leg])
    # Angles above the horizontal: rotation about +Y swings +X down, hence the sign.
    alpha = -q2
    beta = alpha - q3
    r = P.COXA_LEN + a * math.cos(alpha) + b * math.cos(beta)
    z = a * math.sin(alpha) + b * math.sin(beta)
    return np.array([r * math.cos(q1), r * math.sin(q1), z])


def _planar_ik(a: float, b: float, rr: float, zz: float) -> tuple[float, float] | None:
    """Two-link IK in the leg's vertical plane, knee-up branch.

    `rr` is signed: negative means the foot is behind the sweep axis.
    Returns (lift, elbow), or None if the target is out of reach.
    """
    d = math.hypot(rr, zz)
    if not (abs(a - b) + 1e-9 < d < a + b - 1e-9):
        return None
    q3 = math.acos(float(np.clip((d * d - a * a - b * b) / (2.0 * a * b), -1.0, 1.0)))
    alpha = math.atan2(zz, rr) + math.atan2(b * math.sin(q3), a + b * math.cos(q3))
    return -alpha, q3


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def inverse_kinematics(leg: int, p_leg: np.ndarray, clip: bool = False) -> np.ndarray:
    """Foot sphere centre in the leg frame -> (sweep, lift, elbow).

    Picks the knee-up branch (positive elbow), the only one the hardware allows.
    Both sweep branches are considered: a foot can sit behind the sweep axis --
    the manipulator limb folding back over the body does exactly that -- which a
    single atan2 would place 180 deg away and reject.

    Raises Unreachable if no branch is in reach and within the joint limits,
    unless `clip`, which clamps the best candidate to the nearest legal values.
    """
    p = np.asarray(p_leg, float)
    a = P.FEMUR_LEN
    b = float(P.SHIN_LEN[leg])
    radius = math.hypot(p[0], p[1])
    theta = math.atan2(p[1], p[0])

    lo = np.array([P.SWEEP_RANGE[0], P.LIFT_RANGE[0], P.elbow_range(leg)[0]])
    hi = np.array([P.SWEEP_RANGE[1], P.LIFT_RANGE[1], P.elbow_range(leg)[1]])

    candidates: list[np.ndarray] = []
    for signed_radius, q1 in ((radius, theta), (-radius, _wrap_pi(theta + math.pi))):
        sol = _planar_ik(a, b, signed_radius - P.COXA_LEN, p[2])
        if sol is None:
            continue
        candidates.append(np.array([q1, sol[0], sol[1]]))

    if not candidates:
        d = math.hypot(radius - P.COXA_LEN, p[2])
        if clip:
            # Aim at the closest point in reach along the same direction.
            return np.clip(np.array([theta, 0.0, P.elbow_range(leg)[0]]), lo, hi)
        raise Unreachable(
            f"leg {leg}: |target| {d:.4f} m outside reach [{abs(a - b):.4f}, {a + b:.4f}]"
        )

    for q in candidates:   # first is the outward branch, the normal walking pose
        if np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9):
            return q

    if clip:
        return np.clip(candidates[0], lo, hi)
    q = candidates[0]
    bad = [n for n, v, l, h in zip("sweep lift elbow".split(), q, lo, hi) if v < l - 1e-9 or v > h + 1e-9]
    raise Unreachable(f"leg {leg}: {', '.join(bad)} outside joint limits for target {p}")


def home_foot_position(leg: int) -> np.ndarray:
    """Foot sphere centre in the body frame at the CAD standing pose."""
    return leg_to_body(leg, forward_kinematics(leg, np.array([P.HOME_SWEEP, P.HOME_LIFT, P.HOME_ELBOW])))


def home_contact_position(leg: int) -> np.ndarray:
    """Ground contact point in the body frame at the CAD standing pose."""
    p = home_foot_position(leg)
    return p - np.array([0.0, 0.0, P.FOOT_RADIUS[leg]])


def joint_vector(q_legs: np.ndarray, wrist: float = 0.0) -> np.ndarray:
    """(5, 3) per-leg joints -> the 16-vector in model joint order."""
    out = []
    for k in range(P.N_LEGS):
        out.extend(q_legs[k].tolist())
        if k == P.MANIP_LEG:
            out.append(wrist)
    return np.array(out)


def limb_points(q_legs: np.ndarray) -> np.ndarray:
    """Knee and foot of every leg, in the body frame, from the joint encoders.

    Used to tell the sonar which of its returns are the robot's own limbs. A
    swinging leg travels straight through the gap the emitters fire down, and
    without this every stride throws off a phantom object.
    """
    out = []
    for k in range(P.N_LEGS):
        q = np.asarray(q_legs[k], float)
        foot = leg_to_body(k, forward_kinematics(k, q))
        knee_leg = np.array([
            (P.COXA_LEN + P.FEMUR_LEN * math.cos(-q[1])) * math.cos(q[0]),
            (P.COXA_LEN + P.FEMUR_LEN * math.cos(-q[1])) * math.sin(q[0]),
            P.FEMUR_LEN * math.sin(-q[1]),
        ])
        out.append(leg_to_body(k, knee_leg))
        out.append(foot)
    return np.stack(out)
