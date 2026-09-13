"""Geometry, joint limits and actuator constants for Rocky.

Single source of truth for everything the gait and the RL task need to know
about the robot. Values come from the Fusion 360 master model; `tests/test_model.py`
asserts every one of them against model/rocky.xml, so this file cannot drift
from the MJCF without a test failing.

Frames
------
Body frame: origin at the centre of the chassis' bottom face, Z up, +X through
leg 0 (the manipulator limb). Leg k is mounted at MOUNT_ANGLES[k].

Leg frame: origin on the leg's sweep axis, +X radially outward, +Y tangential
(mount angle + 90 deg), Z up. Within a leg, sweep turns about Z and both lift
and elbow turn about Y.

Joint sign convention: zero is the straight leg -- femur pointing radially
outward and horizontal, tibia in line with it. Positive rotation about +Y
swings +X downward, so the standing pose is lift = -50 deg (femur up),
elbow = +122 deg (tibia down and out).
"""

from __future__ import annotations

import math

import numpy as np

# --- leg mounting ---------------------------------------------------------
N_LEGS = 5
MANIP_LEG = 0                       # the manipulator limb, on +X
MOUNT_ANGLES = np.array([0.0, 72.0, 144.0, -144.0, -72.0]) * math.pi / 180.0
SHOULDER_RADIUS = 0.10112712429686843   # body axis -> sweep axis, m
SHOULDER_Z = 0.022                      # height of the sweep/lift axes, m

# --- link lengths ---------------------------------------------------------
COXA_LEN = 0.042        # sweep axis -> lift axis
FEMUR_LEN = 0.130       # lift axis -> elbow axis
TIBIA_LEN = 0.170       # elbow axis -> tip of the structural tibia
WRIST_X = 0.090         # elbow axis -> wrist axis (manipulator leg only)

# Foot contact spheres, as centre offset along the last link's +X and radius.
FOOT_SPHERE_STD = (0.16226, 0.012)      # on the tibia
FOOT_SPHERE_MANIP = (0.07436, 0.010)    # on the gripper, past the wrist

# Effective elbow -> foot-centre length per leg, with the wrist held at zero.
SHIN_LEN = np.array(
    [WRIST_X + FOOT_SPHERE_MANIP[0]] + [FOOT_SPHERE_STD[0]] * 4
)
FOOT_RADIUS = np.array([FOOT_SPHERE_MANIP[1]] + [FOOT_SPHERE_STD[1]] * 4)

# --- joint limits (rad) ---------------------------------------------------
SWEEP_RANGE = (math.radians(-45.0), math.radians(45.0))
LIFT_RANGE = (math.radians(-80.0), math.radians(85.0))
ELBOW_RANGE = (math.radians(10.0), math.radians(130.0))
ELBOW_RANGE_MANIP = (math.radians(10.0), math.radians(135.0))
WRIST_RANGE = (math.radians(-75.0), math.radians(75.0))

def elbow_range(leg: int) -> tuple[float, float]:
    return ELBOW_RANGE_MANIP if leg == MANIP_LEG else ELBOW_RANGE

# --- home pose ------------------------------------------------------------
HOME_LIFT = math.radians(-50.0)
HOME_ELBOW = math.radians(122.0)
HOME_SWEEP = 0.0
HOME_WRIST = 0.0
BASE_HEIGHT = 0.04503       # base origin above the ground in the home pose, m

# --- actuators: Feetech STS3215 at 12 V ------------------------------------
EFFORT_LIMIT = 2.9          # N m
VELOCITY_LIMIT = 4.7        # rad/s
ROTOR_INERTIA = 5.0e-8      # kg m^2 (assumed; no published figure)
GEAR_RATIO = 345.0
ARMATURE = ROTOR_INERTIA * GEAR_RATIO**2
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0
KP = ARMATURE * NATURAL_FREQ**2
KV = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ
FRICTIONLOSS = 0.05

TOTAL_MASS = 4.170          # kg, including 0.43 kg declared payload

# --- names ----------------------------------------------------------------
BASE_BODY = "base"
JOINT_NAMES: tuple[str, ...] = tuple(
    n
    for k in range(N_LEGS)
    for n in (
        (f"sweep_{k}", f"lift_{k}", f"elbow_{k}", f"wrist_{k}")
        if k == MANIP_LEG
        else (f"sweep_{k}", f"lift_{k}", f"elbow_{k}")
    )
)
FOOT_SITES = tuple(f"foot_{k}" for k in range(N_LEGS))
FOOT_GEOMS = tuple(f"foot_{k}_collision" for k in range(N_LEGS))

SWEEP_RE = r"^sweep_[0-4]$"
LIFT_RE = r"^lift_[0-4]$"
ELBOW_RE = r"^elbow_[0-4]$"
WRIST_RE = r"^wrist_0$"


def mount_position(leg: int) -> np.ndarray:
    """Sweep-axis position of a leg, in the body frame."""
    a = MOUNT_ANGLES[leg]
    return np.array([SHOULDER_RADIUS * math.cos(a), SHOULDER_RADIUS * math.sin(a), SHOULDER_Z])
