"""Leg FK/IK: round-trips, limits and agreement with the CAD stance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocky import kinematics as K
from rocky import model_params as P


@pytest.mark.parametrize("leg", range(P.N_LEGS))
def test_fk_ik_round_trip(leg):
    rng = np.random.default_rng(leg)
    tested = 0
    for _ in range(400):
        q = np.array([
            rng.uniform(*P.SWEEP_RANGE),
            rng.uniform(*P.LIFT_RANGE),
            rng.uniform(*P.elbow_range(leg)),
        ])
        p = K.forward_kinematics(leg, q)
        back = K.inverse_kinematics(leg, p)
        assert np.allclose(back, q, atol=1e-9), f"q={np.degrees(q)} -> {np.degrees(back)}"
        tested += 1
    assert tested == 400


@pytest.mark.parametrize("leg", range(P.N_LEGS))
def test_body_leg_frame_round_trip(leg):
    rng = np.random.default_rng(100 + leg)
    for _ in range(50):
        p = rng.uniform(-0.4, 0.4, size=3)
        assert np.allclose(K.leg_to_body(leg, K.body_to_leg(leg, p)), p, atol=1e-12)


@pytest.mark.parametrize("leg", range(P.N_LEGS))
def test_home_pose_matches_cad_stance(leg):
    """Every foot contacts the ground 44.73 mm below the body origin."""
    assert K.home_contact_position(leg)[2] == pytest.approx(-0.044730, abs=2e-5)


def test_home_footprint_is_regular():
    radii = [np.hypot(*K.home_contact_position(k)[:2]) for k in range(P.N_LEGS)]
    assert max(radii) - min(radii) < 1e-3          # within a millimetre
    assert 2 * np.mean(radii) == pytest.approx(0.555, abs=0.005)


@pytest.mark.parametrize("leg", range(P.N_LEGS))
def test_unreachable_targets_raise(leg):
    far = np.array([1.0, 0.0, 0.0])
    with pytest.raises(K.Unreachable):
        K.inverse_kinematics(leg, far)
    folded = np.array([0.01, 0.0, 0.0])
    with pytest.raises(K.Unreachable):
        K.inverse_kinematics(leg, folded)


@pytest.mark.parametrize("leg", range(P.N_LEGS))
def test_clip_always_returns_legal_joints(leg):
    rng = np.random.default_rng(200 + leg)
    lo = np.array([P.SWEEP_RANGE[0], P.LIFT_RANGE[0], P.elbow_range(leg)[0]])
    hi = np.array([P.SWEEP_RANGE[1], P.LIFT_RANGE[1], P.elbow_range(leg)[1]])
    for _ in range(200):
        p = rng.uniform(-0.5, 0.5, size=3)
        q = K.inverse_kinematics(leg, p, clip=True)
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)


def test_elbow_branch_is_knee_up():
    """IK must pick the positive-elbow solution; the hardware cannot do the other."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        q = np.array([0.0, rng.uniform(*P.LIFT_RANGE), rng.uniform(*P.ELBOW_RANGE)])
        assert K.inverse_kinematics(1, K.forward_kinematics(1, q))[2] > 0.0


def test_joint_vector_ordering():
    q = np.zeros((5, 3))
    q[:, 0] = [1, 2, 3, 4, 5]
    v = K.joint_vector(q, wrist=9.0)
    assert len(v) == len(P.JOINT_NAMES) == 16
    assert v[P.JOINT_NAMES.index("sweep_3")] == 4
    assert v[P.JOINT_NAMES.index("wrist_0")] == 9.0
