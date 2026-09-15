"""The torch gait must be the numpy gait. A reference that has drifted from the
gait it claims to represent would silently teach the policy the wrong thing."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocky import model_params as P  # noqa: E402
from rocky.gait import CRUISE_SPEED, GaitParams, WaveGait  # noqa: E402
from rocky.gait_torch import TorchGait  # noqa: E402

NP = WaveGait()
TG = TorchGait()
CMDS = [(0.0, 0.0, 0.0), (CRUISE_SPEED, 0.0, 0.0), (0.03, 0.02, 0.0),
        (0.0, 0.0, 0.2), (-0.03, 0.0, -0.1), (0.05, -0.02, 0.15)]
TIMES = np.linspace(0.0, NP.p.period, 37)


def _phase(t):
    return torch.tensor([t / NP.p.period % 1.0], dtype=torch.float32)


@pytest.mark.parametrize("cmd", CMDS)
def test_foot_targets_match_numpy(cmd):
    for t in TIMES:
        want = NP.foot_targets(t, *cmd)
        got = TG.foot_targets(_phase(t), torch.tensor([cmd], dtype=torch.float32))[0].numpy()
        assert np.abs(got - want).max() < 2e-6, f"t={t} cmd={cmd}"


@pytest.mark.parametrize("cmd", CMDS)
def test_joint_targets_match_numpy(cmd):
    for t in TIMES:
        want = NP.action_vector(t, *cmd)
        got = TG.joint_targets(_phase(t), torch.tensor([cmd], dtype=torch.float32))[0].numpy()
        assert np.abs(got - want).max() < 2e-5, f"t={t} cmd={cmd} maxdiff={np.abs(got-want).max()}"


def test_batched_matches_one_at_a_time():
    ph = torch.rand(9)
    cmd = torch.tensor(CMDS + CMDS[:3], dtype=torch.float32)
    batched = TG.joint_targets(ph, cmd)
    for i in range(9):
        single = TG.joint_targets(ph[i:i+1], cmd[i:i+1])
        assert torch.allclose(batched[i], single[0], atol=1e-6)


def test_reference_respects_joint_limits():
    ph = torch.rand(64)
    cmd = torch.randn(64, 3) * torch.tensor([0.1, 0.05, 0.4])
    q = TG.joint_targets(ph, cmd)
    for k in range(P.N_LEGS):
        i = P.JOINT_NAMES.index(f"sweep_{k}")
        assert q[:, i].min() >= P.SWEEP_RANGE[0] - 1e-6 and q[:, i].max() <= P.SWEEP_RANGE[1] + 1e-6
        i = P.JOINT_NAMES.index(f"elbow_{k}")
        assert q[:, i].min() >= P.elbow_range(k)[0] - 1e-6 and q[:, i].max() <= P.elbow_range(k)[1] + 1e-6


def test_wrist_is_held_at_home():
    q = TG.joint_targets(torch.rand(8), torch.zeros(8, 3))
    assert torch.allclose(q[:, P.JOINT_NAMES.index("wrist_0")], torch.full((8,), P.HOME_WRIST))
