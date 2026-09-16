"""The diffusion policy's data pipeline and sampler."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the policy needs torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_diffusion import DiffusionPolicy, windows  # noqa: E402


def fake(n_ep=3, per_phase=40, obs=7, act=3):
    """Episodes that go brace -> feel -> stack, one with a retry."""
    rows = []
    for e in range(n_ep):
        seq = ["brace"] * per_phase + ["feel"] * per_phase + ["stack"] * per_phase
        if e == 1:                                   # a failed grasp, then a retry
            seq += ["feel"] * per_phase + ["stack"] * per_phase
        for i, ph in enumerate(seq):
            rows.append((e, ph, np.full(obs, e + i / 1000.0), np.full(act, i / 1000.0)))
    return dict(episode=np.array([r[0] for r in rows]),
                phase=np.array([r[1] for r in rows]),
                obs=np.stack([r[2] for r in rows]).astype(np.float32),
                action=np.stack([r[3] for r in rows]).astype(np.float32))


def test_phase_filtering_keeps_only_what_was_asked_for():
    O, A, owner = windows(fake(), n_obs=2, horizon=4, phases=["stack"])
    assert len(O) > 0
    assert set(owner.tolist()) == {0, 1, 2}


def test_runs_are_cut_at_discontinuities_not_just_episodes():
    """A retry sends the demonstrator back to the feel sweep mid-episode."""
    data = fake()
    O, A, owner = windows(data, n_obs=2, horizon=4, phases=["stack"])
    # Episode 1 has two separate stack runs of 40; contiguous windowing of 80
    # would give more windows than two runs of 40 do.
    per_run = 40 - 2 + 1 - 4
    assert (owner == 1).sum() == 2 * per_run
    assert (owner == 0).sum() == per_run


def test_a_window_never_straddles_a_gap():
    data = fake()
    O, A, owner = windows(data, n_obs=2, horizon=4, phases=["stack"])
    for a in A:                     # actions were built as a ramp of 1/1000 per step
        steps = np.diff(a[:, 0])
        assert np.allclose(steps, steps[0], atol=1e-6)


def test_the_noise_schedule_is_monotone_and_bounded():
    net = DiffusionPolicy(4, 2, n_obs=1, horizon=2, steps=30, width=16)
    abar = net.abar.numpy()
    assert abar[0] < 1.0 and abar[-1] > 0.0
    assert np.all(np.diff(abar) < 0.0), "abar must decrease with the step index"
    assert np.all((net.betas.numpy() > 0) & (net.betas.numpy() < 1))


def test_the_sampler_recovers_something_it_can_trivially_learn():
    """One observation, one right answer. If this fails the sampler is wrong.

    It is worth having: an earlier version trained to a perfectly healthy loss
    and sampled chunks two radians away from anything in the data, because the
    reverse step divided by sqrt(alpha) instead of going through x0.
    """
    torch.manual_seed(0)
    n_obs, horizon, obs_dim, act_dim = 1, 4, 3, 2
    net = DiffusionPolicy(obs_dim, act_dim, n_obs, horizon, steps=30, width=48)
    obs = torch.zeros(64, n_obs, obs_dim)
    obs[:32, :, 0] = 1.0
    act = torch.zeros(64, horizon, act_dim)
    act[:32] = 1.0                                  # two observations, two answers
    act[32:] = -1.0
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(450):
        loss = net.loss(obs, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    got = net.act(obs).detach()
    assert torch.isfinite(got).all()
    assert got.shape == act.shape
    assert float((got[:32] - 1.0).abs().mean()) < 0.35
    assert float((got[32:] + 1.0).abs().mean()) < 0.35
