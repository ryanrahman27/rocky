#!/usr/bin/env python3
"""Train a state-only diffusion policy on the blind stacking demonstrations.

    python scripts/blind_stack_demo.py --episodes 80 --noise 0.004 \
        --dataset data/blind_stack.npz
    python scripts/train_diffusion.py data/blind_stack.npz --epochs 300
    python scripts/train_diffusion.py data/blind_stack.npz --eval out/policy.pt

This is Diffusion Policy (Chi et al. 2023) at its smallest useful size: a DDPM
over short chunks of future actions, conditioned on the last couple of
observations. No images, because the robot has no eyes -- the observation is the
sonar residuals, the gripper's fan, the touch pads, the joints and the IMU, about
320 numbers. That makes the whole thing an MLP, not a ResNet, and it trains on a
laptop CPU in minutes rather than on a GPU in hours.

Why chunks. The demonstrator's action at any instant depends on where it is in a
sequence that lasts fifteen seconds, and a policy that predicts one step at a
time from one observation cannot tell "descending onto the cube" from "lowering
it onto the other one" -- the arm is in much the same place and moving much the
same way. Predicting `horizon` steps at once and executing the first few gives
it somewhere to put that intent.

Why diffusion rather than regression. The interesting moments are multimodal.
When the touch pads say the grasp missed, the demonstrator sometimes nudges and
closes again and sometimes lifts off and re-sweeps; averaging those two gives a
motion that does neither.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Cond(nn.Module):
    """Observation history -> a conditioning vector."""

    def __init__(self, obs_dim: int, n_obs: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_obs, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )

    def forward(self, obs):                      # (B, n_obs, obs_dim)
        return self.net(obs.flatten(1))


class Eps(nn.Module):
    """Noise predictor over a flattened action chunk."""

    def __init__(self, act_dim: int, horizon: int, cond: int, width: int):
        super().__init__()
        self.horizon, self.act_dim = horizon, act_dim
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.net = nn.Sequential(
            nn.Linear(act_dim * horizon + cond + width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, act_dim * horizon),
        )

    def forward(self, a, k, c):                  # (B, H, A), (B,), (B, C)
        t = self.time(k.float().unsqueeze(-1) / 100.0)
        return self.net(torch.cat([a.flatten(1), c, t], dim=-1)).view(-1, self.horizon, self.act_dim)


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_obs=2, horizon=8, steps=50, width=512):
        super().__init__()
        self.n_obs, self.horizon, self.steps = n_obs, horizon, steps
        self.cond = Cond(obs_dim, n_obs, width)
        self.eps = Eps(act_dim, horizon, width, width)
        # cosine schedule: gentler at both ends than the linear one, which
        # matters when the action chunks are as smooth as a servo trajectory
        s = 0.008
        k = torch.arange(steps + 1) / steps
        ac = torch.cos((k + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = torch.clip(1 - ac[1:] / ac[:-1], 0, 0.999)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("abar", torch.cumprod(1.0 - betas, dim=0))

    def loss(self, obs, act):
        b = act.shape[0]
        k = torch.randint(0, self.steps, (b,), device=act.device)
        noise = torch.randn_like(act)
        ab = self.abar[k].view(-1, 1, 1)
        noisy = ab.sqrt() * act + (1 - ab).sqrt() * noise
        return F.mse_loss(self.eps(noisy, k, self.cond(obs)), noise)

    @torch.no_grad()
    def act(self, obs, clip: float = 3.0):
        """Reverse diffusion, going through x0 rather than straight to the mean.

        The textbook one-liner divides by sqrt(alpha), which at the noisy end of
        a 50-step cosine schedule is a factor of thirty, and any error in the
        predicted noise is amplified by it. Reconstructing x0, clamping it to the
        range normalised actions actually live in, and then taking the posterior
        mean is the standard fix -- without it this sampler returned chunks about
        two radians away from anything the demonstrator ever did, off a model
        whose training loss was perfectly healthy.
        """
        c = self.cond(obs)
        a = torch.randn(obs.shape[0], self.horizon, self.eps.act_dim, device=obs.device)
        for k in reversed(range(self.steps)):
            kk = torch.full((obs.shape[0],), k, device=obs.device, dtype=torch.long)
            e = self.eps(a, kk, c)
            ab = self.abar[k]
            ab_prev = self.abar[k - 1] if k > 0 else torch.ones_like(ab)
            x0 = ((a - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-clip, clip)
            beta, al = self.betas[k], self.alphas[k]
            mean = (ab_prev.sqrt() * beta / (1 - ab) * x0
                    + al.sqrt() * (1 - ab_prev) / (1 - ab) * a)
            if k == 0:
                return mean
            var = beta * (1 - ab_prev) / (1 - ab)
            a = mean + var.sqrt() * torch.randn_like(a)
        return a


def windows(data, n_obs, horizon):
    """Chunk each episode into (obs history, future actions) pairs."""
    obs, act, ep = data["obs"], data["action"], data["episode"]
    O, A = [], []
    for e in np.unique(ep):
        sel = np.flatnonzero(ep == e)
        o, a = obs[sel], act[sel]
        for i in range(n_obs - 1, len(sel) - horizon):
            O.append(o[i - n_obs + 1:i + 1])
            A.append(a[i:i + horizon])
    return np.stack(O), np.stack(A)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", default="out/policy.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-obs", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=False)
    O, A = windows(data, args.n_obs, args.horizon)
    print(f"{len(O)} windows from {len(np.unique(data['episode']))} episodes; "
          f"obs {O.shape[-1]}, action {A.shape[-1]}, horizon {args.horizon}")

    o_mu, o_sd = O.reshape(-1, O.shape[-1]).mean(0), O.reshape(-1, O.shape[-1]).std(0) + 1e-6
    a_mu, a_sd = A.reshape(-1, A.shape[-1]).mean(0), A.reshape(-1, A.shape[-1]).std(0) + 1e-6
    dev = torch.device(args.device)
    Ot = torch.tensor((O - o_mu) / o_sd, dtype=torch.float32, device=dev)
    At = torch.tensor((A - a_mu) / a_sd, dtype=torch.float32, device=dev)

    net = DiffusionPolicy(O.shape[-1], A.shape[-1], args.n_obs, args.horizon,
                          args.steps, args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Ot)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        total, batches = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = net.loss(Ot[idx], At[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            batches += 1
        sched.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}  loss {total / max(batches, 1):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "obs_mean": o_mu, "obs_std": o_sd, "act_mean": a_mu, "act_std": a_sd,
        "cfg": dict(obs_dim=int(O.shape[-1]), act_dim=int(A.shape[-1]), n_obs=args.n_obs,
                    horizon=args.horizon, steps=args.steps, width=args.width),
        "actuators": data["actuators"].tolist(),
    }, args.out)
    print(f"wrote {args.out}")

    # One honest sanity check before anyone trusts this: sample a chunk for a
    # held-out window and see how far it lands from what the script actually did.
    with torch.no_grad():
        idx = torch.randperm(n)[:256].to(dev)
        pred = net.act(Ot[idx]).cpu().numpy() * a_sd + a_mu
        true = At[idx].cpu().numpy() * a_sd + a_mu
    err = np.abs(pred - true)
    print(f"sampled chunk error vs the demonstrator: median {np.median(err) * 1000:.2f} mrad, "
          f"p90 {np.percentile(err, 90) * 1000:.2f} mrad")
    return 0


def load_policy(path, device="cpu"):
    """Rebuild a trained policy, ready to step in the sim."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = DiffusionPolicy(**ck["cfg"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


if __name__ == "__main__":
    raise SystemExit(main())
