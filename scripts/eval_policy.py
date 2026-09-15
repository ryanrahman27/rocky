#!/usr/bin/env python3
"""Measure what a trained policy actually does, at fixed commands.

    python scripts/eval_policy.py logs/rsl_rl/rocky_velocity/<run>/model_6000.pt
    python scripts/eval_policy.py --scripted          # the analytic gait, as a baseline

Reward curves repeatedly failed to tell us whether this robot was walking:
velocity tracking can read 98% of maximum while the robot stands still, and the
contact-schedule reward is satisfied in full by stepping on the spot. This rolls
the controller out at fixed commands and measures net displacement, which cannot
be gamed by a reward term being the wrong shape.
"""

from __future__ import annotations

import argparse, math, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import rocky.tasks  # noqa: E402,F401
from rocky import model_params as P  # noqa: E402
from rocky.gait import GaitParams, WaveGait  # noqa: E402
from rocky.tasks.constants import ROCKY_ACTION_SCALE  # noqa: E402

COMMANDS = [(0.00, 0.0, 0.0), (0.02, 0.0, 0.0), (0.04, 0.0, 0.0), (0.06, 0.0, 0.0),
            (0.08, 0.0, 0.0), (-0.04, 0.0, 0.0), (0.0, 0.03, 0.0),
            (0.0, 0.0, 0.15), (0.0, 0.0, 0.30)]


def load_actor(path: str):
    """Rebuild the actor from an rsl-rl checkpoint: obs normaliser + ELU MLP."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].float().squeeze()
    std = sd["obs_normalizer._std"].float().squeeze()
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    W = [(sd[f"mlp.{i}.weight"].float(), sd[f"mlp.{i}.bias"].float()) for i in layers]

    def act(obs):
        x = (obs - mean) / std
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < len(W) - 1:
                x = torch.nn.functional.elu(x)
        return x
    return act


def yaw_of(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="rsl-rl .pt checkpoint")
    ap.add_argument("--scripted", action="store_true", help="evaluate the analytic gait instead")
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Rocky")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=16.0)
    args = ap.parse_args()
    if not args.checkpoint and not args.scripted:
        ap.error("give a checkpoint, or --scripted")

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = len(COMMANDS)
    cfg.sim.device = args.device
    tw = cfg.commands["twist"]
    # pin the commands: no heading control, no standing/forward special cases,
    # no resampling, no pushes.
    tw.heading_command = False
    tw.ranges.heading = None
    tw.rel_standing_envs = tw.rel_heading_envs = tw.rel_forward_envs = 0.0
    tw.resampling_time_range = (1e9, 1e9)
    cfg.events.pop("push_robot", None)

    env = ManagerBasedRlEnv(cfg, device=args.device)
    obs, _ = env.reset()
    term = env.command_manager.get_term("twist")
    fixed = torch.tensor(COMMANDS, dtype=torch.float32, device=env.device)

    if args.scripted:
        gait = WaveGait(GaitParams())
        default = np.array([gait.neutral_joint_dict()[n] for n in P.JOINT_NAMES])
        scale = np.array([next(v for k, v in ROCKY_ACTION_SCALE.items() if re.match(k, n))
                          for n in P.JOINT_NAMES])
        t = [0.0]

        def action(_obs):
            a = np.stack([(gait.action_vector(t[0], *c) - default) / scale for c in COMMANDS])
            t[0] += env.step_dt
            return torch.tensor(a, dtype=torch.float32, device=env.device)
        label = "scripted wave gait"
    else:
        actor = load_actor(args.checkpoint)
        action = lambda o: actor(o)  # noqa: E731
        label = Path(args.checkpoint).name

    robot, contact = env.scene["robot"], env.scene["feet_ground_contact"]
    for _ in range(int(args.settle / env.step_dt)):
        term.vel_command_b[:] = fixed
        obs, *_ = env.step(action(obs["actor"]))

    p0 = robot.data.root_link_pos_w[:, :2].clone()
    y0 = yaw_of(robot.data.root_link_quat_w).clone()
    n = int(args.seconds / env.step_dt)
    duty = torch.zeros(len(COMMANDS), device=env.device)
    peak = torch.zeros(len(COMMANDS), 5, device=env.device)
    tilts = []
    for _ in range(n):
        term.vel_command_b[:] = fixed
        obs, *_ = env.step(action(obs["actor"]))
        duty += (contact.data.found > 0).float().mean(dim=1)
        peak = torch.maximum(peak, env.scene["foot_height_scan"].data.heights)
        q = robot.data.root_link_quat_w
        tilts.append(torch.rad2deg(torch.arccos(torch.clamp(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))))

    dp = robot.data.root_link_pos_w[:, :2] - p0
    dyaw = (yaw_of(robot.data.root_link_quat_w) - y0 + math.pi) % (2 * math.pi) - math.pi
    cy, sy = torch.cos(-y0), torch.sin(-y0)
    vx = (cy * dp[:, 0] - sy * dp[:, 1]) / args.seconds
    vy = (sy * dp[:, 0] + cy * dp[:, 1]) / args.seconds
    wz = dyaw / args.seconds
    tilts = torch.stack(tilts)

    print(f"\n{label}  ({args.seconds:.0f}s per command, net displacement)\n")
    print("%6s %6s %6s | %7s %7s %7s | %6s | %5s %7s %6s"
          % ("cvx", "cvy", "cwz", "vx", "vy", "wz", "% cmd", "duty", "clear", "tilt"))
    for e, (cx, cyy, cw) in enumerate(COMMANDS):
        ref = cx or cyy or cw
        got = vx[e] if cx else (vy[e] if cyy else wz[e])
        pct = f"{got / ref * 100:5.0f}%" if ref else "    -"
        print("%6.3f %6.3f %6.3f | %7.4f %7.4f %7.4f | %s | %5.2f %5.0fmm %5.1f"
              % (cx, cyy, cw, vx[e], vy[e], wz[e], pct, duty[e] / n,
                 (peak[e].mean().item() - 0.012) * 1000, tilts[:, e].max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
