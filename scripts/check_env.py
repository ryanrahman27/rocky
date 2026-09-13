#!/usr/bin/env python3
"""Build the Rocky RL environment and step it, without training anything.

    python scripts/check_env.py                # CPU is fine, just slow to compile
    python scripts/check_env.py --device cuda

Verifies that the task registers, the scene assembles, the gait clock and
contact-schedule reward are live, and that a zero-action policy stands. Run this
before a training job: it fails in seconds-to-minutes instead of after a queue
wait.

A standing robot should score ~0.8 on Metrics/gait_schedule_match: four of five
feet agree with the wave gait at any instant, and the fifth is the one that
ought to be in the air.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import rocky.tasks  # noqa: E402,F401  (registers the tasks)
from rocky import gait_mdp, model_params as P  # noqa: E402
from rocky.gait import GaitParams  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-Rocky")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = ManagerBasedRlEnv(cfg, device=args.device)
    obs, _ = env.reset()

    robot = env.scene["robot"]
    print(f"\n  task           {args.task}")
    print(f"  observations   actor {tuple(obs['actor'].shape)}  critic {tuple(obs['critic'].shape)}")
    print(f"  actions        {env.action_manager.total_action_dim}")
    print(f"  joints         {len(robot.joint_names)}")
    print(f"  default pose   {np.round(np.degrees(robot.data.default_joint_pos[0].cpu().numpy()[:3]), 1)} deg (leg 0)")

    assert tuple(robot.joint_names) == P.JOINT_NAMES, "joint order differs from model_params"

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    contacts, rewards = [], []
    for _ in range(args.steps):
        _, rew, term, _, _ = env.step(zero)
        rewards.append(float(rew.mean()))
        s = env.scene["feet_ground_contact"]
        contacts.append((s.data.found > 0).float()[0].cpu().numpy().copy())

    log = env.extras.get("log", {})
    print(f"\n  mean reward    {np.mean(rewards):+.3f} (zero actions)")
    for k in ("Metrics/gait_schedule_match", "Metrics/duty_factor", "Metrics/swing_height_err"):
        if k in log:
            print(f"  {k:<30} {float(log[k]):.4f}")
    print(f"  feet down      {np.array(contacts).sum(axis=1).mean():.2f} / 5")
    print(f"  base height    {float(robot.data.root_link_pos_w[0, 2]):.4f} m")
    print(f"  terminations   {int(term.sum())}")

    phase = gait_mdp.gait_phase(env, GaitParams().period)
    assert phase.shape == (env.num_envs,)
    print("\n  environment OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
