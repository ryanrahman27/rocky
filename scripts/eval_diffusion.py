#!/usr/bin/env python3
"""Roll the diffusion policy out in the simulator and see if it stacks anything.

    python scripts/eval_diffusion.py out/blind_policy.pt --episodes 40
    python scripts/eval_diffusion.py out/blind_policy.pt --episodes 1 \
        --video docs/rocky_policy.mp4

The split is the one the task has anyway. Perception -- bracing, lifting the arm
and sweeping the gripper's fan across the front until both cubes are located --
stays scripted, because it is a sensing routine rather than a skill. What the
policy replaces is everything after: approach, descend, close, lift, carry,
place, release, retract and stow, about fifteen seconds of it, driven by the
sonar, the touch pads, the joints and the two cube positions perception handed
over.

It runs at the rate the demonstrations were recorded at, samples a chunk of
future actions, executes the first few and re-plans -- the standard receding
horizon. Between updates the targets are held, which costs nothing because the
servos are position sources anyway.

Reported: how often the red cube ends up on the blue one, measured exactly the
way `blind_stack_demo.py` measures it, so the two numbers can be compared.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocky import arm                                             # noqa: E402
from rocky.sonar import observation                               # noqa: E402
from blind_stack_demo import Blind, parked_pose, CONTROL_HZ       # noqa: E402
from stack_demo import Scene, gait_joints, layout                 # noqa: E402
from train_diffusion import load_policy                           # noqa: E402


class Runner:
    """Perception scripted, manipulation learned."""

    def __init__(self, net, ck, n_action: int, device="cpu"):
        self.net, self.ck, self.n_action = net, ck, n_action
        self.dev = torch.device(device)
        self.o_mu = torch.tensor(ck["obs_mean"], dtype=torch.float32, device=self.dev)
        self.o_sd = torch.tensor(ck["obs_std"], dtype=torch.float32, device=self.dev)
        self.a_mu = np.asarray(ck["act_mean"], np.float64)
        self.a_sd = np.asarray(ck["act_std"], np.float64)
        self.n_obs = net.n_obs

    def reset(self):
        self.hist: list[np.ndarray] = []
        self.chunk: np.ndarray | None = None
        self.used = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.hist.append(obs)
        self.hist = self.hist[-self.n_obs:]
        while len(self.hist) < self.n_obs:            # first tick: repeat it
            self.hist.insert(0, self.hist[0])
        if self.chunk is None or self.used >= self.n_action:
            o = torch.tensor(np.stack(self.hist)[None], dtype=torch.float32, device=self.dev)
            o = (o - self.o_mu) / self.o_sd
            with torch.no_grad():
                chunk = self.net.act(o)[0].cpu().numpy()
            self.chunk = chunk * self.a_sd + self.a_mu
            self.used = 0
        a = self.chunk[self.used]
        self.used += 1
        return a


def episode(args, rng, runner, renderer=None, frames=None):
    sc = Scene()
    red, blue = layout(rng, args)
    base_xy, yaw, seed = parked_pose(rng, red, blue)
    b = Blind(sc)
    sc.reset(b.gait.neutral_joint_targets(), base_xy, yaw, red, blue)
    for _ in range(300):
        sc.apply(gait_joints(b.gait, 0.0, 0.0, 0.0, 0.0))
        mujoco.mj_step(sc.m, sc.d)

    # --- scripted half: brace, lift, feel ---------------------------------
    handover = {}

    def watch(t, phase, joints):
        if phase == "stack" and "t" not in handover:
            handover["t"] = t

    b.run(timeout=args.perception_timeout, on_step=watch,
          start_phase="brace", seed_pair=seed, stop_at="stack")

    # --- learned half ------------------------------------------------------
    runner.reset()
    lo, hi = sc.m.actuator_ctrlrange[:, 0], sc.m.actuator_ctrlrange[:, 1]
    dt = sc.dt
    n_ctrl = max(1, int(round(1.0 / (args.policy_hz * dt))))
    joints = dict(b.last_joints)
    names = sc.names
    steps = int(args.seconds / dt)
    nxt = 0.0
    for i in range(steps):
        if i % n_ctrl == 0:
            obs = observation(b.sonar, sc.d, sc.qadr, names, b.belief)
            a = runner.act(obs)
            # A servo cannot be driven past its stops, and an untrained sampler
            # will happily ask it to. Clamp to the actuator range and treat any
            # non-finite value as "hold what you had".
            a = np.where(np.isfinite(a), a, [joints[n] for n in names])
            a = np.clip(a, lo, hi)
            joints = {n: float(v) for n, v in zip(names, a)}
        sc.apply(joints)
        mujoco.mj_step(sc.m, sc.d)
        wz, v = b.imu()
        b.belief[:2] = _carry(b.belief[:2], dt, wz, v)
        b.belief[2:] = _carry(b.belief[2:], dt, wz, v)
        if renderer is not None and i * dt >= nxt:
            nxt += 1.0 / args.fps
            renderer.update_scene(sc.d, camera="closeup")
            frames.append(renderer.render().copy())

    for _ in range(300):
        mujoco.mj_step(sc.m, sc.d)
    r, bl = sc.d.xpos[sc.body["red"]], sc.d.xpos[sc.body["blue"]]
    dz, dxy = float(r[2] - bl[2]), float(np.hypot(*(r[:2] - bl[:2])))
    return dict(dz=dz, dxy=dxy,
                # Two bars. `stacked` is the strict one the scripted
                # demonstrator is scored by. `on_top` is the physical question:
                # a 40 mm cube offset by 18 mm is still sitting squarely on the
                # one below it, and calling that a failure tells you nothing
                # about whether the policy can do the task.
                stacked=abs(dz - arm.CUBE) < 0.006 and dxy < 0.014,
                on_top=abs(dz - arm.CUBE) < 0.008 and dxy < 0.020,
                lifted=dz > 0.015)


def _carry(p, dt, wz, v):
    c, s = math.cos(-wz * dt), math.sin(-wz * dt)
    return (np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
            - np.asarray(v, float)[:2] * dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--n-action", type=int, default=4, help="chunk steps executed per re-plan")
    ap.add_argument("--policy-hz", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=17.0)
    ap.add_argument("--perception-timeout", type=float, default=12.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--cubes", type=float, nargs=4, default=None)
    args = ap.parse_args()

    net, ck = load_policy(args.checkpoint, args.device)
    runner = Runner(net, ck, args.n_action, args.device)
    rng = np.random.default_rng(args.seed)
    frames = [] if args.video else None
    renderer = None

    ok = lifted = on_top = 0
    offsets = []
    for ep in range(args.episodes):
        ren = None
        if args.video and ep == 0:
            renderer = mujoco.Renderer(Scene().m, height=720, width=960)
            ren = renderer
        res = episode(args, rng, runner, ren, frames)
        ok += res["stacked"]
        on_top += res["on_top"]
        lifted += res["lifted"]
        if res["on_top"]:
            offsets.append(res["dxy"] * 1000)
        verdict = ("stacked" if res["stacked"] else
                   "on top " if res["on_top"] else
                   "dropped" if res["lifted"] else "no grasp")
        print(f"  ep {ep:3d}  dz {res['dz'] * 1000:6.1f} mm  dxy {res['dxy'] * 1000:6.1f} mm  "
              f"{verdict}", flush=True)
    n = args.episodes
    print(f"\n{on_top}/{n} on top and supported, {ok}/{n} inside the strict 14 mm, "
          f"{lifted}/{n} got the cube off the floor")
    if offsets:
        print(f"when it lands: offset median {np.median(offsets):.1f} mm, "
              f"worst {max(offsets):.1f} mm")

    if args.video and frames:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=args.fps, codec="libx264",
                        output_params=["-crf", "30", "-preset", "slow"], pixelformat="yuv420p")
        print(f"wrote {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
