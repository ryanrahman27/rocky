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
from rocky.approach import spawn_pose                             # noqa: E402
from rocky.sonar import observation                               # noqa: E402
from blind_stack_demo import Blind, parked_pose, CONTROL_HZ       # noqa: E402
from stack_demo import Scene, gait_joints, layout                 # noqa: E402
from train_diffusion import load_policy                           # noqa: E402


class Film:
    """Streams frames straight to the encoder, cutting cameras at the handover.

    Buffering a full-pipeline episode is two gigabytes of uint8 -- 70 seconds at
    30 fps and 960x720 -- so the frames go to ffmpeg as they are rendered. The
    rangefinder rays draw as lines, which looks like what it is and compresses
    like static, hence the crf.
    """

    def __init__(self, model, path, fps, width, height):
        import imageio.v2 as imageio

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ren = mujoco.Renderer(model, height=height, width=width)
        self.writer = imageio.get_writer(
            path, fps=fps, codec="libx264", pixelformat="yuv420p",
            output_params=["-crf", "28", "-preset", "slow"])
        self.fps, self.next, self.n = fps, 0.0, 0

    def maybe(self, sc, t, phase):
        if t < self.next:
            return
        self.next = t + 1.0 / self.fps
        # Three shots: wide while it crosses the room, medium over the brace
        # and the feel sweep, then tight on the hand for the pick and place,
        # which is the part that is actually being driven by the policy.
        cam = ("scene" if phase in ("listen", "walk")
               else "hand" if phase == "stack" else "closeup")
        self.ren.update_scene(sc.d, camera=cam)
        self.writer.append_data(self.ren.render())
        self.n += 1

    def close(self):
        self.writer.close()


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


def episode(args, rng, runner, film_path=None):
    """One episode; `film_path` records it, otherwise it runs headless."""
    sc = Scene()
    red, blue = layout(rng, args)
    b = Blind(sc, locomotion=args.locomotion)
    film = None if film_path is None else Film(sc.m, film_path, args.fps,
                                               args.width, args.height)

    if args.walk_in:
        # The whole thing: find the cubes, walk to them, brace, feel, then hand
        # over. --locomotion swaps the analytic gait for the trained PPO policy;
        # both consume the twist `rocky.approach` emits.
        base_xy, yaw = spawn_pose(rng, (np.array(red) + np.array(blue)) / 2.0)
        seed, start, timeout = None, "listen", args.walk_timeout
    else:
        base_xy, yaw, seed = parked_pose(rng, red, blue)
        start, timeout = "brace", args.perception_timeout

    sc.reset(b.gait.neutral_joint_targets(), base_xy, yaw, red, blue)
    for _ in range(300):
        sc.apply(gait_joints(b.gait, 0.0, 0.0, 0.0, 0.0))
        mujoco.mj_step(sc.m, sc.d)

    # --- scripted half: search, approach, brace, feel ----------------------
    watch = None if film is None else (lambda t, phase, joints: film.maybe(sc, t, phase))
    res = b.run(timeout=timeout, on_step=watch,
                start_phase=start, seed_pair=seed, stop_at="stack")
    if res["phase"] != "stack":
        # Never found them, or never got parked. Not the policy's fault, and
        # scoring it as a manipulation failure would be dishonest.
        if film is not None:
            film.close()
        return dict(dz=0.0, dxy=9.99, stacked=False, on_top=False, lifted=False,
                    reached=False, walk_s=res["t"])
    walk_s = res["t"]

    # --- learned half ------------------------------------------------------
    runner.reset()
    lo, hi = sc.m.actuator_ctrlrange[:, 0], sc.m.actuator_ctrlrange[:, 1]
    dt = sc.dt
    n_ctrl = max(1, int(round(1.0 / (args.policy_hz * dt))))
    joints = dict(b.last_joints)
    names = sc.names
    steps = int(args.seconds / dt)
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
        if film is not None:
            film.maybe(sc, walk_s + i * dt, "stack")

    for i in range(300):
        mujoco.mj_step(sc.m, sc.d)
        if film is not None:
            film.maybe(sc, walk_s + args.seconds + i * dt, "stack")
    if film is not None:
        film.close()
        print(f"  wrote {film_path} ({film.n} frames)")
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
                lifted=dz > 0.015, reached=True, walk_s=walk_s)


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
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--walk-in", action="store_true",
                    help="run the whole thing: search, approach, feel, then the policy")
    ap.add_argument("--walk-timeout", type=float, default=110.0)
    ap.add_argument("--locomotion", default=None,
                    help="rsl-rl checkpoint to walk with; omit for the analytic gait")
    ap.add_argument("--torch-seed", type=int, default=0,
                    help="the sampler draws noise; fix it so a good episode can be re-filmed")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--cubes", type=float, nargs=4, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.torch_seed)
    net, ck = load_policy(args.checkpoint, args.device)
    runner = Runner(net, ck, args.n_action, args.device)
    rng = np.random.default_rng(args.seed)
    ok = lifted = on_top = reached = 0
    offsets = []
    for ep in range(args.episodes):
        res = episode(args, rng, runner, args.video if ep == 0 else None)
        ok += res["stacked"]
        on_top += res["on_top"]
        lifted += res["lifted"]
        reached += res["reached"]
        if res["on_top"]:
            offsets.append(res["dxy"] * 1000)
        verdict = ("never reached the cubes" if not res["reached"] else
                   "stacked" if res["stacked"] else
                   "on top " if res["on_top"] else
                   "dropped" if res["lifted"] else "no grasp")
        walk = f"walk {res['walk_s']:5.1f}s  " if args.walk_in else ""
        print(f"  ep {ep:3d}  {walk}dz {res['dz'] * 1000:6.1f} mm  "
              f"dxy {res['dxy'] * 1000:6.1f} mm  {verdict}", flush=True)
    n = args.episodes
    if args.walk_in:
        print(f"\n{reached}/{n} reached the cubes")
    print(f"{on_top}/{n} on top and supported, {ok}/{n} inside the strict 14 mm, "
          f"{lifted}/{n} got the cube off the floor")
    if offsets:
        print(f"when it lands: offset median {np.median(offsets):.1f} mm, "
              f"worst {max(offsets):.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
