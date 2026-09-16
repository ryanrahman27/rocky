#!/usr/bin/env python3
"""Walk up to two cubes, brace, and stack the red one on the blue.

    python scripts/stack_demo.py --video docs/rocky_stack.mp4
    python scripts/stack_demo.py --episodes 20 --no-video        # success rate
    python scripts/stack_demo.py --dataset data/stack.npz        # VLA demos

The run is two controllers with a handoff between them, which is the whole point
of the exercise:

    walk      a locomotion controller drives the base to a stand-off pose in
              front of the cubes. Today that is the analytic wave gait; the
              trained policy plugs in at the same place -- it consumes the same
              (vx, vy, wz) twist `rocky.approach` produces.
    settle    both controllers hold the walking stance for a beat.
    stack     `rocky.stack` widens the four stance legs into a brace, lifts leg 0
              off the ground, and runs the pick-and-place. The four legs hold
              station for the whole sequence; only the arm moves.

`--dataset` writes what a VLA needs to learn the second half: both camera views,
the joint state, the action the script issued, and the phase label, at the
control rate, for the manipulation segment only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Offscreen rendering needs a GL backend, and a headless box has no display for
# the default one to attach to. Only force software GL when there is clearly
# nothing better; a workstation with a GPU keeps whatever it already has.
if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rocky import arm, model_params as P                        # noqa: E402
from rocky.approach import Approach, ApproachParams, spawn_pose  # noqa: E402
from rocky.gait import WaveGait                                  # noqa: E402
from rocky.stack import StackController, StackPlan               # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "model" / "rocky_cubes.xml"
ARM_JOINTS = ("sweep_0", "lift_0", "elbow_0", "wrist_0")


def layout(rng, args):
    """Cube positions, in the world frame."""
    if args.cubes is not None:
        rx, ry, bx, by = args.cubes
        return np.array([rx, ry]), np.array([bx, by])
    sep = rng.uniform(0.115, 0.150)
    bearing = rng.uniform(-math.radians(12.0), math.radians(12.0))
    R = rng.uniform(0.340, 0.380)
    d = sep / (2.0 * R)
    return (np.array([R * math.cos(bearing + d), R * math.sin(bearing + d)]),
            np.array([R * math.cos(bearing - d), R * math.sin(bearing - d)]))


class Scene:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(self.m.nu)]
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)   # noqa: E731
        self.qadr = {n: self.m.jnt_qposadr[jid(n)] for n in self.names}
        self.cube_adr = {c: self.m.jnt_qposadr[jid(f"cube_{c}_free")] for c in ("red", "blue")}
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.body = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"cube_{c}")
                     for c in ("red", "blue")}
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "start")
        self.dt = float(self.m.opt.timestep)

    def reset(self, stance, base_xy, yaw, red, blue, height=0.075):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        for k in range(P.N_LEGS):
            for name, v in zip((f"sweep_{k}", f"lift_{k}", f"elbow_{k}"), stance[k]):
                self.d.qpos[self.qadr[name]] = float(v)
        self.d.qpos[self.qadr["wrist_0"]] = P.HOME_WRIST
        self.d.qpos[self.qadr["jaw_r"]] = self.d.qpos[self.qadr["jaw_l"]] = arm.JAW_OPEN
        self.d.qpos[0:2] = base_xy
        self.d.qpos[2] = height
        self.d.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        for c, xy in (("red", red), ("blue", blue)):
            a = self.cube_adr[c]
            self.d.qpos[a:a + 7] = (xy[0], xy[1], arm.CUBE / 2, 1, 0, 0, 0)
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def apply(self, joints: dict[str, float]) -> np.ndarray:
        ctrl = np.array([joints[n] for n in self.names])
        self.d.ctrl[:] = ctrl
        return ctrl

    def arm_q(self) -> np.ndarray:
        return np.array([self.d.qpos[self.qadr[n]] for n in ARM_JOINTS])

    def tilt(self) -> float:
        return math.degrees(math.acos(min(1.0, self.d.xmat[self.base].reshape(3, 3)[2, 2])))


def gait_joints(gait: WaveGait, t: float, vx: float, vy: float, wz: float) -> dict[str, float]:
    q = gait.joint_targets(t, vx, vy, wz)
    out = {"jaw_r": arm.JAW_OPEN, "jaw_l": arm.JAW_OPEN, "wrist_0": P.HOME_WRIST}
    for k in range(P.N_LEGS):
        out[f"sweep_{k}"], out[f"lift_{k}"], out[f"elbow_{k}"] = (float(v) for v in q[k])
    return out


def run(args, rng, renderer=None, cams=None):
    sc = Scene()
    gait = WaveGait()
    appr = Approach(ApproachParams(stand_off=args.stand_off), gait)
    stack = StackController(StackPlan())
    red, blue = layout(rng, args)
    mid = (red + blue) / 2.0

    if args.start is not None:
        base_xy, yaw = np.array(args.start[:2]), args.start[2]
    else:
        base_xy, yaw = spawn_pose(rng, (red + blue) / 2.0)
    sc.reset(gait.neutral_joint_targets(), base_xy, yaw, red, blue)

    frames, records = [], []
    step = None
    t = 0.0
    phase, t_phase = "walk", 0.0
    handoff_t = None
    max_tilt = 0.0
    next_frame = 0.0
    limit = args.walk_timeout + 1.0 + stack.duration + 1.0

    while t < limit:
        if phase == "walk":
            tw = appr.command(t, sc.d.xpos[sc.base], sc.d.xquat[sc.base], mid, red - blue)
            joints = gait_joints(gait, t, tw.vx, tw.vy, tw.wz)
            if tw.arrived or t > args.walk_timeout:
                phase, t_phase, handoff_t = "settle", t, t
                parked = parked_geometry(sc, red, blue)
        elif phase == "settle":
            joints = gait_joints(gait, t_phase, 0.0, 0.0, 0.0)
            if t - t_phase > 1.0:
                stack.reset(red, blue)
                phase, t_phase = "stack", t
        else:
            step = stack.step(t - t_phase, sc.d.xpos[sc.base], sc.d.xquat[sc.base], sc.arm_q())
            joints = step.joints
            if step.done and t - t_phase > stack.duration + 0.8:
                break

        ctrl = sc.apply(joints)
        mujoco.mj_step(sc.m, sc.d)
        t += sc.dt
        max_tilt = max(max_tilt, sc.tilt())

        if renderer is not None and t >= next_frame:
            next_frame += 1.0 / args.fps
            renderer.update_scene(sc.d, camera=(args.camera if phase == "walk"
                                               else args.manip_camera))
            frames.append(renderer.render().copy())
        if (args.dataset and step is not None
                and len(records) * args.record_dt <= t - t_phase):
            records.append(sample(sc, cams, step, ctrl, t - t_phase))

    r, b = sc.d.xpos[sc.body["red"]], sc.d.xpos[sc.body["blue"]]
    parked = locals().get("parked", {})
    stacked = (abs(float(r[2] - b[2]) - arm.CUBE) < 0.006
               and float(np.hypot(*(r[:2] - b[:2]))) < 0.014)
    return dict(stacked=stacked, dz=float(r[2] - b[2]),
                dxy=float(np.hypot(*(r[:2] - b[:2]))), max_tilt=max_tilt,
                walk_s=handoff_t, total_s=t, frames=frames, records=records,
                red=red, blue=blue, **parked)


def parked_geometry(sc, red, blue):
    """Where the cubes sit in the body frame at the moment of handoff.

    This is what decides whether the arm can do its half of the job, so it is
    worth recording even when the stack succeeds.
    """
    R = sc.d.xmat[sc.base].reshape(3, 3)
    out = {}
    for name, xy in (("red", red), ("blue", blue)):
        p = R.T @ (np.array([xy[0], xy[1], arm.CUBE / 2]) - sc.d.xpos[sc.base])
        out[f"{name}_range"] = float(np.hypot(p[0], p[1]))
        out[f"{name}_bearing"] = math.degrees(math.atan2(p[1], p[0]))
    return out


def sample(sc, cams, step, ctrl, t_rel):
    """One VLA training sample: what the robot sees, is, and was told to do."""
    views = {}
    for name, ren in (cams or {}).items():
        ren.update_scene(sc.d, camera=name)
        views[name] = ren.render().copy()
    return dict(t=t_rel, phase=step.phase, ctrl=ctrl.copy(),
                qpos=sc.d.qpos.copy(), qvel=sc.d.qvel.copy(), **views)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None, help="write an MP4 of the first episode")
    ap.add_argument("--no-video", dest="video", action="store_const", const=None)
    ap.add_argument("--dataset", default=None, help="write an .npz of manipulation demos")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--record-dt", type=float, default=0.05, help="dataset sample period, s")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam-width", type=int, default=224)
    ap.add_argument("--cam-height", type=int, default=224)
    ap.add_argument("--camera", default="scene", help="camera for the walk")
    ap.add_argument("--manip-camera", default="closeup", help="camera from the handoff on")
    ap.add_argument("--stand-off", type=float, default=ApproachParams.stand_off)
    ap.add_argument("--walk-timeout", type=float, default=45.0)
    ap.add_argument("--cubes", type=float, nargs=4, default=None,
                    metavar=("RX", "RY", "BX", "BY"))
    ap.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scratch = Scene()
    renderer = cams = None
    if args.video:
        renderer = mujoco.Renderer(scratch.m, height=args.height, width=args.width)
    if args.dataset:
        cams = {n: mujoco.Renderer(scratch.m, height=args.cam_height, width=args.cam_width)
                for n in ("head", "wrist")}

    ok = 0
    episodes = []
    for ep in range(args.episodes):
        res = run(args, rng, renderer if ep == 0 else None, cams)
        ok += res["stacked"]
        episodes.append(res)
        print(f"  ep {ep:3d}  walk {res['walk_s']:5.1f}s  total {res['total_s']:5.1f}s  "
              f"dz {res['dz']*1000:6.1f} mm  dxy {res['dxy']*1000:5.1f} mm  "
              f"tilt {res['max_tilt']:4.1f} deg  "
              f"parked R {res.get('red_range', 0)*1000:3.0f}/{res.get('blue_range', 0)*1000:3.0f} mm "
              f"th {res.get('red_bearing', 0):+5.1f}/{res.get('blue_bearing', 0):+5.1f}  "
              f"{'stacked' if res['stacked'] else 'FAILED'}",
              flush=True)
    print(f"\n{ok}/{args.episodes} stacked")

    if args.video and episodes[0]["frames"]:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, episodes[0]["frames"], fps=args.fps,
                        codec="libx264", quality=8)
        print(f"wrote {args.video}  ({len(episodes[0]['frames'])} frames)")

    if args.dataset:
        recs = [r for e in episodes for r in e["records"]]
        Path(args.dataset).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset,
            episode=np.concatenate([np.full(len(e["records"]), i) for i, e in enumerate(episodes)]),
            t=np.array([r["t"] for r in recs], np.float32),
            phase=np.array([r["phase"] for r in recs]),
            ctrl=np.array([r["ctrl"] for r in recs], np.float32),
            qpos=np.array([r["qpos"] for r in recs], np.float32),
            qvel=np.array([r["qvel"] for r in recs], np.float32),
            head=np.array([r["head"] for r in recs], np.uint8),
            wrist=np.array([r["wrist"] for r in recs], np.uint8),
            actuators=np.array(scratch.names),
        )
        print(f"wrote {args.dataset}  ({len(recs)} samples)")
    return 0 if ok == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
