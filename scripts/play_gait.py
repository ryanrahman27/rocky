#!/usr/bin/env python3
"""Drive Rocky's scripted wave gait in plain MuJoCo -- no mjlab, no GPU needed.

    python scripts/play_gait.py --vx 0.075 --seconds 12
    python scripts/play_gait.py --vx 0.05 --wz 0.3 --gif docs/gait.gif

Prints achieved vs commanded velocity, servo tracking error and body sway, which
is the honest check on whether a set of GaitParams is physically realisable.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402

from rocky import ROCKY_STANDALONE_XML, kinematics as K, model_params as P  # noqa: E402
from rocky.gait import GaitParams, WaveGait, support_margin  # noqa: E402


def settle_pose(model, data, gait: WaveGait) -> None:
    """Put the robot in the gait's neutral stance, feet just touching the floor."""
    q = gait.action_vector(0.0, 0.0, 0.0, 0.0)
    data.qpos[:3] = (0.0, 0.0, gait.p.stance_height + 0.0005)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = q
    data.qvel[:] = 0.0
    data.ctrl[:] = q
    mujoco.mj_forward(model, data)


def run(args) -> int:
    gait = WaveGait(
        GaitParams(
            period=args.period,
            duty=args.duty,
            step_height=args.step_height,
            stance_height=args.stance_height,
            footprint_scale=args.footprint,
        )
    )
    feas = gait.check_feasible(args.vx, args.vy, args.wz)
    print(f"gait: period {gait.p.period:.2f}s  duty {gait.p.duty:.2f}  "
          f"stride {gait.stride_length(math.hypot(args.vx, args.vy)) * 1000:.0f}mm  "
          f"step {gait.p.step_height * 1000:.0f}mm  footprint {2 * np.hypot(*gait.neutral[1, :2]) * 1000:.0f}mm")
    print(f"reference feasibility: within_limits={feas['within_limits']}  "
          f"peak joint speed {feas['max_joint_speed']:.2f} rad/s "
          f"({feas['max_joint_speed'] / P.VELOCITY_LIMIT * 100:.0f}% of no-load)")

    model = mujoco.MjModel.from_xml_path(str(ROCKY_STANDALONE_XML))
    data = mujoco.MjData(model)
    settle_pose(model, data, gait)

    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in P.JOINT_NAMES]
    qadr = np.array([model.jnt_qposadr[i] for i in jid])
    foot_sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in P.FOOT_SITES]

    frames, renderer, cam = [], None, None
    if args.gif or args.mp4:
        renderer = mujoco.Renderer(model, args.height, args.width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.elevation, cam.azimuth = 1.0, -18.0, 130.0

    # Hold the stance for a beat so the servos settle, then walk.
    hold = 0.5
    steps = int((args.seconds + hold) / model.opt.timestep)
    log = {k: [] for k in ("t", "x", "y", "yaw", "z", "tilt", "err", "slip", "margin")}
    stance_prev = {}
    for i in range(steps):
        t = data.time - hold
        if t < 0.0:
            target = gait.action_vector(0.0, 0.0, 0.0, 0.0)
        else:
            target = gait.action_vector(t, args.vx, args.vy, args.wz)
        data.ctrl[:] = target
        mujoco.mj_step(model, data)

        if t >= 0.0:
            up = data.xmat[1].reshape(3, 3)[:, 2]
            qm = data.qpos[qadr]
            log["t"].append(t)
            log["x"].append(data.qpos[0])
            log["y"].append(data.qpos[1])
            w, x, y, z = data.qpos[3:7]
            log["yaw"].append(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            log["z"].append(data.qpos[2])
            log["tilt"].append(math.degrees(math.acos(np.clip(up[2], -1, 1))))
            log["err"].append(np.abs(qm - target).max())
            # slip: horizontal motion of feet that should be planted
            sw = gait.in_swing(t)
            slip = 0.0
            for k in range(P.N_LEGS):
                p = data.site_xpos[foot_sid[k]].copy()
                if not sw[k] and k in stance_prev:
                    slip = max(slip, float(np.linalg.norm(p[:2] - stance_prev[k][:2])))
                stance_prev[k] = p if not sw[k] else stance_prev.get(k, p)
                if sw[k]:
                    stance_prev.pop(k, None)
            log["slip"].append(slip)
            log["margin"].append(support_margin(gait.foot_targets(t, args.vx, args.vy, args.wz), sw))

            if renderer is not None and len(frames) < args.seconds * args.fps and \
                    i % max(1, int(1.0 / (args.fps * model.opt.timestep))) == 0:
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.06]
                if args.orbit:
                    cam.azimuth = 130.0 + 20.0 * math.sin(2 * math.pi * t / 10.0)
                renderer.update_scene(data, cam)
                frames.append(renderer.render())

    t = np.array(log["t"])
    dur = t[-1] - t[0]
    vx_ach = (log["x"][-1] - log["x"][0]) / dur
    vy_ach = (log["y"][-1] - log["y"][0]) / dur
    wz_ach = (log["yaw"][-1] - log["yaw"][0]) / dur
    print(f"\ncommanded  vx {args.vx:+.3f}  vy {args.vy:+.3f}  wz {args.wz:+.3f}")
    print(f"achieved   vx {vx_ach:+.3f}  vy {vy_ach:+.3f}  wz {wz_ach:+.3f}"
          f"   ({vx_ach / args.vx * 100:.0f}% of commanded forward)" if args.vx else "")
    print(f"body height {np.mean(log['z']) * 1000:.1f} +- {np.std(log['z']) * 1000:.1f} mm   "
          f"tilt max {max(log['tilt']):.2f} deg   lateral drift {log['y'][-1] * 1000:+.0f} mm")
    print(f"servo tracking error: mean {np.degrees(np.mean(log['err'])):.2f} deg, "
          f"max {np.degrees(np.max(log['err'])):.2f} deg")
    print(f"stance foot slip per step: max {max(log['slip']) * 1000:.1f} mm")
    print(f"support margin: {min(log['margin']) * 1000:.0f}..{max(log['margin']) * 1000:.0f} mm")

    if frames:
        import imageio.v2 as imageio
        out = args.gif or args.mp4
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=args.fps, loop=0) if args.gif else \
            imageio.mimsave(out, frames, fps=args.fps)
        print(f"wrote {out} ({len(frames)} frames)")
    return 0


def main() -> int:
    d = GaitParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vx", type=float, default=0.075, help="forward velocity, m/s")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0, help="yaw rate, rad/s")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--period", type=float, default=d.period)
    ap.add_argument("--duty", type=float, default=d.duty)
    ap.add_argument("--step-height", type=float, default=d.step_height)
    ap.add_argument("--stance-height", type=float, default=d.stance_height)
    ap.add_argument("--footprint", type=float, default=d.footprint_scale)
    ap.add_argument("--gif", type=str, default=None)
    ap.add_argument("--mp4", type=str, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--orbit", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
