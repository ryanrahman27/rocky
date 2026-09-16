#!/usr/bin/env python3
"""Stack two cubes with no cameras: sonar, proprioception and touch only.

    python scripts/blind_stack_demo.py --episodes 10
    python scripts/blind_stack_demo.py --video docs/rocky_blind.mp4

Rocky has no eyes. Eridians perceive shape by sound, which is why he could never
see the stars and why everything he and Grace exchanged had to be handed over
physically. Taking that seriously removes the camera entirely -- and with it the
case for a vision-language policy, since there is no image to condition on. What
is left is a ring of rangefinders standing in for the sonar, the joint encoders,
and two touch pads.

Nothing in this script reads a cube's true pose. The phases:

    listen    turn on the spot until the sonar hears two objects
    walk      close on them CRABWISE, holding them 36 deg off the body axis so
              they stay in the gap between two limbs and stay audible
    square    drop the crab and turn to face them; they go behind the front limb
              and the belief coasts on IMU dead reckoning
    feel      brace, lift the arm, and sweep the gripper's own fan across the
              front until both cubes are located to a couple of centimetres
    stack     pick and place, with the grasp confirmed by the touch pads --
              a blind robot's only way of knowing whether it actually got it,
              and the thing that lets it try again when it did not
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocky import arm, model_params as P                         # noqa: E402
from rocky.approach import Approach, ApproachParams, spawn_pose   # noqa: E402
from rocky.gait import WaveGait                                   # noqa: E402
from rocky.locomotion import PolicyWalker                         # noqa: E402
from rocky.kinematics import limb_points                          # noqa: E402
from rocky.sonar import Sonar, PairTracker, _cluster, observation  # noqa: E402
from rocky.stack import StackController, StackPlan                # noqa: E402
from stack_demo import Scene, gait_joints, layout                 # noqa: E402

SWEEP_LIMITS = (-30.0, 30.0)      # arm sweep covered by the feel scan, deg
SWEEP_RATE = 26.0                 # deg per second
RECENTRE_S = 0.7                  # bringing the arm back before the pick
LONE_PATIENCE = 16.0              # seconds before settling for one object
SEARCH_PERIOD = 8.0               # seconds per yaw sweep while searching
FEEL_MIN_HITS = 3                 # fewer than this and it is noise, not a cube
GRIP_FORCE = 0.5                  # N on both pads = something is in the hand
CONTROL_HZ = 50.0                 # the servo bus rate; also 4x faster to simulate
SONAR_HZ = 20.0                   # a sonar ping is not free, in silicon or in life


def _carry_xy(p, dt, wz, v):
    c, s = math.cos(-wz * dt), math.sin(-wz * dt)
    return (np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
            - np.asarray(v, float)[:2] * dt)


def parked_pose(rng, red, blue, prior_sigma: float = 0.020):
    """A robot already standing where the walk-up would have left it.

    Generating manipulation data does not need the approach re-simulated every
    time -- it is the locomotion controller's job and it triples the cost of an
    episode. This puts the robot at a randomised stand-off instead, spread wider
    than the approach controller actually parks, so the data covers poses the
    script would rarely produce.

    The belief it starts with is the TRUE pair blurred by 20 mm, which is what
    the body ring's own accuracy is. That stands in for the handover, not for
    perception: the feel sweep still has to find the cubes properly, and it is
    the sweep's 5 mm answer that the pick is actually made from.
    """
    red, blue = np.asarray(red, float), np.asarray(blue, float)
    mid = (red + blue) / 2.0
    axis = red - blue
    n = np.array([axis[1], -axis[0]])
    n = n / max(float(np.hypot(*n)), 1e-9)
    if float(n @ -mid) < 0.0:
        n = -n
    stand = rng.uniform(0.330, 0.400)
    base = mid + stand * n
    yaw = math.atan2(*(mid - base)[::-1]) + rng.uniform(-0.10, 0.10)
    c, s_ = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, s_], [-s_, c]])                      # world -> body
    belief = type("P", (), {
        "left": R @ (red - base) + rng.normal(0.0, prior_sigma, 2),
        "right": R @ (blue - base) + rng.normal(0.0, prior_sigma, 2),
    })()
    return base, yaw, belief


def _lone_pair(target):
    """Stand in for a pair when only one object was ever heard."""
    t = np.asarray(target, float)
    n = np.array([-t[1], t[0]])
    n = n / max(float(np.hypot(*n)), 1e-9) * 0.065
    return type("P", (), {"left": t + n, "right": t - n})()


class Blind:
    """The state machine. Egocentric throughout: no world frame anywhere."""

    def __init__(self, scene: Scene, plan: StackPlan | None = None, retries: int = 2,
                 noise: float = 0.0, rng=None, locomotion=None):
        self.sc = scene
        self.sonar = Sonar(scene.m)
        self.track = PairTracker(self.sonar)
        self.gait = WaveGait()
        #: The trained PPO policy, if one was supplied, else the analytic gait.
        #: Both consume the same twist, which is the whole point of `approach`
        #: producing one rather than joint angles.
        self.walker = None if locomotion is None else PolicyWalker(locomotion, scene.m)
        self.appr = Approach(ApproachParams(), self.gait)
        self.stack = StackController(plan or StackPlan())
        self.retries = retries
        #: Gaussian jitter on the arm targets. A demonstrator that never wobbles
        #: teaches a policy nothing about recovering from a wobble, and the first
        #: time it lands slightly off-nominal it has no idea what to do.
        self.noise = noise
        self.frame_belief = True
        self.belief = np.zeros(4)
        self.last_joints: dict[str, float] = {}
        self.rng = rng if rng is not None else np.random.default_rng(0)
        sid = lambda n: mujoco.mj_name2id(scene.m, mujoco.mjtObj.mjOBJ_SENSOR, n)  # noqa: E731
        self.gyro = scene.m.sensor_adr[sid("imu_ang_vel")]
        self.vel = scene.m.sensor_adr[sid("imu_lin_vel")]

    # -- what the robot can feel -------------------------------------------
    def imu(self):
        d = self.sc.d
        return float(d.sensordata[self.gyro + 2]), d.sensordata[self.vel:self.vel + 2].copy()

    def locomote(self, t: float, vx: float, vy: float, wz: float) -> dict[str, float]:
        """Joint targets for the walking half, from whichever controller is in charge."""
        if self.walker is None:
            return gait_joints(self.gait, t, vx, vy, wz)
        joints = self.walker.step(self.sc.d, (vx, vy, wz))
        joints["jaw_r"] = joints["jaw_l"] = arm.JAW_OPEN
        return joints

    def limbs(self):
        """Where the robot's own knees and feet are, from its joint encoders."""
        d, qa = self.sc.d, self.sc.qadr
        q = np.array([[d.qpos[qa[f"sweep_{k}"]], d.qpos[qa[f"lift_{k}"]], d.qpos[qa[f"elbow_{k}"]]]
                      for k in range(P.N_LEGS)])
        return limb_points(q)

    def run(self, log=None, on_step=None, timeout: float = 90.0, debug=None, record=None,
            start_phase: str = "listen", seed_pair=None, stop_at: str | None = None):
        sc, d = self.sc, self.sc.d
        t, dt = 0.0, sc.dt
        n_ctrl = max(1, int(round(1.0 / (CONTROL_HZ * dt))))
        # The locomotion policy reads the IMU every control tick, so the
        # sensor stage cannot be skipped while it is driving.
        n_sens = n_ctrl if self.walker is not None else max(1, int(round(1.0 / (SONAR_HZ * dt))))
        sensor_bit = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        joints = gait_joints(self.gait, 0.0, 0.0, 0.0, 0.0)
        tick = 0
        phase, t_phase = start_phase, 0.0
        self.track.reset()
        if self.walker is not None:
            self.walker.reset()
        pair = None
        self.parked = _lone_pair((0.36, 0.0)) if seed_pair is None else seed_pair
        self.lone = None
        if start_phase != "listen":
            self.stack.reset(self.parked.left, self.parked.right,
                             floor_z=self.sonar.floor_z(d), frame="body")
            self.belief = np.concatenate([self.parked.left, self.parked.right])
        red = blue = None
        attempt = 0
        stack_t0 = 0.0
        result = dict(phase_log=[], attempts=0, grip_checked=False, gripped=False)
        self.belief = np.zeros(4)

        while t < timeout:
            if tick % n_ctrl:
                sc.d.ctrl[:] = [joints[n] for n in sc.names]
                mujoco.mj_step(sc.m, sc.d)
                t += dt
                tick += 1
                if on_step is not None:
                    on_step(t, phase, joints)
                continue
            cdt = n_ctrl * dt
            wz, v = self.imu()
            pair = self.track.update(d, cdt, wz, v, limbs=self.limbs())
            if debug is not None:
                debug(t, phase, pair)

            if phase == "listen":
                # Turn so the gaps sweep across whatever is out there, and creep
                # forward while doing it: at the far end of a spawn a cube is
                # 2 deg wide against a 5 deg ray spacing and is only heard now
                # and then. Closing the range is what makes it audible. Once one
                # object is firm, steer at it; once two are, start the approach.
                seen = self.track.candidates()
                if seen:
                    b0 = math.atan2(seen[0][1], seen[0][0])
                    wz_cmd = float(np.clip(1.2 * b0, -0.3, 0.3))
                    vx_cmd = 0.03 if abs(b0) < math.radians(40.0) else 0.0
                else:
                    # Turning and creeping at once traces a circle of radius
                    # v/w -- 60 mm at these rates, which is to say the robot
                    # spends the whole episode standing still while appearing to
                    # search. Alternate instead: turn to sweep the gaps across
                    # the room, then walk to change where it listens from.
                    # Walk and sweep at the same time. Turning on the spot and
                    # creeping traces a 60 mm circle; alternating legs is a
                    # random walk that covers almost no ground. Cruising ahead
                    # while yawing back and forth both closes the range -- the
                    # sonar only reaches about 1.3 m -- and sweeps the gaps
                    # across whatever is out there.
                    wz_cmd = 0.25 * math.sin(2.0 * math.pi * t / SEARCH_PERIOD)
                    vx_cmd = 0.05
                joints = self.locomote(t, vx_cmd, 0.0, wz_cmd)
                if pair is not None and pair.separation > 0.06:
                    self.appr.reset()
                    phase, t_phase = "walk", t
                elif seen and t > LONE_PATIENCE and float(np.hypot(*seen[0])) < 0.75:
                    # Only ever heard one of them. Walk up to that one anyway and
                    # let the hand sort out what is actually there -- the feel
                    # sweep sees both cubes from close up whatever the ring did.
                    self.lone = seen[0].copy()
                    self.appr.reset()
                    phase, t_phase = "walk", t
            elif phase == "walk":
                target, axis = None, None
                if pair is not None:
                    target, axis = pair.midpoint, pair.axis
                elif self.lone is not None:
                    seen = self.track.candidates()
                    if seen:
                        self.lone = seen[0].copy()
                    else:
                        self.lone = _carry_xy(self.lone, cdt, wz, v)
                    target = self.lone
                    axis = np.array([-target[1], target[0]])   # face it squarely
                if target is None:
                    joints = self.locomote(t, 0.0, 0.0, 0.25)
                else:
                    r = float(np.hypot(*target))
                    crab = None if r > self.appr.p.square_up_range else 0.0
                    tw = self.appr.command_body(t, target, axis, crab)
                    joints = self.locomote(t, tw.vx, tw.vy, tw.wz)
                    if tw.arrived:
                        self.parked = pair if pair is not None else _lone_pair(target)
                        phase, t_phase = "settle", t
            elif phase == "settle":
                joints = self.locomote(t, 0.0, 0.0, 0.0)
                if t - t_phase > 0.8:
                    held = pair if pair is not None else self.parked
                    # In the body frame the floor is not at z = 0, it is a ride
                    # height below the origin -- and the sonar is what knows how
                    # far. Leaving it at zero puts every waypoint 78 mm up in the
                    # air and the jaws close on nothing at all.
                    self.stack.reset(held.left, held.right,
                                     floor_z=self.sonar.floor_z(d), frame="body")
                    self.belief = np.concatenate([held.left, held.right])
                    phase, t_phase = "brace", t
            elif phase == "brace":
                self.stack.carry(cdt, wz, v)
                step = self.stack.step(t - t_phase, None, None, sc.arm_q())
                joints = step.joints
                if t - t_phase >= self.stack.time_of("to_red"):
                    self._scan_reset(t)
                    phase, t_phase = "feel", t
            elif phase == "feel":
                joints, done = self._scan_step(t - t_phase, pair)
                if done:
                    red, blue = self._scan_result(pair if pair is not None else self.parked)
                    self.stack.reset(red, blue, floor_z=self.sonar.floor_z(d), frame="body")
                    self.belief = np.concatenate([red, blue])
                    stack_t0 = self.stack.time_of("to_red")
                    phase, t_phase = "stack", t - stack_t0
            else:                                   # stack
                self.stack.carry(cdt, wz, v)
                rel = t - t_phase
                step = self.stack.step(rel, None, None, sc.arm_q())
                joints = step.joints
                # The one moment a blind robot can check its own work.
                if step.phase == "lift" and step.progress > 0.5 and not result["grip_checked"]:
                    result["grip_checked"] = True
                    result["gripped"] = self.sonar.gripping(d, GRIP_FORCE)
                    if not result["gripped"] and attempt < self.retries:
                        attempt += 1
                        result["attempts"] = attempt
                        self._scan_reset(t)
                        phase, t_phase = "feel", t
                        result["grip_checked"] = False
                        continue
                if rel >= self.stack.duration:
                    break

            if self.frame_belief and phase in ("brace", "feel", "stack"):
                self.belief[:2] = _carry_xy(self.belief[:2], dt, wz, v)
                self.belief[2:] = _carry_xy(self.belief[2:], dt, wz, v)
            if stop_at is not None and phase == stop_at:
                # Hand over mid-episode: the caller takes the robot from here.
                self.last_joints = dict(joints)
                result.update(t=t, red=red, blue=blue, phase=phase)
                return result
            if self.noise > 0.0 and phase in ("feel", "stack"):
                joints = dict(joints)
                for n in ("sweep_0", "lift_0", "elbow_0", "wrist_0"):
                    joints[n] = float(joints[n] + self.rng.normal(0.0, self.noise))
            self.last_joints = dict(joints)
            if record is not None:
                record(t, phase, joints)
            sc.apply(joints)
            mujoco.mj_step(sc.m, sc.d)
            t += dt
            tick += 1
            if on_step is not None:
                on_step(t, phase, joints)
            if log is not None and (not result["phase_log"] or result["phase_log"][-1][1] != phase):
                result["phase_log"].append((round(t, 2), phase))

        result.update(t=t, red=red, blue=blue, phase=phase)
        return result

    # -- the feel sweep ----------------------------------------------------
    def _scan_reset(self, t: float) -> None:
        self._hits: list[np.ndarray] = []
        self._ready = None

    def _scan_step(self, rel: float, pair):
        """Swing the arm across the front, listening through the open hand."""
        lo, hi = SWEEP_LIMITS
        span = (hi - lo) / SWEEP_RATE
        if self._ready is None:
            base = self.stack.step(self.stack.time_of("to_red") - 1e-6, None, None, self.sc.arm_q())
            self._ready = np.array([base.joints[n] for n in
                                    ("sweep_0", "lift_0", "elbow_0", "wrist_0")])
            self._legs = {k: v for k, v in base.joints.items() if not k.endswith("_0")}
        # Sweep across, then bring the arm back to where the pick starts from.
        # Jumping straight from the far end of the scan into the first Cartesian
        # waypoint is a 30 deg step and the servos slam through it.
        frac = min(1.0, rel / span)
        sweep = math.radians(lo + (hi - lo) * frac)
        if rel > span:
            k = min(1.0, (rel - span) / RECENTRE_S)
            sweep = math.radians(hi) + (self._ready[0] - math.radians(hi)) * (k * k * (3 - 2 * k))
        joints = dict(self._legs)
        joints["sweep_0"] = sweep
        joints["lift_0"], joints["elbow_0"], joints["wrist_0"] = self._ready[1:]
        joints["jaw_r"] = joints["jaw_l"] = arm.JAW_OPEN
        # The base pose here is only a frame conversion: site-in-world minus
        # base-in-world is a function of the joint angles and nothing else, so
        # this is the same number forward kinematics gives on hardware.
        d = self.sc.d
        if rel <= span:
            for p in self.sonar.feel_hits(d, d.xpos[self.sc.base], d.xquat[self.sc.base]):
                self._hits.append(p[:2])
        return joints, rel >= span + RECENTRE_S

    def _scan_result(self, pair):
        """Two cube centres out of the sweep, falling back on the ring's belief."""
        pts = np.array(self._hits) if self._hits else np.zeros((0, 2))
        groups = [g for g in _cluster(pts, 0.05) if len(g) >= FEEL_MIN_HITS]
        cents = sorted((pts[g].mean(axis=0) for g in groups),
                       key=lambda c: math.atan2(c[1], c[0]), reverse=True)
        if len(cents) >= 2:
            return cents[0], cents[-1]
        return pair.left, pair.right


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--dataset", default=None, help="write an .npz of blind demos")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="rad of jitter on the arm targets, for recovery data")
    ap.add_argument("--record-dt", type=float, default=0.05)
    ap.add_argument("--skip-walk", action="store_true",
                    help="spawn already parked; generates manipulation data 3x faster")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--wide", action="store_true", help="randomise past the success envelope")
    ap.add_argument("--locomotion", default=None,
                    help="rsl-rl checkpoint to walk with; omit for the analytic gait")
    ap.add_argument("--timeout", type=float, default=110.0)
    ap.add_argument("--cubes", type=float, nargs=4, default=None)
    ap.add_argument("--start", type=float, nargs=3, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 7919 * args.shard)
    ok = 0
    frames = []
    episodes_out = []
    for ep in range(args.episodes):
        sc = Scene()
        red, blue = layout(rng, args)
        seed_pair, start_phase = None, "listen"
        if args.start is not None:
            base_xy, yaw = np.array(args.start[:2]), args.start[2]
        elif args.skip_walk:
            base_xy, yaw, seed_pair = parked_pose(rng, red, blue)
            start_phase = "brace"
        else:
            base_xy, yaw = spawn_pose(rng, (red + blue) / 2.0)
        b = Blind(sc, retries=args.retries, noise=args.noise,
                  rng=np.random.default_rng(args.seed * 1000 + ep),
                  locomotion=args.locomotion)
        sc.reset(b.gait.neutral_joint_targets(), base_xy, yaw, red, blue)
        for _ in range(300):
            sc.apply(gait_joints(b.gait, 0.0, 0.0, 0.0, 0.0))
            mujoco.mj_step(sc.m, sc.d)
        if b.walker is not None:
            b.walker.reset()

        renderer = None
        if args.video and ep == 0:
            renderer = mujoco.Renderer(sc.m, height=args.height, width=args.width)
            nxt = [0.0]

            def grab(t, phase, joints):
                if t >= nxt[0]:
                    nxt[0] += 1.0 / args.fps
                    renderer.update_scene(sc.d, camera="scene" if phase in ("listen", "walk")
                                          else "closeup")
                    frames.append(renderer.render().copy())
        else:
            grab = None

        samples = []
        if args.dataset:
            obs_names = sc.names
            nxt_rec = [0.0]

            def record(t, phase, joints):
                if t < nxt_rec[0]:
                    return
                nxt_rec[0] = t + args.record_dt
                samples.append((t, phase,
                                observation(b.sonar, sc.d, sc.qadr, obs_names, b.belief),
                                np.array([joints[n] for n in obs_names], np.float32)))
        else:
            record = None

        res = b.run(log=True, on_step=grab, timeout=args.timeout, record=record,
                    start_phase=start_phase, seed_pair=seed_pair)
        for _ in range(300):
            mujoco.mj_step(sc.m, sc.d)
        r, bl = sc.d.xpos[sc.body["red"]], sc.d.xpos[sc.body["blue"]]
        stacked = (abs(float(r[2] - bl[2]) - arm.CUBE) < 0.006
                   and float(np.hypot(*(r[:2] - bl[:2]))) < 0.014)
        ok += stacked
        phases = " ".join(f"{p}@{tt:.0f}" for tt, p in res["phase_log"])
        print(f"  ep {ep:3d}  {res['t']:5.1f}s  dz {(r[2]-bl[2])*1000:6.1f} mm  "
              f"dxy {np.hypot(*(r[:2]-bl[:2]))*1000:5.1f} mm  "
              f"grip {'yes' if res['gripped'] else 'NO '}  retries {res['attempts']}  "
              f"{'stacked' if stacked else 'FAILED'}   [{phases}]", flush=True)
        if args.dataset:
            episodes_out.append((ep, stacked, samples))

    print(f"\n{ok}/{args.episodes} stacked, blind")
    if args.dataset:
        keep = [(ep, s_) for ep, good, s_ in episodes_out if good]
        flat = [row for _, s_ in keep for row in s_]
        Path(args.dataset).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dataset,
            episode=np.array([ep for ep, s_ in keep for _ in s_], np.int32),
            t=np.array([r[0] for r in flat], np.float32),
            phase=np.array([r[1] for r in flat]),
            obs=np.stack([r[2] for r in flat]) if flat else np.zeros((0, 0), np.float32),
            action=np.stack([r[3] for r in flat]) if flat else np.zeros((0, 0), np.float32),
            actuators=np.array(Scene().names),
        )
        print(f"wrote {args.dataset}  ({len(flat)} samples from {len(keep)} successful episodes)")
    if args.video and frames:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        # The rangefinder rays render as lines, which looks great and compresses
        # terribly -- straight out of imageio this clip is 110 MB.
        imageio.mimsave(args.video, frames, fps=args.fps, codec="libx264",
                        output_params=["-crf", "30", "-preset", "slow"],
                        pixelformat="yuv420p")
        print(f"wrote {args.video}  ({len(frames)} frames)")
    return 0 if ok == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
