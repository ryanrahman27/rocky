# Rocky

A five-legged walking robot, from CAD to a trained locomotion policy.

Rocky is a pentapod: a 250 mm pentagonal body carrying five identical 342 mm
limbs at 72° spacing, one of which doubles as a manipulator. This repo holds the
simulation model exported from the Fusion 360 master, an analytic wave gait, and
a PPO locomotion task for [mjlab](https://github.com/mujocolab/mjlab).

![Rocky walking with the scripted wave gait](docs/gait_wave.gif)

*Open-loop wave gait in MuJoCo. One foot swings at a time, in the order
0 → 2 → 4 → 1 → 3.*

## Quick start

```bash
uv sync --extra gait                       # mujoco only, no CUDA needed
uv run python scripts/play_gait.py --vx 0.052 --seconds 12
uv run python -m mujoco.viewer --mjcf=model/rocky_standalone.xml
```

Training needs Linux + an NVIDIA GPU (mjlab runs on MuJoCo-Warp):

```bash
uv sync                                    # pulls mjlab, torch, warp
uv run list-envs | grep Rocky
uv run play  Mjlab-Velocity-Flat-Rocky --agent zero      # sanity: it should stand
uv run train Mjlab-Velocity-Flat-Rocky --env.scene.num-envs 4096 --agent.max-iterations 3000
uv run play  Mjlab-Velocity-Flat-Rocky --wandb-run-path <org/project/run-id>
```

## Layout

```
model/          MJCF + URDF exported from Fusion, per-link meshes, collision hulls
  rocky.xml            the model mjlab loads (no actuators; CAD "home" keyframe)
  rocky_standalone.xml + floor, light and position servos, for plain MuJoCo
  rocky.urdf           for Isaac Lab / pinocchio / ROS
  build_model.py       regenerates all of the above from the Fusion export
src/rocky/
  model_params.py      lengths, limits, servo constants (asserted against the MJCF)
  kinematics.py        per-leg analytic FK/IK
  gait.py              the wave gait
  tasks/               mjlab task: entity, env cfgs, PPO cfg
  tasks/mdp/           gait-phase observation and reward terms
scripts/        play_gait.py (open loop), train.sh, play.sh
docs/           model.md (the robot), gait.md (the gait and what limits it)
tests/          kinematics, gait invariants, model/MJCF agreement, reward maths
```

## The robot

4.170 kg all-up, including 430 g of declared payload. 16 actuated joints: five
legs × (sweep, lift, elbow), plus a wrist on the manipulator limb, whose gripper
is welded shut so its rubber pad serves as the fifth foot. Every joint is a
Feetech STS3215 — 2.9 N·m, 1:345, and compliant enough that it shapes how the
robot walks (see below). [docs/model.md](docs/model.md) has the joint table,
mass breakdown and modelling assumptions.

## The gait

A wave gait: one foot swings at a time while the other four stay planted, so the
support polygon always contains the body axis with ~80 mm of margin. Swing order
is the star sequence 0 → 2 → 4 → 1 → 3, which puts consecutive swing legs 144°
apart rather than 72°, keeping the support shift symmetric.

```python
from rocky.gait import WaveGait, GaitParams
gait = WaveGait(GaitParams(period=2.4, stance_height=0.065))
q = gait.action_vector(t=1.3, vx=0.05)        # 16 joint targets
```

Defaults: 2.4 s period, 0.8 duty, 30 mm step height, 65 mm stance height, 582 mm
footprint. Nominal cruise is 52 mm/s from a 100 mm stride.

**Open loop it reaches about 78% of the commanded speed**, and that shortfall is
the servos, not the gait. Raising friction to μ=5 changes nothing; raising the
servo gain to kp=60 recovers 92%. The stance legs push the body forward, the
reaction deflects the position servos, and roughly a fifth of the commanded
stroke disappears into that deflection. It is the clearest argument in the repo
for closing the loop — which is what the policy does.
[docs/gait.md](docs/gait.md) works through the measurements.

## The RL task

`Mjlab-Velocity-Flat-Rocky` is mjlab's velocity-tracking task with two additions
that carry the gait over:

- **A gait clock** in the observation — (sin, cos) of the cycle phase, so the
  policy knows where in the rhythm it is.
- **A contact-schedule reward** — each foot is rewarded for being on the ground
  exactly when the wave gait says it should be. Gated by command magnitude, so
  standing still means all five feet planted rather than marching on the spot.

The policy chooses its own joint motion; only the *timing* is prescribed. Set
`cfg.rewards["gait_contact"].weight = 0.0` to let PPO find its own rhythm, or
raise it to hold the wave gait harder.

Commands span ±0.08 m/s forward, ±0.04 m/s lateral and ±0.5 rad/s yaw — the
envelope the gait analysis says the servos can actually deliver. Actions are
joint-position offsets from the gait's neutral stance.

## Tests

```bash
uv run --extra dev pytest          # or: PYTHONPATH=src pytest
```

Covers IK/FK round-trips and joint limits, the gait invariants (exactly one leg
airborne, correct swing order, support margin), agreement between
`model_params.py` and the compiled MJCF, URDF/MJCF equivalence, and the reward
maths against a stub env.

## Status and next steps

Done: model, gait, task, harness — and the gait verified in MuJoCo. **Not yet
done: the policy itself.** Training has never been run; it needs a CUDA box.
When you do run it, the things most likely to need attention:

- `ROTOR_INERTIA` in `model_params.py` is assumed (Feetech publish no figure) and
  sets armature, kp and kv together. It is the main sim-to-real unknown; consider
  randomizing actuator gains once a policy trains.
- If the policy shuffles instead of stepping, raise `air_time`'s weight from 0.
- If it tracks the schedule but barely moves, the gait period (2.4 s) is the
  ceiling on speed — shorten it, or move to a ripple gait with two legs swinging.
