"""Running the trained PPO locomotion policy without mjlab.

The policy was trained in mjlab, which needs CUDA and MuJoCo-Warp. The stacking
scene is plain MuJoCo on a CPU. This bridges the two: it rebuilds the actor from
an rsl-rl checkpoint -- an observation normaliser and a four-layer ELU MLP, no
framework required -- and reassembles the observation the policy was trained on
out of ordinary MjData.

Every number here was read out of the task config rather than guessed
(`load_env_cfg("Mjlab-Velocity-Flat-Rocky")`), because an observation vector
assembled in the wrong order produces a robot that thrashes, and one assembled
in the right order with the wrong scale produces a robot that walks *badly* --
which is far harder to notice.

    observation (62)                       source
    ---------------------------------------------------------------
     0: 3  base linear velocity            velocimeter at the imu site
     3: 6  base angular velocity           gyro at the imu site
     6: 9  projected gravity               -(world up, in the base frame)
     9:25  joint pos - default             encoders
    25:41  joint velocity                  encoders
    41:57  previous raw action             held here
    57:60  commanded twist                 whatever is driving
    60:62  gait clock                      (sin, cos) of 2*pi*t/2.4

    action (16) -> joint target = default + action * per-joint scale

The clock is why this is stateful: the policy was given a metronome so it could
learn a rhythm, and it counts control steps, not wall time.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import model_params as P
from .gait import GaitParams, WaveGait

#: Control period, from the task's decimation (4) and sim timestep (5 ms).
STEP_DT = 0.02
#: Gait-clock period the policy was trained against, seconds.
CLOCK_PERIOD = 2.4
#: Per-joint action scale, from the task's JointPositionActionCfg.
ACTION_SCALE = {"sweep": 0.25, "lift": 0.4, "elbow": 0.4, "wrist": 0.3}

#: The 16 walking joints, in the order the model declares them. Read by NAME,
#: because rocky_manip.xml inserts the two jaw joints in the middle of this list
#: and a slice would silently shift everything from leg 1 onwards.
JOINT_NAMES = tuple(
    n for k in range(P.N_LEGS)
    for n in ((f"sweep_{k}", f"lift_{k}", f"elbow_{k}")
              + ((f"wrist_{k}",) if k == P.MANIP_LEG else ()))
)


def default_joint_pos(gait: WaveGait | None = None) -> np.ndarray:
    """The stance the policy's actions are offsets from.

    This has to be the entity's init state from training, to the last decimal:
    the action is added to it, so an error here is a permanent posture bias the
    policy never saw and cannot correct for.
    """
    neutral = (gait or WaveGait(GaitParams())).neutral_joint_dict()
    return np.array([neutral[n] for n in JOINT_NAMES])


def action_scales() -> np.ndarray:
    return np.array([ACTION_SCALE[n.rsplit("_", 1)[0]] for n in JOINT_NAMES])


def load_actor(path: str | Path):
    """Rebuild the actor from an rsl-rl checkpoint: normaliser plus ELU MLP."""
    import torch

    sd = torch.load(str(path), map_location="cpu", weights_only=False)["actor_state_dict"]
    mean = sd["obs_normalizer._mean"].float().squeeze().numpy()
    std = sd["obs_normalizer._std"].float().squeeze().numpy()
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    layers = [(sd[f"mlp.{i}.weight"].float().numpy(), sd[f"mlp.{i}.bias"].float().numpy())
              for i in idx]

    def act(obs: np.ndarray) -> np.ndarray:
        x = (obs - mean) / std
        for i, (w, b) in enumerate(layers):
            x = x @ w.T + b
            if i < len(layers) - 1:
                x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))   # ELU
        return x

    return act, len(mean), layers[-1][1].shape[0]


class PolicyWalker:
    """Drives the 16 walking joints from a trained checkpoint.

    Feed it the same (vx, vy, wz) twist `rocky.approach` produces for the
    scripted gait; it is a drop-in replacement for `WaveGait` in that role.
    """

    def __init__(self, checkpoint, model, gait: WaveGait | None = None):
        import mujoco

        self.act, obs_dim, act_dim = load_actor(checkpoint)
        if obs_dim != 62 or act_dim != len(JOINT_NAMES):
            raise ValueError(f"checkpoint is {obs_dim}->{act_dim}, expected 62->16")
        self.default = default_joint_pos(gait)
        self.scale = action_scales()

        sensor = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): model.sensor_adr[i]
                  for i in range(model.nsensor)}
        for name in ("imu_lin_vel", "imu_ang_vel", "imu_upvector"):
            if name not in sensor:
                raise ValueError(f"model has no {name} sensor; the policy needs it")
        self._lin, self._ang, self._up = (sensor[n] for n in
                                          ("imu_lin_vel", "imu_ang_vel", "imu_upvector"))
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self._qadr = np.array([model.jnt_qposadr[jid(n)] for n in JOINT_NAMES])
        self._vadr = np.array([model.jnt_dofadr[jid(n)] for n in JOINT_NAMES])
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(len(JOINT_NAMES))
        self.steps = 0

    def observe(self, data, command) -> np.ndarray:
        s = data.sensordata
        phase = (self.steps * STEP_DT / CLOCK_PERIOD) % 1.0
        return np.concatenate([
            s[self._lin:self._lin + 3],
            s[self._ang:self._ang + 3],
            -s[self._up:self._up + 3],
            data.qpos[self._qadr] - self.default,
            data.qvel[self._vadr],
            self.last_action,
            np.asarray(command, float)[:3],
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
        ])

    def step(self, data, command) -> dict[str, float]:
        """One control tick: observation in, joint targets out."""
        action = self.act(self.observe(data, command))
        self.last_action = action
        self.steps += 1
        target = self.default + action * self.scale
        return {n: float(v) for n, v in zip(JOINT_NAMES, target)}
