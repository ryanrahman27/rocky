"""Velocity-tracking environments for Rocky, shaped toward the wave gait."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from rocky import model_params as P
from rocky.gait import GaitParams
from rocky.tasks.constants import ROCKY_ACTION_SCALE, get_rocky_robot_cfg
from rocky import gait_mdp

GAIT = GaitParams()

# Command envelope, from the gait feasibility study in docs/gait.md. A one-leg-
# at-a-time wave gait moves the swing foot at five times body speed, so the
# servos -- not the geometry -- set the ceiling: about 0.078 m/s at this period.
# The open-loop script reaches 0.042 m/s; the range leaves the policy room to
# beat that without asking for speeds the hardware cannot reach.
#
# Yaw and translation compete for the same foot workspace: the scripted gait
# turns at 0.4 rad/s on the spot but only 0.08 rad/s at cruise. Commands are
# sampled independently, so the ceiling here is deliberately below the standing
# limit -- otherwise a large part of the command distribution is unachievable.
MAX_FWD = 0.08
MAX_LAT = 0.04
MAX_YAW = 0.3

#: Below this combined |v| + |w| a command counts as "stand still".
CMD_DEADBAND = 0.012


def rocky_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rocky velocity tracking on generated rough terrain."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.scene.entities = {"robot": get_rocky_robot_cfg()}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = P.BASE_BODY
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in P.FOOT_SITES
            )
            # Ring a little wider than the 12 mm foot sphere.
            sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=P.FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Rocky rides ~55 mm off the ground, so carapace contact is the clear
    # "bottomed out" signal.
    base_ground_cfg = ContactSensorCfg(
        name="base_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=("base_collision",)),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, base_ground_cfg)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ROCKY_ACTION_SCALE

    cfg.viewer.body_name = P.BASE_BODY
    cfg.viewer.distance = 1.2
    cfg.viewer.elevation = -15.0

    # --- gait clock: the policy needs to know where it is in the cycle -------
    clock = ObservationTermCfg(func=gait_mdp.gait_clock, params={"period": GAIT.period})
    cfg.observations["actor"].terms["gait_clock"] = clock
    cfg.observations["critic"].terms["gait_clock"] = clock

    # --- gait shaping -------------------------------------------------------
    # Weight 2.5 puts the schedule on a par with velocity tracking (2.0 + 2.0).
    # At 1.0 a policy that tracks velocity with a shuffle scores ~93% of the
    # velocity reward and ~75% of this one, and has no reason to restructure its
    # gait; the last quarter of the gait reward has to be worth more than the
    # velocity tracking it costs to reorganise.
    cfg.rewards["gait_contact"] = RewardTermCfg(
        func=gait_mdp.gait_contact_schedule,
        weight=2.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "period": GAIT.period,
            "swing_order": GAIT.swing_order,
            "swing_fraction": GAIT.swing_fraction,
        },
    )
    cfg.rewards["gait_swing_height"] = RewardTermCfg(
        func=gait_mdp.gait_swing_clearance,
        weight=-1.5,
        params={
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "period": GAIT.period,
            "swing_order": GAIT.swing_order,
            "swing_fraction": GAIT.swing_fraction,
            "step_height": GAIT.step_height,
            "foot_radius": float(P.FOOT_RADIUS[1]),
        },
    )

    # Per-axis foot friction randomization (feet are condim 6).
    cfg.events.pop("foot_friction", None)
    for name, axes, rng, dist in (
        ("foot_friction_slide", [0], (0.3, 1.5), "uniform"),
        ("foot_friction_spin", [1], (1e-4, 2e-2), "log_uniform"),
        ("foot_friction_roll", [2], (1e-5, 5e-3), "log_uniform"),
    ):
        params = {
            "asset_cfg": SceneEntityCfg("robot", geom_names=P.FOOT_GEOMS),
            "operation": "abs",
            "axes": axes,
            "ranges": rng,
            "shared_random": True,
        }
        if dist != "uniform":
            params["distribution"] = dist
        cfg.events[name] = EventTermCfg(
            mode="startup", func=envs_mdp.dr.geom_friction, params=params
        )
    cfg.events["base_com"].params["asset_cfg"].body_names = (P.BASE_BODY,)

    # Posture: tight on sweep (it sets the footprint), loose on lift/elbow which
    # do the stepping. The wrist is along for the ride, so pin it.
    cfg.rewards["pose"].params["std_standing"] = {
        P.SWEEP_RE: 0.05, P.LIFT_RE: 0.05, P.ELBOW_RE: 0.10, P.WRIST_RE: 0.05,
    }
    walking = {P.SWEEP_RE: 0.20, P.LIFT_RE: 0.30, P.ELBOW_RE: 0.60, P.WRIST_RE: 0.10}
    cfg.rewards["pose"].params["std_walking"] = walking
    cfg.rewards["pose"].params["std_running"] = walking

    cfg.rewards["upright"].params["asset_cfg"].body_names = (P.BASE_BODY,)
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (P.BASE_BODY,)
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = P.FOOT_SITES
    cfg.rewards["foot_clearance"].params["target_height"] = GAIT.step_height
    cfg.rewards["foot_swing_height"].params["target_height"] = GAIT.step_height

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    # Air time is what the gait schedule already dictates; leave it off unless
    # the policy starts shuffling.
    cfg.rewards["air_time"].weight = 0.0

    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-MAX_FWD * 0.6, MAX_FWD)
    twist.ranges.lin_vel_y = (-MAX_LAT, MAX_LAT)
    twist.ranges.ang_vel_z = (-MAX_YAW, MAX_YAW)

    # mjlab's velocity task is tuned for quadrupeds that cruise at 1-3 m/s, and
    # several of its numbers are in velocity units. Rocky tops out at 0.08 m/s,
    # so left alone they mean something entirely different here:
    #
    #   rel_forward_envs  20% of envs get lin_vel_x forced to clamp(min=0.3),
    #                     hardcoded in the command term -- 4x Rocky's top speed,
    #                     unreachable no matter what ranges say. Turned off; the
    #                     sampled range already covers straight-line walking.
    #   command_threshold the "is it being asked to move?" gate on the foot and
    #                     landing rewards. At 0.05 it sits inside Rocky's command
    #                     range, so those rewards would switch on and off mid-range.
    #   walking_threshold the posture reward's standing/walking switch. At 0.05
    #                     Rocky counts as standing while walking, and the tight
    #                     standing posture std then fights the gait.
    #
    # CMD_DEADBAND just has to separate "commanded to stand" (exactly zero) from
    # "commanded to move" (anything Rocky can actually do).
    twist.rel_forward_envs = 0.0
    for term, key in (
        ("pose", "walking_threshold"),
        ("foot_clearance", "command_threshold"),
        ("foot_swing_height", "command_threshold"),
        ("foot_slip", "command_threshold"),
        ("soft_landing", "command_threshold"),
        ("air_time", "command_threshold"),
    ):
        if term in cfg.rewards and key in cfg.rewards[term].params:
            cfg.rewards[term].params[key] = CMD_DEADBAND
    cfg.rewards["pose"].params["running_threshold"] = MAX_FWD + MAX_YAW

    # The stock command curriculum is quadruped-sized: its first stage overwrites
    # lin_vel_x with (-1.0, 1.0) at step 0 and later ramps to 3 m/s -- ~13x what
    # Rocky can do. Training against commands the robot cannot reach produces
    # flailing rather than a gait. "step" counts policy steps, so these are
    # independent of num_envs and num_steps_per_env.
    if "command_vel" in cfg.curriculum:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {"step": 0,
             "lin_vel_x": (-0.02, 0.03),
             "ang_vel_z": (-MAX_YAW * 0.5, MAX_YAW * 0.5)},
            {"step": 72_000,
             "lin_vel_x": (-MAX_FWD * 0.45, MAX_FWD * 0.7),
             "ang_vel_z": (-MAX_YAW * 0.75, MAX_YAW * 0.75)},
            {"step": 144_000,
             "lin_vel_x": (-MAX_FWD * 0.6, MAX_FWD),
             "ang_vel_z": (-MAX_YAW, MAX_YAW)},
        ]

    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact, params={"sensor_name": base_ground_cfg.name}
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain, mode="reset", params={}
        )
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def rocky_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rocky velocity tracking on flat ground. Start here."""
    cfg = rocky_rough_env_cfg(play=play)

    cfg.sim.njmax = 400
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    remove = {"terrain_scan", "base_ground_touch"}
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name not in remove)
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(50.0)}
    )
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist = cfg.commands["twist"]
        assert isinstance(twist, UniformVelocityCommandCfg)
        twist.ranges.lin_vel_x = (-MAX_FWD, MAX_FWD * 1.2)
        twist.ranges.ang_vel_z = (-MAX_YAW * 1.2, MAX_YAW * 1.2)

    return cfg
