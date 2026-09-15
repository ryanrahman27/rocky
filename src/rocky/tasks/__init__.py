"""Rocky tasks for mjlab.

Loaded through the `mjlab.tasks` entry point declared in pyproject.toml, which
registers:

    Mjlab-Velocity-Flat-Rocky     flat ground, start here
    Mjlab-Velocity-Rough-Rocky    generated rough terrain, with a height scan
  Mjlab-Velocity-Rough-Blind-Rocky
                                rough terrain, proprioception only -- same
                                observation layout as the flat task, so a
                                flat-trained checkpoint runs unchanged
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    rocky_flat_env_cfg,
    rocky_rough_blind_env_cfg,
    rocky_rough_env_cfg,
)
from .rl_cfg import rocky_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Rocky",
    env_cfg=rocky_flat_env_cfg(),
    play_env_cfg=rocky_flat_env_cfg(play=True),
    rl_cfg=rocky_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Rocky",
    env_cfg=rocky_rough_env_cfg(),
    play_env_cfg=rocky_rough_env_cfg(play=True),
    rl_cfg=rocky_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Blind-Rocky",
    env_cfg=rocky_rough_blind_env_cfg(),
    play_env_cfg=rocky_rough_blind_env_cfg(play=True),
    rl_cfg=rocky_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
