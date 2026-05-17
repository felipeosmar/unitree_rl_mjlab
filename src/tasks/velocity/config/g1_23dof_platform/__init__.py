"""Register the G1-23DOF balance-on-platform task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.config.g1_23dof.rl_cfg import (
    unitree_g1_23dof_ppo_runner_cfg,
)
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import unitree_g1_23dof_platform_env_cfg


register_mjlab_task(
    task_id="Unitree-G1-23Dof-Platform",
    env_cfg=unitree_g1_23dof_platform_env_cfg(),
    play_env_cfg=unitree_g1_23dof_platform_env_cfg(play=True),
    rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
