"""Unitree G1-23DOF balance-on-platform environment configuration.

Reuses the flat-terrain velocity env_cfg as a starting point but:
  - locks the velocity command to (0, 0, 0) so the only "task" is to stand
    upright in place;
  - swaps the mild 5-6 s push-robot disturbance for an aggressive impulse
    train applied to the torso every 0.5-1.5 s, simulating the perturbations
    the robot would feel on a tilting boat-like platform (without needing to
    actually add a moving platform to the training scene);
  - drops walking-shaped rewards (foot clearance, foot slip, gait tracking)
    and bumps the orientation/joint-regularization weights so the policy is
    rewarded for stability rather than locomotion.

The resulting policy should generalise to the unitree_mujoco boat scenario:
the perturbations the boat applies to the robot's CoM look similar to the
training-time torso impulses.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.config.g1_23dof.env_cfgs import (
    unitree_g1_23dof_flat_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def unitree_g1_23dof_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the platform-balance env config (G1 23-DOF)."""
    cfg = unitree_g1_23dof_flat_env_cfg(play=play)

    # 1) Lock the velocity command to (0, 0, 0). The robot's only task is to
    #    stay upright in place.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    # Don't let curriculum widen the command range later.
    if hasattr(twist_cmd.ranges, "heading"):
        twist_cmd.ranges.heading = (0.0, 0.0)

    # 2) Heavier, more frequent perturbations — replace the gentle linear push
    #    with one that hits roll/pitch hard, so the robot has to lean into the
    #    disturbance like it would on a tilting plate.
    cfg.events["push_robot"] = EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.8, 1.8),
        params={
            "velocity_range": {
                "x": (-0.6, 0.6),
                "y": (-0.6, 0.6),
                "z": (-0.3, 0.3),
                "roll":  (-1.0, 1.0),   # ±57 °/s angular kick
                "pitch": (-1.0, 1.0),
                "yaw":   (-0.4, 0.4),
            }
        },
    )

    # 3) Reward shaping for "stand still on a moving floor":
    #    - Strengthen the upright penalty (body_orientation_l2 is negative).
    #    - Strengthen the angular-velocity penalty so the torso stays calm.
    #    - Drop foot_clearance and foot_slip (no walking here).
    #    - Keep alive_bonus + pose terms.
    cfg.rewards["body_orientation_l2"].weight *= 3.0
    cfg.rewards["body_ang_vel"].weight        *= 2.0
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = 0.0
    if "foot_slip" in cfg.rewards:
        cfg.rewards["foot_slip"].weight = 0.0
    # Tighten the pose std (penalize deviation from FixStand more sharply).
    pose = cfg.rewards.get("pose")
    if pose is not None:
        pose.params["std_standing"] = {".*": 0.03}

    # 4) For a balance task, no need for the long curriculum we use in
    #    velocity training (which gradually widens the command range).
    if not play and isinstance(cfg.curriculum, dict):
        cfg.curriculum.pop("command_range", None)

    return cfg
