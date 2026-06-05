"""Unitree Go2 velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_obstacles_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree Go2 flat observations + obstacle terrain for testing."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Remove terrain_scan sensor and height_scan (not in flat-trained policy).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  if "height_scan" in cfg.observations["actor"].terms:
    del cfg.observations["actor"].terms["height_scan"]
  if "height_scan" in cfg.observations["critic"].terms:
    del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0
      # Mix of flat, stairs, boxes, and obstacles
      cfg.scene.terrain.terrain_generator.sub_terrains = {
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
          proportion=0.15, step_height_range=(0.0, 0.15), step_width=0.3,
          platform_width=3.0, border_width=1.0,
        ),
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
          proportion=0.15, step_height_range=(0.0, 0.15), step_width=0.3,
          platform_width=3.0, border_width=1.0,
        ),
        "open_stairs": terrain_gen.BoxOpenStairsTerrainCfg(
          proportion=0.1, step_height_range=(0.1, 0.2), step_width_range=(0.4, 0.8),
          platform_width=1.0, border_width=0.25,
        ),
        "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
          proportion=0.1, step_width=0.8, step_height_range=(0.1, 0.3),
          platform_width=1.0, border_width=0.25,
        ),
        "random_spread_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
          proportion=0.15, num_boxes=60, box_width_range=(0.2, 0.8),
          box_length_range=(0.2, 0.8), box_height_range=(0.05, 0.4),
          platform_width=1.0, border_width=0.25,
        ),
        "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
          proportion=0.1, stone_size_range=(0.4, 0.8), stone_distance_range=(0.2, 0.5),
          stone_height=0.2, stone_height_variation=0.1, stone_size_variation=0.2,
          displacement_range=0.1, floor_depth=2.0, platform_width=1.0, border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
          proportion=0.05, obstacle_width_range=(0.3, 1.0), obstacle_height_range=(0.05, 0.3),
          num_obstacles=40, border_width=0.25,
        ),
      }

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_go2_gallop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree Go2 gallop (bound) high-speed configuration.

  Bound gait: front pair and rear pair move in phase. Used as a fast,
  energy-efficient mode for sustained high speeds (~2-3.5 m/s). Trained
  as a separate policy from the trot policy.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)

  # Faster cadence for bound (~2.85 Hz). Trot uses 0.6s (~1.67 Hz).
  gallop_period = 0.35
  # Bound: FR+FL together, RR+RL together. Order is (FR, FL, RR, RL).
  gallop_offset = [0.0, 0.0, 0.5, 0.5]

  cfg.observations["actor"].terms["phase"].params["period"] = gallop_period
  cfg.observations["critic"].terms["phase"].params["period"] = gallop_period

  cfg.rewards["foot_gait"].params["period"] = gallop_period
  cfg.rewards["foot_gait"].params["offset"] = gallop_offset
  # Stance fraction: bound has shorter stance / longer aerial phase.
  cfg.rewards["foot_gait"].params["threshold"] = 0.45
  # Reinforce bound shape more than trot did.
  cfg.rewards["foot_gait"].weight = 0.8

  # Higher swing height — bound has clear aerial phase.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.14

  # Earlier running posture: gallop is always in the running regime.
  cfg.rewards["pose"].params["walking_threshold"] = 0.1
  cfg.rewards["pose"].params["running_threshold"] = 0.8

  # Re-tune posture stds — wider hip/thigh range for bound dynamics.
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.20,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.45,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.55,
  }

  # Command range targeted at high forward speed; reduce lateral/yaw.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.5, 3.5)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

  # Curriculum: ramp forward speed gradually so policy converges.
  # (play=True clears the curriculum upstream — only set in train mode.)
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0,          "lin_vel_x": (0.5, 1.5), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.5, 0.5)},
      {"step": 5000 * 24,  "lin_vel_x": (0.5, 2.2), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.6, 0.6)},
      {"step": 15000 * 24, "lin_vel_x": (0.5, 2.8), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.7, 0.7)},
      {"step": 30000 * 24, "lin_vel_x": (0.5, 3.5), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.8, 0.8)},
    ]

  if play:
    twist_cmd.ranges.lin_vel_x = (0.5, 3.5)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

  return cfg
