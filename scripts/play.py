"""Script to play RL agent with RSL-RL."""

# Patch warp's API surface (mjlab 1.2.0 expects wp.context.runtime.driver_version
# which exists in warp ≥1.14; we have 1.13). Must run before any mjlab import.
import src.warp_compat  # noqa: F401

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  gamepad: str | None = "/dev/input/js0"
  """Path to gamepad device for Xbox controller input (None to disable)."""
  nconmax: int | None = None
  """Override sim.nconmax (max contacts per world). Useful with num_envs=1 in
  Rough terrain where the heuristic underestimates contacts and triggers
  `nconmax overflow`. Try 256 if you hit that error."""
  gallop_checkpoint: str | None = None
  """Optional second policy file (bound/gallop). When set, the Xbox Y button toggles
  between the primary policy (trot) and this one."""
  gallop_task_id: str = "Unitree-Go2-Gallop"
  """Task id used to load the agent cfg for the gallop policy. Must match the task
  the gallop checkpoint was trained with."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


# Gait params per policy (must match the task configs the policies were trained on).
_TROT_PERIOD: float = 0.6
_TROT_MAX_VX: float = 2.0
_GALLOP_PERIOD: float = 0.35
_GALLOP_MAX_VX: float = 3.5
_GAMEPAD_BTN_TOGGLE: int = 3  # Xbox 'Y' button.


class PolicySwitcher:
  """Holds two inference policies and switches the env's phase observation
  period to match the active policy. Used to run a single env with two
  separately-trained policies (trot + bound)."""

  def __init__(self, env, policies, names, max_speeds, periods):
    self._env = env
    self._policies = policies
    self._names = names
    self._max_speeds = max_speeds
    self._periods = periods
    self._idx = 0
    self._apply()

  def _apply(self) -> None:
    om = self._env.unwrapped.observation_manager
    period = self._periods[self._idx]
    for grp in ("actor", "critic"):
      try:
        term_cfg = om.get_term_cfg(grp, "phase")
      except ValueError:
        continue
      term_cfg.params["period"] = period
    print(
      f"[Modo] {self._names[self._idx]} "
      f"(period={period:.2f}s, max_vx={self._max_speeds[self._idx]:.1f} m/s)"
    )

  def toggle(self) -> None:
    self._idx = (self._idx + 1) % len(self._policies)
    self._apply()

  @property
  def max_speed(self) -> float:
    return self._max_speeds[self._idx]

  def __call__(self, obs):
    return self._policies[self._idx](obs)


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  if cfg.device is not None:
    device = cfg.device
  else:
    if not torch.cuda.is_available():
      raise RuntimeError(
        "CUDA GPU not available. Pass --device cpu explicitly to run on CPU."
      )
    device = "cuda:0"
  print(f"[INFO] Device: {device}")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.nconmax is not None:
    env_cfg.sim.nconmax = cfg.nconmax
    print(f"[INFO] sim.nconmax override: {cfg.nconmax}")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Optionally load a second (gallop/bound) policy and wrap as a switcher.
  switcher: PolicySwitcher | None = None
  if TRAINED_MODE and cfg.gallop_checkpoint is not None:
    gallop_path = Path(cfg.gallop_checkpoint)
    if not gallop_path.exists():
      raise FileNotFoundError(f"Gallop checkpoint not found: {gallop_path}")
    print(f"[INFO]: Loading gallop checkpoint: {gallop_path.name}")
    gallop_agent_cfg = load_rl_cfg(cfg.gallop_task_id)
    gallop_runner_cls = load_runner_cls(cfg.gallop_task_id) or MjlabOnPolicyRunner
    gallop_runner = gallop_runner_cls(env, asdict(gallop_agent_cfg), device=device)
    gallop_runner.load(
      str(gallop_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    gallop_policy = gallop_runner.get_inference_policy(device=device)
    switcher = PolicySwitcher(
      env=env,
      policies=[policy, gallop_policy],
      names=["trote", "galope"],
      max_speeds=[_TROT_MAX_VX, _GALLOP_MAX_VX],
      periods=[_TROT_PERIOD, _GALLOP_PERIOD],
    )
    policy = switcher
    print("[INFO]: Pressione o botão Y do Xbox para alternar entre trote e galope.")

  # Enable hardware gamepad if requested.
  if cfg.gamepad is not None:
    from src.tasks.velocity.mdp.gamepad import Gamepad

    cmd_mgr = env.unwrapped.command_manager
    for name in cmd_mgr.active_terms:
      term = cmd_mgr.get_term(name)
      if hasattr(term, 'vel_command_b'):
        gp = Gamepad(device=cfg.gamepad)
        if gp.start():
          _original_compute = term.compute
          _ranges = term.cfg.ranges
          _btn_prev = [False]

          def _patched_compute(
            dt,
            _orig=_original_compute,
            _gp=gp,
            _r=_ranges,
            _t=term,
            _sw=switcher,
            _prev=_btn_prev,
          ):
            _orig(dt)
            if not _gp.connected:
              return
            if _sw is not None:
              pressed = _gp.button(_GAMEPAD_BTN_TOGGLE)
              if pressed and not _prev[0]:
                _sw.toggle()
              _prev[0] = pressed
              max_x = _sw.max_speed
            else:
              max_x = _r.lin_vel_x[1]
            _t.vel_command_b[0, 0] = _gp.left_y * max_x
            _t.vel_command_b[0, 1] = -_gp.left_x * _r.lin_vel_y[1]
            _t.vel_command_b[0, 2] = -_gp.right_x * _r.ang_vel_z[1]

          term.compute = _patched_compute
          print(f"[INFO]: Xbox gamepad enabled — left stick: move, right stick: turn")
        else:
          print(f"[WARN]: Gamepad device not found: {cfg.gamepad}")
        break

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
