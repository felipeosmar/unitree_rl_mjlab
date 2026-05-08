"""Simulate a checkpoint and optionally record video or open viser viewer.

Modes:
  --mode video  : Headless batch — records mp4 and exits (default)
  --mode viser  : Opens viser web viewer for interactive inspection

Usage:
  # Batch video recording:
  python scripts/simulate_checkpoint.py Unitree-Go2-Flat \
    --checkpoint-file logs/rsl_rl/go2_velocity/.../model_1000.pt

  # Interactive viser viewer:
  python scripts/simulate_checkpoint.py Unitree-Go2-Flat \
    --checkpoint-file logs/rsl_rl/go2_velocity/.../model_1000.pt \
    --mode viser --viser-port 8080
"""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
import viser

import mjlab
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViserPlayViewer


@dataclass(frozen=True)
class SimConfig:
    checkpoint_file: str
    """Path to the .pt checkpoint file."""
    mode: Literal["video", "viser"] = "video"
    """Simulation mode: 'video' for batch recording, 'viser' for web viewer."""
    num_envs: int = 4
    """Number of parallel environments."""
    video_length: int = 300
    """Steps to record (video mode only)."""
    video_dir: str | None = None
    """Output directory for videos. Defaults to <checkpoint_dir>/videos/sim/"""
    viser_port: int = 8080
    """Port for viser web viewer (viser mode only)."""
    device: str | None = None


def main():
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    cfg = tyro.cli(
        SimConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    # Force EGL for video mode (headless).
    if cfg.mode == "video":
        os.environ.setdefault("MUJOCO_GL", "egl")

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load configs.
    env_cfg = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)
    env_cfg.scene.num_envs = cfg.num_envs

    # Resolve checkpoint.
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
        print(f"[ERROR] Checkpoint not found: {resume_path}")
        sys.exit(1)

    ckpt_name = resume_path.stem  # e.g. "model_1000"
    print(f"[INFO] Task: {chosen_task}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Mode: {cfg.mode} | Envs: {cfg.num_envs} | Device: {device}")

    # Create environment.
    render_mode = "rgb_array" if cfg.mode == "video" else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if cfg.mode == "video":
        video_dir = Path(cfg.video_dir) if cfg.video_dir else resume_path.parent / "videos" / "sim"
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Video dir: {video_dir}")
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
            name_prefix=ckpt_name,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy.
    runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    if cfg.mode == "viser":
        # Interactive viser viewer.
        server = viser.ViserServer(port=cfg.viser_port)
        print(f"[INFO] Viser viewer at http://localhost:{cfg.viser_port}")
        ViserPlayViewer(env, policy, viser_server=server).run()
    else:
        # Batch video recording.
        obs, _ = env.get_observations()
        for _ in range(cfg.video_length + 10):
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
        print(f"[INFO] Video saved to: {video_dir}")

    env.close()


if __name__ == "__main__":
    main()
