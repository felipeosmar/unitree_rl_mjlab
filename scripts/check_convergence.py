"""Analisa convergência de uma run de treino RL via eventos do TensorBoard.

Uso:
  .venv/bin/python scripts/check_convergence.py
      # auto-detecta a run mais recente sob logs/rsl_rl/

  .venv/bin/python scripts/check_convergence.py --experiment go2_rough_smooth
      # última run de um experiment específico

  .venv/bin/python scripts/check_convergence.py logs/rsl_rl/.../<run_dir>
      # path direto

  .venv/bin/python scripts/check_convergence.py --watch
      # modo observador (re-roda a cada 30s)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ANSI colors.
G = "\033[32m"   # green
Y = "\033[33m"   # yellow
R = "\033[31m"   # red
B = "\033[1m"    # bold
D = "\033[2m"    # dim
RST = "\033[0m"


@dataclass
class MetricSpec:
  tag: str
  label: str
  good: float          # threshold pra verde
  bad: float           # threshold pra vermelho
  higher_is_better: bool
  critical: bool = True   # se False, contribui menos pro veredito final


# Targets baseados em runs convergidos (Go2 velocity, terreno Flat).
METRICS: list[MetricSpec] = [
  MetricSpec("Train/mean_reward",                    "mean_reward",            good=45.0, bad=35.0, higher_is_better=True),
  MetricSpec("Train/mean_episode_length",            "ep_length",              good=950,  bad=700,  higher_is_better=True),
  MetricSpec("Episode_Reward/track_linear_velocity", "track_lin_vel",          good=0.80, bad=0.55, higher_is_better=True),
  MetricSpec("Episode_Reward/track_angular_velocity","track_ang_vel",          good=0.85, bad=0.65, higher_is_better=True),
  MetricSpec("Metrics/twist/error_vel_xy",           "err_vel_xy (m/s)",       good=0.40, bad=1.00, higher_is_better=False),
  MetricSpec("Metrics/twist/error_vel_yaw",          "err_vel_yaw (rad/s)",    good=0.30, bad=0.60, higher_is_better=False),
  MetricSpec("Episode_Reward/foot_gait",             "foot_gait",              good=0.40, bad=0.30, higher_is_better=True, critical=False),
  MetricSpec("Episode_Reward/is_terminated",         "is_terminated",          good=-0.005, bad=-0.02, higher_is_better=True),
  MetricSpec("Episode_Termination/illegal_contact",  "illegal_contact",        good=0.05, bad=0.30, higher_is_better=False, critical=False),
  MetricSpec("Curriculum/terrain_levels",            "terrain_levels",         good=0.70, bad=0.40, higher_is_better=True, critical=False),
  MetricSpec("Policy/mean_std",                      "policy_std",             good=0.45, bad=0.70, higher_is_better=False, critical=False),
  MetricSpec("Loss/value",                           "loss_value",             good=0.05, bad=0.20, higher_is_better=False, critical=False),
]

MIN_ITER_TO_TRUST = 3000  # abaixo disso, qualquer veredito é tentativo


def find_latest_run(experiment: str | None = None, root: Path = Path("logs/rsl_rl")) -> Path | None:
  if not root.exists():
    return None
  if experiment is not None:
    base = root / experiment
    if not base.exists():
      return None
    candidates = sorted(base.glob("20*"), key=lambda p: p.stat().st_mtime, reverse=True)
  else:
    # Procura globalmente: pega a run com tfevents mais recente.
    all_runs = []
    for exp in root.iterdir():
      if not exp.is_dir():
        continue
      for run in exp.glob("20*"):
        evt = next(run.glob("events.out.tfevents.*"), None)
        if evt is not None:
          all_runs.append((evt.stat().st_mtime, run))
    all_runs.sort(reverse=True)
    candidates = [r for _, r in all_runs]
  return candidates[0] if candidates else None


def color_for(spec: MetricSpec, val: float) -> str:
  if spec.higher_is_better:
    if val >= spec.good:
      return G
    if val <= spec.bad:
      return R
    return Y
  else:
    if val <= spec.good:
      return G
    if val >= spec.bad:
      return R
    return Y


def trend_arrow(early: float, late: float, higher_is_better: bool) -> str:
  if abs(late - early) / (abs(early) + 1e-6) < 0.03:
    return f"{D}→{RST}"
  improving = (late > early) == higher_is_better
  if improving:
    return f"{G}↑{RST}" if higher_is_better else f"{G}↓{RST}"
  return f"{R}↓{RST}" if higher_is_better else f"{R}↑{RST}"


def analyze(run_dir: Path) -> int:
  evt = next(run_dir.glob("events.out.tfevents.*"), None)
  if evt is None:
    print(f"{R}ERRO:{RST} sem events.out.tfevents.* em {run_dir}")
    return 2

  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  available_tags = set(ea.Tags()["scalars"])

  reward_tag = "Train/mean_reward"
  if reward_tag not in available_tags:
    print(f"{R}ERRO:{RST} run sem '{reward_tag}' — não dá pra avaliar.")
    return 2

  reward_vals = np.array([s.value for s in ea.Scalars(reward_tag)])
  reward_steps = np.array([s.step for s in ea.Scalars(reward_tag)])
  n = len(reward_vals)
  current_iter = int(reward_steps[-1]) if n else 0

  # Cabeçalho.
  print()
  print(f"{B}Run:{RST}        {run_dir}")
  print(f"{B}Iter atual:{RST} {current_iter}")
  print(f"{B}Reward:{RST}     pico={reward_vals.max():.2f} @ iter={int(reward_steps[reward_vals.argmax()])}, "
        f"agora={reward_vals[-1]:.2f}")
  print()

  # Janelas: early (primeiros 10%), mid (50%), late (últimos 10%).
  def win(arr: np.ndarray, where: Literal["early", "mid", "late"]) -> float:
    L = len(arr)
    if where == "early":
      return float(arr[: max(1, L // 10)].mean())
    if where == "mid":
      half = L // 2
      qrt = max(1, L // 20)
      return float(arr[half - qrt: half + qrt].mean())
    return float(arr[-max(1, L // 10):].mean())

  print(f"{B}{'Métrica':<20} {'early':>10} {'mid':>10} {'agora':>10}  {'tendência':<12} {'veredito':<10}{RST}")
  print(D + "-" * 78 + RST)

  green_critical = 0
  yellow_critical = 0
  red_critical = 0
  total_critical = 0

  for spec in METRICS:
    if spec.tag not in available_tags:
      continue
    arr = np.array([s.value for s in ea.Scalars(spec.tag)])
    if len(arr) < 10:
      continue
    early = win(arr, "early")
    mid   = win(arr, "mid")
    late  = win(arr, "late")
    col = color_for(spec, late)
    arrow = trend_arrow(mid, late, spec.higher_is_better)
    verdict = {G: "OK", Y: "alerta", R: "RUIM"}[col]
    print(f"{spec.label:<20} {early:>10.3f} {mid:>10.3f} {col}{late:>10.3f}{RST}   "
          f"{arrow:<12} {col}{verdict:<10}{RST}")
    if spec.critical:
      total_critical += 1
      if col == G:
        green_critical += 1
      elif col == Y:
        yellow_critical += 1
      else:
        red_critical += 1

  # Detecção de regressão: pico > recente em > 15%.
  peak = reward_vals.max()
  recent = win(reward_vals, "late")
  regression_pct = (peak - recent) / (abs(peak) + 1e-6) * 100
  has_regression = regression_pct > 15 and current_iter > 2000

  print()
  print(f"{B}Veredito:{RST}")

  if current_iter < MIN_ITER_TO_TRUST:
    print(f"  {Y}⏳ Cedo demais{RST} — apenas {current_iter} iters. Volte após {MIN_ITER_TO_TRUST}+.")
    return 0

  if has_regression:
    print(f"  {R}❌ REGRESSÃO detectada{RST}: reward caiu {regression_pct:.1f}% do pico ({peak:.2f}).")
    print(f"     Provável: curriculum step destruiu a política. Não vai recuperar treinando mais.")
    return 1

  if red_critical >= 2:
    print(f"  {R}❌ Não convergiu{RST}: {red_critical}/{total_critical} métricas críticas em vermelho.")
    return 1

  if green_critical == total_critical:
    print(f"  {G}✅ CONVERGIU{RST}: todas as {total_critical} métricas críticas no verde. Pode parar.")
    return 0

  if green_critical >= total_critical - 1 and yellow_critical <= 1:
    print(f"  {G}✅ Quase convergido{RST}: {green_critical}/{total_critical} no verde. Pode parar ou treinar mais 1-2k iters.")
    return 0

  print(f"  {Y}⚠️  Treinando bem mas ainda não convergiu{RST}: "
        f"{green_critical} verde / {yellow_critical} amarelo / {red_critical} vermelho. Treine mais.")
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run_dir", nargs="?", default=None, help="Path direto pra run dir (override do auto-detect)")
  ap.add_argument("--experiment", "-e", default=None, help="Nome do experiment (pega último run dele)")
  ap.add_argument("--watch", action="store_true", help="Re-analisa a cada 30s")
  args = ap.parse_args()

  if args.run_dir is not None:
    run = Path(args.run_dir)
    if not run.exists():
      print(f"{R}ERRO:{RST} {run} não existe", file=sys.stderr)
      return 2
  else:
    run = find_latest_run(args.experiment)
    if run is None:
      hint = f" (em logs/rsl_rl/{args.experiment})" if args.experiment else ""
      print(f"{R}ERRO:{RST} nenhuma run encontrada{hint}", file=sys.stderr)
      return 2

  if not args.watch:
    return analyze(run)

  while True:
    print("\033[2J\033[H", end="")  # clear screen
    print(f"{D}[watch] Ctrl+C pra sair{RST}\n")
    code = analyze(run)
    print(f"\n{D}Próxima atualização em 30s...{RST}")
    try:
      time.sleep(30)
    except KeyboardInterrupt:
      print()
      return code


if __name__ == "__main__":
  sys.exit(main())
