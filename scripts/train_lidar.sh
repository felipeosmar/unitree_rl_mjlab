#!/usr/bin/env bash
# train_lidar.sh
#
# Treina o Go2 utilizando o LiDAR L1 para percepção de ambiente.

set -euo pipefail

# Usa todas as threads CPU para operações PyTorch (OpenMP, MKL, etc.)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-32}"

# Configuração otimizada para RTX PRO 6000 Blackwell (98 GB VRAM).
NUM_ENVS="${NUM_ENVS:-16384}"
ITERS="${ITERS:-30000}"
EXPERIMENT="${EXPERIMENT:-go2_lidar_flat}"
STEPS="${STEPS:-48}"
MINI_BATCHES="${MINI_BATCHES:-16}"
EPOCHS="${EPOCHS:-8}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="logs/pipeline"

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_lidar_${TS}.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo "[$(ts)] === INICIANDO TREINAMENTO COM LIDAR (L1) ==="
  echo "[$(ts)] Task:              Unitree-Go2-Flat"
  echo "[$(ts)] Experiment:        $EXPERIMENT"
  echo "[$(ts)] num_envs:          $NUM_ENVS"
  echo "[$(ts)] num_steps_per_env: $STEPS"
  echo "[$(ts)] num_mini_batches:  $MINI_BATCHES"
  echo "[$(ts)] num_learning_epochs: $EPOCHS"
  echo "[$(ts)] max_iterations:    $ITERS"
  echo ""
} >> "$LOG_FILE"

echo "Log de treinamento em: $LOG_FILE"

# Watcher para monitorar checkpoints
(
  LOG_ROOT="logs/rsl_rl/$EXPERIMENT"
  sleep 10
  RUN_DIR=$(ls -td "$LOG_ROOT"/20* 2>/dev/null | head -1 || true)
  if [[ -n "$RUN_DIR" ]]; then
    declare -A SEEN
    while true; do
      for ckpt in "$RUN_DIR"/model_*.pt; do
        [[ -f "$ckpt" ]] || continue
        name=$(basename "$ckpt")
        if [[ -z "${SEEN[$name]:-}" ]]; then
          SEEN[$name]=1
          echo "[$(ts)] >>> CHECKPOINT salvo: $name" >> "$LOG_FILE"
        fi
      done
      sleep 10
    done
  fi
) &
WATCHER_PID=$!
trap 'kill $WATCHER_PID 2>/dev/null || true' EXIT INT TERM

"$PYTHON" scripts/train.py Unitree-Go2-Flat \
  --agent.experiment-name "$EXPERIMENT" \
  --agent.max-iterations "$ITERS" \
  --agent.num-steps-per-env "$STEPS" \
  --agent.algorithm.num-mini-batches "$MINI_BATCHES" \
  --agent.algorithm.num-learning-epochs "$EPOCHS" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.wandb-project "" \
  2>&1 | tee -a "$LOG_FILE"