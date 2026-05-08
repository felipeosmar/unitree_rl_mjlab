#!/usr/bin/env bash
# train_gallop.sh
#
# Treina Unitree-Go2-Gallop (bound gait, alta velocidade) por 50000 iters.
# Política separada do trote — destinada a ser carregada como segundo
# checkpoint no play.py e chaveada via gamepad.
#
# Uso:
#   bash scripts/train_gallop.sh                # 4096 envs, 50000 iters
#   NUM_ENVS=2048 bash scripts/train_gallop.sh  # override env count
#   ITERS=30000  bash scripts/train_gallop.sh   # override iterations
#
# Monitorar:
#   tail -F logs/pipeline/train_gallop_<TS>.log
#
# Só checkpoints:
#   tail -F logs/pipeline/train_gallop_<TS>.log | grep CHECKPOINT

set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-50000}"
EXPERIMENT="${EXPERIMENT:-go2_gallop}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="logs/pipeline"

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_gallop_${TS}.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo "[$(ts)] === INICIANDO TREINAMENTO GALOPE ==="
  echo "[$(ts)] Task:           Unitree-Go2-Gallop"
  echo "[$(ts)] Experiment:     $EXPERIMENT"
  echo "[$(ts)] num_envs:       $NUM_ENVS"
  echo "[$(ts)] max_iterations: $ITERS"
  echo "[$(ts)] gait:           bound (offset [0,0,0.5,0.5], period 0.35s)"
  echo "[$(ts)] vel range:      0.5-3.5 m/s (curriculum 0.5-1.5 -> 0.5-3.5)"
  echo "[$(ts)] log_file:       $LOG_FILE"
  echo ""
} >> "$LOG_FILE"

echo "Log: $LOG_FILE"
echo ""
echo "Monitorar tudo:"
echo "  tail -F $LOG_FILE"
echo ""
echo "Monitorar só checkpoints:"
echo "  tail -F $LOG_FILE | grep CHECKPOINT"
echo ""

# Watcher em background: detecta novo checkpoint.
LOG_ROOT="logs/rsl_rl/$EXPERIMENT"
(
  RUN_DIR=""
  for _ in $(seq 1 60); do
    RUN_DIR=$(ls -td "$LOG_ROOT"/20* 2>/dev/null | head -1 || true)
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
      DIR_TS=$(stat -c %Y "$RUN_DIR" 2>/dev/null || echo 0)
      SCRIPT_TS=$(date +%s)
      if (( DIR_TS >= SCRIPT_TS - 60 )); then
        break
      fi
    fi
    RUN_DIR=""
    sleep 2
  done

  if [[ -z "$RUN_DIR" ]]; then
    echo "[$(ts)] AVISO: watcher nao encontrou run_dir apos 120s" >> "$LOG_FILE"
    exit 0
  fi
  echo "[$(ts)] watcher monitorando: $RUN_DIR" >> "$LOG_FILE"

  declare -A SEEN
  while true; do
    for ckpt in "$RUN_DIR"/model_*.pt; do
      [[ -f "$ckpt" ]] || continue
      name=$(basename "$ckpt")
      if [[ -z "${SEEN[$name]:-}" ]]; then
        SEEN[$name]=1
        size=$(stat -c %s "$ckpt" 2>/dev/null | awk '{printf "%.1fMB", $1/1048576}')
        echo "[$(ts)] >>> CHECKPOINT salvo: $name ($size)" >> "$LOG_FILE"
      fi
    done
    sleep 5
  done
) &
WATCHER_PID=$!
trap 'kill $WATCHER_PID 2>/dev/null || true' EXIT INT TERM

"$PYTHON" scripts/train.py Unitree-Go2-Gallop \
  --agent.experiment-name "$EXPERIMENT" \
  --agent.max-iterations "$ITERS" \
  --env.scene.num-envs "$NUM_ENVS" \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(ts)] === TREINAMENTO FINALIZADO ===" >> "$LOG_FILE"
