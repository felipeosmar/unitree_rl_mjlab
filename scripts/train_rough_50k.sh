#!/usr/bin/env bash
# train_rough_50k.sh
#
# Treina Unitree-Go2-Rough com 4096 envs por 50000 iterações.
# Loga em arquivo timestamped, com marcação especial em cada novo checkpoint.
#
# Uso:
#   bash scripts/train_rough_50k.sh                # 4096 envs, 50000 iters
#   NUM_ENVS=2048 bash scripts/train_rough_50k.sh  # override env count
#   ITERS=10000  bash scripts/train_rough_50k.sh   # override iterations
#
# Monitorar com:
#   tail -F logs/pipeline/train_rough_50k_<TS>.log
#
# Para ver SÓ as marcações de checkpoint:
#   tail -F logs/pipeline/train_rough_50k_<TS>.log | grep CHECKPOINT

set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-50000}"
EXPERIMENT="${EXPERIMENT:-go2_stage2_rough}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="logs/pipeline"

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_rough_50k_${TS}.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

{
  echo "[$(ts)] === INICIANDO TREINAMENTO ==="
  echo "[$(ts)] Task:           Unitree-Go2-Rough"
  echo "[$(ts)] Experiment:     $EXPERIMENT"
  echo "[$(ts)] num_envs:       $NUM_ENVS"
  echo "[$(ts)] max_iterations: $ITERS"
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

# Watcher em background: detecta novo checkpoint e escreve linha com timestamp.
LOG_ROOT="logs/rsl_rl/$EXPERIMENT"
(
  RUN_DIR=""
  for _ in $(seq 1 60); do
    RUN_DIR=$(ls -td "$LOG_ROOT"/20* 2>/dev/null | head -1 || true)
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
      # Usa apenas dirs criados depois deste script (evita pegar runs antigas).
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

# Roda treino. stdout completo vai pro log via tee.
"$PYTHON" scripts/train.py Unitree-Go2-Rough \
  --agent.experiment-name "$EXPERIMENT" \
  --agent.max-iterations "$ITERS" \
  --env.scene.num-envs "$NUM_ENVS" \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(ts)] === TREINAMENTO FINALIZADO ===" >> "$LOG_FILE"
