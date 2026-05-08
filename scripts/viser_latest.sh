#!/usr/bin/env bash
# viser_latest.sh
#
# Lança o viser viewer com o último checkpoint disponível, em CPU,
# com Xbox gamepad. Garante uma única instância (mata viser anterior).
#
# Uso:
#   bash scripts/viser_latest.sh                       # Unitree-Go2-Rough, exp go2_stage2_rough
#   bash scripts/viser_latest.sh Unitree-Go2-Flat      # outra task
#   EXPERIMENT=go2_stage1_flat bash scripts/viser_latest.sh Unitree-Go2-Flat
#
# Variáveis de ambiente:
#   EXPERIMENT  Nome do experimento sob logs/rsl_rl/ (default: go2_stage2_rough)
#   GAMEPAD     Device do controle (default: /dev/input/js0)
#   PYTHON      Interpretador Python (default: .venv/bin/python)

set -euo pipefail

TASK="${1:-Unitree-Go2-Rough}"
EXPERIMENT="${EXPERIMENT:-go2_stage2_rough}"
GAMEPAD="${GAMEPAD:-/dev/input/js0}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG_ROOT="logs/rsl_rl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 1) Localiza a run mais recente do experimento
LATEST_RUN=$(ls -td "$LOG_ROOT/$EXPERIMENT"/20* 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_RUN" || ! -d "$LATEST_RUN" ]]; then
  echo "ERRO: nenhuma run encontrada em $LOG_ROOT/$EXPERIMENT" >&2
  exit 1
fi

# 2) Localiza o último checkpoint (ordenado por mtime)
LATEST_CKPT=$(ls -t "$LATEST_RUN"/model_*.pt 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_CKPT" ]]; then
  echo "ERRO: nenhum checkpoint em $LATEST_RUN" >&2
  exit 1
fi

log "Task:       $TASK"
log "Run:        $LATEST_RUN"
log "Checkpoint: $(basename "$LATEST_CKPT")"

# 3) Garante instância única — mata viser anterior se houver
EXISTING_PIDS=$(pgrep -f "scripts/play\.py.*--viewer viser" 2>/dev/null || true)
if [[ -n "$EXISTING_PIDS" ]]; then
  log "Matando instância(s) anterior(es): $EXISTING_PIDS"
  # shellcheck disable=SC2086
  kill $EXISTING_PIDS 2>/dev/null || true
  sleep 2
  for pid in $EXISTING_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
fi

# 4) Avisa se gamepad ausente (play.py segue mesmo sem ele)
if [[ ! -e "$GAMEPAD" ]]; then
  log "AVISO: $GAMEPAD não encontrado. O viser sobe, mas sem controle de velocidade."
fi

# 5) Lança play.py em CPU + viser + gamepad com 1 ambiente
# nconmax=256 evita overflow esporadico em terrain Rough com 1 env (heuristica
# default subestima quando num_envs=1).
log "Lançando viser em CPU (http://localhost:8080)"
exec "$PYTHON" scripts/play.py "$TASK" \
  --checkpoint-file "$LATEST_CKPT" \
  --device cpu \
  --viewer viser \
  --gamepad "$GAMEPAD" \
  --num-envs 1 \
  --nconmax 256
