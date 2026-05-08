#!/usr/bin/env bash
# compare_checkpoints.sh
#
# Roda múltiplos checkpoints sequencialmente no viser pra comparação visual.
# Mata o viser anterior antes de subir o próximo. Espera você fechar o
# browser/tab (ou Ctrl+C) entre um e outro pra avançar.
#
# Uso:
#   bash scripts/compare_checkpoints.sh
#       # Default: go2_velocity/2026-04-15_16-10-05, iters 3000 5000 8000 10000
#
#   bash scripts/compare_checkpoints.sh <run_dir>
#       # Default iters [3000 5000 8000 10000]
#
#   bash scripts/compare_checkpoints.sh <run_dir> 1000 3000 5000 7000 10000
#       # Lista customizada
#
# Variáveis:
#   TASK     Task id (default: Unitree-Go2-Flat — usa Unitree-Go2-Rough se a run for rough)
#   GAMEPAD  Device do controle (default: /dev/input/js0)
#   PYTHON   Interpretador (default: .venv/bin/python)

set -euo pipefail

DEFAULT_RUN="logs/rsl_rl/go2_velocity/2026-04-15_16-10-05"
DEFAULT_ITERS=(3000 5000 8000 10000)

RUN_DIR="${1:-$DEFAULT_RUN}"
shift || true
if [[ $# -gt 0 ]]; then
  ITERS=("$@")
else
  ITERS=("${DEFAULT_ITERS[@]}")
fi

GAMEPAD="${GAMEPAD:-/dev/input/js0}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERRO: run_dir não existe: $RUN_DIR" >&2
  exit 1
fi

# Detecta task automaticamente pelo terrain_type no env.yaml
detect_task() {
  local env_yaml="$RUN_DIR/params/env.yaml"
  if [[ -f "$env_yaml" ]]; then
    if grep -q "terrain_type: plane" "$env_yaml"; then
      echo "Unitree-Go2-Flat"
    else
      echo "Unitree-Go2-Rough"
    fi
  else
    echo "Unitree-Go2-Flat"
  fi
}
TASK="${TASK:-$(detect_task)}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Run:        $RUN_DIR"
log "Task:       $TASK (auto-detectado)"
log "Iters:      ${ITERS[*]}"
log "Gamepad:    $GAMEPAD"
echo ""
log "Cada checkpoint sobe em http://localhost:8080"
log "Feche o browser e dê Ctrl+C aqui pra avançar pro próximo."
echo ""

kill_viser() {
  local pids
  pids=$(pgrep -f "scripts/play\.py.*--viewer viser" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi
}

# Cleanup ao sair (Ctrl+C global aborta tudo)
trap 'echo ""; log "Abortado."; kill_viser; exit 130' INT TERM

for iter in "${ITERS[@]}"; do
  CKPT="$RUN_DIR/model_${iter}.pt"
  if [[ ! -f "$CKPT" ]]; then
    log "AVISO: $CKPT não existe — pulando"
    continue
  fi

  echo ""
  echo "============================================================"
  log "▶ Testando model_${iter}.pt"
  echo "============================================================"

  kill_viser

  # Trap específico desse iter: Ctrl+C avança em vez de abortar tudo.
  set +e
  trap 'echo ""; log "Avançando pro próximo..."; kill_viser; break_inner=1' INT
  break_inner=0
  "$PYTHON" scripts/play.py "$TASK" \
    --checkpoint-file "$CKPT" \
    --device cpu \
    --viewer viser \
    --gamepad "$GAMEPAD" \
    --num-envs 1 \
    --nconmax 256 \
    || true
  trap 'echo ""; log "Abortado."; kill_viser; exit 130' INT TERM
  set -e

  if [[ $break_inner -eq 0 ]]; then
    # play.py terminou sozinho (ex: erro). Pergunta se continua.
    read -r -p "Continuar pro próximo iter? [S/n] " ans
    if [[ "$ans" == "n" || "$ans" == "N" ]]; then
      break
    fi
  fi
done

kill_viser
echo ""
log "Comparação finalizada."
