#!/usr/bin/env bash
# train_and_simulate.sh
#
# Pipeline: treina uma policy e simula automaticamente cada checkpoint novo.
# Suporta dois modos de simulacao:
#   video : grava mp4 headless (default) — ideal para revisao posterior
#   viser : abre viewer web para inspecao interativa de cada checkpoint
#
# Uso:
#   bash scripts/train_and_simulate.sh <TASK> [OPTIONS]
#
# Exemplos:
#   bash scripts/train_and_simulate.sh Unitree-Go2-Flat
#   bash scripts/train_and_simulate.sh Unitree-Go2-Flat --sim-mode viser
#   bash scripts/train_and_simulate.sh Unitree-Go2-Rough --num-envs 2048 --sim-envs 4
#   bash scripts/train_and_simulate.sh Unitree-Go2-Flat --train-args "--agent.max-iterations 5000"
#
# Videos ficam em: logs/rsl_rl/<experiment>/<run>/videos/sim/
# Viser fica em:   http://localhost:<viser-port>

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASK=""
NUM_ENVS=4096
SIM_ENVS=4            # Ambientes na simulacao (poucos = rapido)
SIM_LENGTH=300         # Frames por video (modo video)
SIM_MODE="video"       # "video" ou "viser"
VISER_PORT=8080        # Porta do viser (modo viser)
POLL_INTERVAL=30       # Segundos entre checagens de novos checkpoints
TRAIN_ARGS=""          # Args extras passados direto ao train.py
PYTHON="${PYTHON:-.venv/bin/python}"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Uso: $0 <TASK> [OPTIONS]

  TASK            Task ID (ex: Unitree-Go2-Flat, Unitree-Go2-Rough)

Opcoes:
  --num-envs N    Ambientes de treinamento (default: $NUM_ENVS)
  --sim-envs N    Ambientes na simulacao (default: $SIM_ENVS)
  --sim-length N  Frames por video, modo video (default: $SIM_LENGTH)
  --sim-mode M    Modo de simulacao: "video" ou "viser" (default: $SIM_MODE)
  --viser-port P  Porta do viewer viser (default: $VISER_PORT)
  --poll N        Intervalo de polling em segundos (default: $POLL_INTERVAL)
  --train-args S  Args extras para train.py (entre aspas)
  -h, --help      Mostra esta ajuda
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage
TASK="$1"; shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-envs)    NUM_ENVS="$2"; shift 2 ;;
    --sim-envs)    SIM_ENVS="$2"; shift 2 ;;
    --sim-length)  SIM_LENGTH="$2"; shift 2 ;;
    --sim-mode)    SIM_MODE="$2"; shift 2 ;;
    --viser-port)  VISER_PORT="$2"; shift 2 ;;
    --poll)        POLL_INTERVAL="$2"; shift 2 ;;
    --train-args)  TRAIN_ARGS="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *)             echo "Arg desconhecido: $1"; usage ;;
  esac
done

if [[ "$SIM_MODE" != "video" && "$SIM_MODE" != "viser" ]]; then
  echo "ERRO: --sim-mode deve ser 'video' ou 'viser'"
  exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

SIM_PID=""

cleanup() {
  log "Encerrando..."
  # Para simulacao viser em andamento
  if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
  # Para treinamento
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "Parando treinamento (PID $TRAIN_PID)"
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  log "Pipeline encerrado."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 1. Lanca treinamento em background
# ---------------------------------------------------------------------------
log "=== Iniciando treinamento: $TASK ==="
log "    Envs: $NUM_ENVS | Sim mode: $SIM_MODE | Args: ${TRAIN_ARGS:-nenhum}"

# shellcheck disable=SC2086
$PYTHON scripts/train.py "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  $TRAIN_ARGS \
  &
TRAIN_PID=$!
log "Treinamento lancado (PID: $TRAIN_PID)"

# Espera o diretorio de log aparecer
sleep 5
LOG_ROOT="logs/rsl_rl"

# Descobre o diretorio da run mais recente
find_run_dir() {
  local latest
  latest=$(find "$LOG_ROOT" -maxdepth 2 -mindepth 2 -type d -newer "$0" 2>/dev/null | sort | tail -1)
  echo "$latest"
}

RUN_DIR=""
for attempt in $(seq 1 30); do
  RUN_DIR=$(find_run_dir)
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    break
  fi
  log "Aguardando diretorio de log... (tentativa $attempt/30)"
  sleep 5
done

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  log "ERRO: Nao encontrou diretorio de log apos 150s. Abortando."
  exit 1
fi

log "Diretorio da run: $RUN_DIR"

# ---------------------------------------------------------------------------
# 2. Monitora checkpoints e simula
# ---------------------------------------------------------------------------
SIMULATED_FILE="$RUN_DIR/.simulated_checkpoints"
touch "$SIMULATED_FILE"

simulate_checkpoint_video() {
  local ckpt_path="$1"
  local ckpt_name
  ckpt_name=$(basename "$ckpt_path" .pt)

  log ">>> [video] Simulando: $ckpt_name"

  $PYTHON scripts/simulate_checkpoint.py "$TASK" \
    --checkpoint-file "$ckpt_path" \
    --mode video \
    --num-envs "$SIM_ENVS" \
    --video-length "$SIM_LENGTH" \
    2>&1 | while IFS= read -r line; do
      echo "  [sim/$ckpt_name] $line"
    done

  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    log "<<< [video] Concluido: $ckpt_name"
  else
    log "!!! [video] Falhou: $ckpt_name"
  fi

  echo "$ckpt_path" >> "$SIMULATED_FILE"
}

simulate_checkpoint_viser() {
  local ckpt_path="$1"
  local ckpt_name
  ckpt_name=$(basename "$ckpt_path" .pt)

  # Para viser anterior se estiver rodando
  if [[ -n "$SIM_PID" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    log "Parando viser anterior..."
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi

  log ">>> [viser] Carregando: $ckpt_name -> http://localhost:$VISER_PORT"

  $PYTHON scripts/simulate_checkpoint.py "$TASK" \
    --checkpoint-file "$ckpt_path" \
    --mode viser \
    --num-envs "$SIM_ENVS" \
    --viser-port "$VISER_PORT" \
    &
  SIM_PID=$!

  echo "$ckpt_path" >> "$SIMULATED_FILE"
}

log "=== Monitorando checkpoints (poll: ${POLL_INTERVAL}s, modo: $SIM_MODE) ==="
if [[ "$SIM_MODE" == "video" ]]; then
  log "    Videos em: $RUN_DIR/videos/sim/"
else
  log "    Viser em:  http://localhost:$VISER_PORT"
fi

SIM_PIDS=()

while true; do
  # Verifica se treinamento ainda esta rodando
  TRAIN_ALIVE=true
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    TRAIN_ALIVE=false
    log "Treinamento finalizou. Processando checkpoints restantes..."
  fi

  # Lista checkpoints disponiveis (ordenados por iteracao)
  mapfile -t ALL_CKPTS < <(
    find "$RUN_DIR" -maxdepth 1 -name "model_*.pt" 2>/dev/null | sort -t_ -k2 -n
  )

  # Processa checkpoints novos
  for ckpt in "${ALL_CKPTS[@]}"; do
    if grep -qxF "$ckpt" "$SIMULATED_FILE" 2>/dev/null; then
      continue
    fi

    # Espera o arquivo estar completo
    sleep 2

    if [[ "$SIM_MODE" == "viser" ]]; then
      simulate_checkpoint_viser "$ckpt"
    else
      # Modo video: limita a 1 simulacao por vez (GPU)
      NEW_PIDS=()
      for pid in "${SIM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          NEW_PIDS+=("$pid")
        fi
      done
      SIM_PIDS=("${NEW_PIDS[@]}")

      if [[ ${#SIM_PIDS[@]} -ge 1 ]]; then
        log "Aguardando simulacao anterior..."
        wait "${SIM_PIDS[0]}" 2>/dev/null || true
        SIM_PIDS=("${SIM_PIDS[@]:1}")
      fi

      simulate_checkpoint_video "$ckpt" &
      SIM_PIDS+=($!)
    fi
  done

  # Se treinamento acabou e tudo foi processado, sai
  if [[ "$TRAIN_ALIVE" == "false" ]]; then
    if [[ "$SIM_MODE" == "video" ]]; then
      for pid in "${SIM_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
      done
    fi
    break
  fi

  sleep "$POLL_INTERVAL"
done

# ---------------------------------------------------------------------------
# 3. Resumo final
# ---------------------------------------------------------------------------
TOTAL_CKPTS=$(wc -l < "$SIMULATED_FILE" 2>/dev/null || echo 0)

log ""
log "========================================="
log "  PIPELINE CONCLUIDO"
log "========================================="
log "  Task:        $TASK"
log "  Run:         $RUN_DIR"
log "  Checkpoints: $TOTAL_CKPTS simulados"
if [[ "$SIM_MODE" == "video" ]]; then
  TOTAL_VIDEOS=$(find "$RUN_DIR/videos/sim" -name "*.mp4" 2>/dev/null | wc -l)
  log "  Videos:      $TOTAL_VIDEOS gerados em $RUN_DIR/videos/sim/"
fi
log "  Policy ONNX: $RUN_DIR/policy.onnx"
log ""
log "  Re-simular um checkpoint:"
log "    $PYTHON scripts/simulate_checkpoint.py $TASK --checkpoint-file $RUN_DIR/model_XXXX.pt --mode viser"
log "========================================="
