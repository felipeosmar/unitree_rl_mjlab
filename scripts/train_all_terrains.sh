#!/usr/bin/env bash
# train_all_terrains.sh
#
# Treinamento do Go2 em terreno rough com curriculum.
#
# Uso:
#   bash scripts/train_all_terrains.sh [NUM_ENVS]
#
# Exemplos:
#   bash scripts/train_all_terrains.sh        # 4096 envs
#   bash scripts/train_all_terrains.sh 2048   # 2048 envs

set -euo pipefail

# ---------------------------------------------------------------------------
# Parâmetros configuráveis
# ---------------------------------------------------------------------------
NUM_ENVS="${1:-4096}"
STAGE2_EXPERIMENT="go2_stage2_rough"
STAGE2_ITERS=10000

LOG_ROOT="logs/rsl_rl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Estágio 2: Rough terrain com curriculum
# ---------------------------------------------------------------------------
log "=== Rough terrain com curriculum ($STAGE2_ITERS iterações) ==="

.venv/bin/python scripts/train.py Unitree-Go2-Rough --agent.experiment-name "$STAGE2_EXPERIMENT" --agent.max-iterations "$STAGE2_ITERS" --env.scene.num-envs "$NUM_ENVS"

# Localiza a run do estágio 2
STAGE2_RUN=$(ls -td "$LOG_ROOT/$STAGE2_EXPERIMENT"/20* 2>/dev/null | head -1)
STAGE2_RUN_NAME=$(basename "$STAGE2_RUN")

log "=== TREINAMENTO CONCLUÍDO ==="
log "Rough: $LOG_ROOT/$STAGE2_EXPERIMENT/$STAGE2_RUN_NAME"
log ""
log "Policy ONNX final: $LOG_ROOT/$STAGE2_EXPERIMENT/$STAGE2_RUN_NAME/policy.onnx"
log ""
log "Para transferir para outro computador, execute:"
log "  bash scripts/fetch_policy.sh ws1 $STAGE2_EXPERIMENT/$STAGE2_RUN_NAME"
