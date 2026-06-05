#!/usr/bin/env bash
# Watchdog for an RL training run. Monitors a train.py PID, reads progress
# from tfevents (Train/mean_reward, same source the pipeline uses for peak
# detection) and writes STATUS_<state>.txt into the run_dir so the pipeline's
# watcher.wait_for_finish() can detect completion.
#
# States written:
#   STATUS_DONE.txt        train exited cleanly and reached --min-iter
#   STATUS_CRASHED.txt     train PID died before reaching --min-iter
#   STATUS_REGRESSION.txt  reward dropped > --regression-pct from peak after min-iter
#
# Args:
#   --pid <pid>             PID of the train.py process to monitor (required)
#   --run-dir <dir>         Run directory where tfevents live and STATUS_* go (required)
#   --interval <sec>        Poll interval, default 300
#   --min-iter <int>        Iteration threshold to qualify as "done", default 3000
#   --regression-pct <int>  Reward drop % from peak that triggers regression, default 25

set -u

TRAIN_PID=""
RUN_DIR=""
INTERVAL=300
MIN_ITER=3000
REGRESSION_PCT=25

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) TRAIN_PID="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --min-iter) MIN_ITER="$2"; shift 2 ;;
        --regression-pct) REGRESSION_PCT="$2"; shift 2 ;;
        *) echo "[watchdog] arg desconhecido: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$TRAIN_PID" || -z "$RUN_DIR" ]]; then
    echo "Uso: $0 --pid <pid> --run-dir <dir> [--interval N] [--min-iter N] [--regression-pct N]" >&2
    exit 2
fi

mkdir -p "$RUN_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJLAB_DIR="$(dirname "$SCRIPT_DIR")"
PY="$MJLAB_DIR/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
if [[ -z "$PY" ]]; then
    echo "[watchdog] python not found (looked at $MJLAB_DIR/.venv/bin/python)" >&2
    exit 2
fi

log() { echo "[$(date '+%H:%M:%S')] [watchdog] $*"; }

write_status() {
    local s="$1"
    echo "$(date -Iseconds): $s" > "$RUN_DIR/STATUS_${s}.txt"
    log "wrote STATUS_${s}.txt"
}

# Returns "peak_iter peak_reward last_iter last_reward" on stdout, or empty
# string if tfevents do not yet contain the reward tag.
read_summary() {
    "$PY" - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
if not any(run_dir.glob("events.out.tfevents.*")):
    sys.exit(0)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print(f"# tensorboard import failed: {e}", file=sys.stderr)
    sys.exit(0)
ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
try:
    ea.Reload()
except Exception:
    sys.exit(0)
tag = "Train/mean_reward"
if tag not in ea.Tags().get("scalars", []):
    sys.exit(0)
sc = ea.Scalars(tag)
if not sc:
    sys.exit(0)
peak = max(sc, key=lambda s: s.value)
last = sc[-1]
print(f"{peak.step} {peak.value} {last.step} {last.value}")
PY
}

log "up — pid=$TRAIN_PID run_dir=$RUN_DIR interval=${INTERVAL}s min_iter=$MIN_ITER regression=${REGRESSION_PCT}%"

while true; do
    alive=1
    kill -0 "$TRAIN_PID" 2>/dev/null || alive=0

    summary="$(read_summary 2>/dev/null || true)"
    peak_iter=0
    peak_reward=0
    last_iter=0
    last_reward=0
    if [[ -n "$summary" ]]; then
        read -r peak_iter peak_reward last_iter last_reward <<<"$summary"
    fi

    if [[ "$alive" == "0" ]]; then
        if (( last_iter >= MIN_ITER )); then
            log "train pid $TRAIN_PID exited at iter=$last_iter (>= min=$MIN_ITER)"
            write_status DONE
        else
            log "train pid $TRAIN_PID died at iter=$last_iter (< min=$MIN_ITER)"
            write_status CRASHED
        fi
        exit 0
    fi

    if (( last_iter >= MIN_ITER )) && [[ -n "$summary" ]]; then
        is_regression=$("$PY" -c "p=$peak_reward; l=$last_reward; pct=$REGRESSION_PCT; print(1 if l < p*(1 - pct/100.0) else 0)")
        if [[ "$is_regression" == "1" ]]; then
            log "regression: last=$last_reward < ${REGRESSION_PCT}% below peak=$peak_reward@iter=$peak_iter"
            write_status REGRESSION
            exit 0
        fi
    fi

    log "alive iter=$last_iter peak=${peak_reward}@${peak_iter}"
    sleep "$INTERVAL"
done
