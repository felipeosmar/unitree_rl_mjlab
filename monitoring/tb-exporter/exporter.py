"""TensorBoard event log -> Prometheus exporter (incremental).

Watches LOGS_ROOT (/logs/rsl_rl by default) for run directories of the form
<experiment>/<run-timestamp>/events.out.tfevents.* and parses TFRecord events
incrementally: each file's byte offset is tracked across scans, so only the
records appended since the last scan are decoded. Offsets are persisted to
STATE_FILE so restarts don't re-parse history.

TFRecord layout: uint64 length | uint32 len_crc | payload | uint32 data_crc.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from prometheus_client import Counter, Gauge, start_http_server
from tensorboard.compat.proto.event_pb2 import Event

LOGS_ROOT = Path(os.environ.get("LOGS_ROOT", "/logs/rsl_rl"))
STATE_FILE = Path(os.environ.get("STATE_FILE", "/state/offsets.json"))
SCRAPE_PORT = int(os.environ.get("SCRAPE_PORT", "9105"))
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "20"))

_INVALID = re.compile(r"[^a-zA-Z0-9_]")
_HEADER = struct.Struct("<Q")
_HEADER_SIZE = 12
_FOOTER_SIZE = 4

gauges: dict[str, Gauge] = {}

records_processed = Counter(
    "tb_exporter_records_processed_total",
    "Total TFRecord events successfully decoded",
    labelnames=("experiment", "run"),
)
records_corrupt = Counter(
    "tb_exporter_records_corrupt_total",
    "TFRecord events that failed to decode and were skipped",
    labelnames=("experiment", "run"),
)
files_tracked = Gauge(
    "tb_exporter_files_tracked",
    "Number of tfevents files currently tracked",
)


@dataclass
class FileState:
    offset: int = 0
    inode: int = 0
    size: int = 0


file_states: dict[str, FileState] = {}


def metric_name(tag: str) -> str:
    name = tag.replace("/", "_").replace("-", "_").replace(" ", "_")
    name = _INVALID.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return f"rsl_rl_{name}" if name else "rsl_rl_unknown"


def get_or_create(name: str, tag: str) -> Gauge:
    g = gauges.get(name)
    if g is None:
        g = Gauge(name, f"rsl_rl scalar: {tag}", labelnames=("experiment", "run"))
        gauges[name] = g
    return g


def extract_scalar(value) -> float | None:
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        t = value.tensor
        if t.float_val:
            return float(t.float_val[0])
        if t.double_val:
            return float(t.double_val[0])
        if t.tensor_content:
            try:
                arr = np.frombuffer(t.tensor_content, dtype=np.float32)
                if arr.size:
                    return float(arr[0])
            except ValueError:
                return None
    return None


def iter_new_events(path: Path, state: FileState) -> Iterator[Event | None]:
    """Yield Event protos appended since state.offset, advancing state.

    Yields None for records whose payload failed to decode (counted as corrupt).
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return
    # File rotated (new inode) or truncated -> reset.
    if state.inode and (st.st_ino != state.inode or st.st_size < state.offset):
        state.offset = 0
    state.inode = st.st_ino
    state.size = st.st_size
    if state.offset >= st.st_size:
        return

    with path.open("rb") as f:
        f.seek(state.offset)
        while True:
            hdr = f.read(_HEADER_SIZE)
            if len(hdr) < _HEADER_SIZE:
                break  # partial header — file is mid-write
            (length,) = _HEADER.unpack_from(hdr, 0)
            payload = f.read(length)
            if len(payload) < length:
                break  # partial payload
            footer = f.read(_FOOTER_SIZE)
            if len(footer) < _FOOTER_SIZE:
                break
            state.offset = f.tell()
            try:
                yield Event.FromString(payload)
            except Exception:
                yield None  # signal corrupt record


def load_state() -> None:
    if not STATE_FILE.exists():
        return
    try:
        raw = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[tb-exporter] could not load state: {exc}")
        return
    for key, data in raw.items():
        try:
            file_states[key] = FileState(**data)
        except TypeError:
            continue
    print(f"[tb-exporter] loaded {len(file_states)} file offsets from {STATE_FILE}")


def save_state() -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in file_states.items()}))
        tmp.replace(STATE_FILE)
    except OSError as exc:
        print(f"[tb-exporter] could not save state: {exc}")


def scan_once() -> None:
    if not LOGS_ROOT.exists():
        print(f"[tb-exporter] LOGS_ROOT {LOGS_ROOT} missing")
        return

    step_gauge = get_or_create("rsl_rl_step", "global step (latest)")
    total_new = 0

    for exp_dir in sorted(LOGS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        experiment = exp_dir.name
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = run_dir.name
            for ev_file in sorted(run_dir.glob("events.out.tfevents.*")):
                key = f"{experiment}|{run}|{ev_file.name}"
                state = file_states.setdefault(key, FileState())
                latest_step: int | None = None
                new_records = 0
                for event in iter_new_events(ev_file, state):
                    if event is None:
                        records_corrupt.labels(experiment=experiment, run=run).inc()
                        continue
                    if not event.HasField("summary"):
                        continue
                    step = int(event.step)
                    for value in event.summary.value:
                        scalar = extract_scalar(value)
                        if scalar is None:
                            continue
                        gauge = get_or_create(metric_name(value.tag), value.tag)
                        gauge.labels(experiment=experiment, run=run).set(scalar)
                        new_records += 1
                    if latest_step is None or step > latest_step:
                        latest_step = step
                if latest_step is not None:
                    step_gauge.labels(experiment=experiment, run=run).set(latest_step)
                if new_records:
                    records_processed.labels(experiment=experiment, run=run).inc(new_records)
                    total_new += new_records
                    print(
                        f"[tb-exporter] {experiment}/{run}/{ev_file.name}: "
                        f"+{new_records} samples (offset={state.offset})"
                    )

    files_tracked.set(len(file_states))
    if total_new:
        save_state()


def main() -> None:
    print(
        f"[tb-exporter] starting on :{SCRAPE_PORT}, watching {LOGS_ROOT} "
        f"(incremental, state={STATE_FILE})"
    )
    load_state()
    start_http_server(SCRAPE_PORT)
    while True:
        try:
            scan_once()
        except Exception as exc:
            print(f"[tb-exporter] scan failed: {exc}")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
