"""On-disk layout for one resampling run.

    artifacts/rollouts/<run>/
      manifest.json                     model, decoding, tokenizer and engine provenance
      runs.jsonl                        settings of each invocation, which resumes may change
      traces.jsonl                      one BaseTrace per line
      skipped.jsonl                     case_id and reason for every trace not prepared
      rollouts/case_id=<id>/part-*.parquet

Nothing is ever rewritten: a batch lands as a new part file, so an interrupted run resumes by
counting what is already there.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from dctax.config import data_root
from dctax.rollouts.schema import BaseTrace, Rollout

_SAFE_CASE_ID = re.compile(r"[A-Za-z0-9._-]+")

ROLLOUT_SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("trace_id", pa.string()),
        ("boundary_index", pa.int32()),
        ("rollout_index", pa.int32()),
        ("prefix_token_count", pa.int32()),
        ("prefix_sha256", pa.string()),
        ("output_token_ids", pa.list_(pa.int32())),
        ("continuation", pa.string()),
        ("finish_reason", pa.string()),
        ("seed", pa.int64()),
    ]
)


@dataclass(frozen=True)
class Stored:
    """Rollouts already on disk at one boundary."""

    count: int
    next_rollout_index: int


@dataclass(frozen=True)
class Pending:
    """Rollouts still owed at one boundary."""

    boundary_index: int
    n_needed: int
    first_rollout_index: int


def run_dir(run: str) -> Path:
    return data_root() / "rollouts" / run


def _case_dir(run: str, case_id: str) -> Path:
    if not _SAFE_CASE_ID.fullmatch(case_id):
        raise ValueError(f"case_id {case_id!r} is not safe as a path segment")
    return run_dir(run) / "rollouts" / f"case_id={case_id}"


def _append_jsonl(path: Path, payloads: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_manifest(run: str, payload: dict) -> Path:
    """Record the run's provenance, refusing to change it once written."""
    path = run_dir(run) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"{path}: manifest differs from this run's settings; "
                f"use a new run name rather than mixing generations"
            )
        return path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def read_manifest(run: str) -> dict:
    return json.loads((run_dir(run) / "manifest.json").read_text(encoding="utf-8"))


def append_run(run: str, settings: dict) -> int:
    """Record one invocation's settings; these may differ between resumes."""
    return _append_jsonl(run_dir(run) / "runs.jsonl", [settings])


def load_runs(run: str) -> list[dict]:
    return _read_jsonl(run_dir(run) / "runs.jsonl")


def append_traces(run: str, traces: Sequence[BaseTrace]) -> int:
    return _append_jsonl(run_dir(run) / "traces.jsonl", (t.as_dict() for t in traces))


def load_traces(run: str) -> list[BaseTrace]:
    return [BaseTrace.from_dict(p) for p in _read_jsonl(run_dir(run) / "traces.jsonl")]


def trace_ids(run: str) -> set[str]:
    """Trace ids already written, so a resumed run does not duplicate them."""
    return {p["trace_id"] for p in _read_jsonl(run_dir(run) / "traces.jsonl")}


def log_skips(run: str, errors: Iterable) -> int:
    """Record excluded traces so exclusions stay auditable."""
    return _append_jsonl(
        run_dir(run) / "skipped.jsonl",
        ({"case_id": e.case_id, "reason": e.reason} for e in errors),
    )


def append_rollouts(run: str, rollouts: Sequence[Rollout]) -> Path | None:
    """Write one batch as a new part file. All rollouts must share a case."""
    if not rollouts:
        return None
    case_ids = {r.case_id for r in rollouts}
    if len(case_ids) != 1:
        raise ValueError(f"a part file holds one case, got {sorted(case_ids)}")

    directory = _case_dir(run, rollouts[0].case_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"part-{len(list(directory.glob('part-*.parquet'))):05d}.parquet"

    table = pa.Table.from_pylist([r.as_dict() for r in rollouts], schema=ROLLOUT_SCHEMA)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    return path


def part_files(run: str) -> list[Path]:
    """Every rollout part file of a run, in a stable order."""
    root = run_dir(run) / "rollouts"
    return sorted(root.glob("case_id=*/part-*.parquet")) if root.exists() else []


def progress(run: str) -> dict[tuple[str, int], Stored]:
    """Rollouts already stored per (trace_id, boundary_index).

    Only the three identity columns are read, so resuming does not load continuations.
    """
    counts: dict[tuple[str, int], int] = {}
    highest: dict[tuple[str, int], int] = {}
    for path in part_files(run):
        table = pq.read_table(path, columns=["trace_id", "boundary_index", "rollout_index"])
        for trace_id, boundary, index in zip(
            table.column("trace_id").to_pylist(),
            table.column("boundary_index").to_pylist(),
            table.column("rollout_index").to_pylist(),
        ):
            key = (trace_id, int(boundary))
            counts[key] = counts.get(key, 0) + 1
            highest[key] = max(highest.get(key, -1), int(index))
    return {key: Stored(counts[key], highest[key] + 1) for key in counts}


def pending(
    trace: BaseTrace, stored: dict[tuple[str, int], Stored], n_rollouts: int
) -> list[Pending]:
    """Boundaries of `trace` still short of `n_rollouts`, with the next index to use."""
    out = []
    for boundary_index in range(trace.n_boundaries):
        done = stored.get((trace.trace_id, boundary_index))
        have = done.count if done else 0
        if have < n_rollouts:
            out.append(
                Pending(
                    boundary_index=boundary_index,
                    n_needed=n_rollouts - have,
                    first_rollout_index=done.next_rollout_index if done else 0,
                )
            )
    return out


def verify(run: str) -> dict:
    """Parse every part file and report what is readable.

    Run this after copying a run between machines. A part truncated in transit keeps its
    final name and a plausible size, so file counts and directory listings both pass; only
    reading the parquet footer detects it. Transfer with `rsync --partial-dir=.rsync-partial`
    rather than `--partial`, which leaves interrupted files under the final name.
    """
    parts = part_files(run)
    rows, unreadable = 0, []
    for path in parts:
        try:
            rows += pq.read_metadata(path).num_rows
            pq.read_schema(path)
        except Exception as exc:  # noqa: BLE001: any failure to parse is a failure to trust
            unreadable.append((str(path), f"{type(exc).__name__}: {exc}"))
    return {"parts": len(parts), "rows": rows, "unreadable": unreadable}


def load_rollouts(run: str, case_id: str | None = None) -> list[Rollout]:
    """Every stored rollout, or only those for one case."""
    paths = (
        sorted(_case_dir(run, case_id).glob("part-*.parquet"))
        if case_id is not None
        else part_files(run)
    )
    out: list[Rollout] = []
    for path in paths:
        out.extend(Rollout.from_dict(row) for row in pq.read_table(path).to_pylist())
    return out
