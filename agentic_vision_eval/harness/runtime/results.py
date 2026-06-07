"""Run-level result aggregation helpers.

Used by both `eval.py` (end-of-run summary) and `rejudge.py`
(re-aggregate after re-judging).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from ..scoring.vision2web import is_vision2web_source


def _record_key(record: dict, path: str | Path | None = None) -> str:
    if record.get("task_key"):
        return str(record["task_key"])
    if path is not None:
        return Path(path).parent.name.removeprefix("task_")
    return str(record.get("id"))


def failed_ids(run_dir: Path) -> set[str]:
    """Task keys whose result.json has a non-null `error` (timeouts, agent
    crashes, judge errors during the original run, ...)."""
    failed: set[str] = set()
    for f in glob.glob(str(run_dir / "task_*" / "result.json")):
        try:
            r = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            failed.add(Path(f).parent.name.removeprefix("task_"))
            continue
        if r.get("error"):
            failed.add(_record_key(r, f))
    return failed


def aggregate_results(run_dir: Path) -> dict[str, Any]:
    """Walk task_*/result.json under run_dir and rewrite results.jsonl.

    Returns aggregate stats:
        total, correct, accuracy, errors, timeouts, total_tokens,
        avg_tool_calls (mean of per-task n_tool_calls), results_path,
        error_ids, timeout_ids,
        avg_k, accuracy_avg_k   (= mean of per-task score_avg; equals
                                 `accuracy` when avg_k == 1)
    """
    rows: list[dict] = []
    for f in sorted(glob.glob(str(run_dir / "task_*" / "result.json"))):
        try:
            r = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            continue
        r.setdefault("task_key", _record_key(r, f))
        rows.append(r)
    rows.sort(
        key=lambda r: (
            str(r.get("source_dataset") or ""),
            r["id"],
            r["task_key"],
        )
    )

    scores = [r.get("score", 0) for r in rows]
    # Continuous-score detection: Vision2Web uses float scores in [0, 100]
    # rather than binary 0/1. Key off source_dataset so it stays correct even
    # when every task scored <= 1 (e.g. an all-zero run); keep the float>1
    # heuristic as a fallback for records that lack source_dataset.
    is_continuous = (any(is_vision2web_source(r.get("source_dataset")) for r in rows)
                     or any(isinstance(s, float) and s > 1 for s in scores))
    if is_continuous:
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        correct = sum(1 for s in scores if s >= 50)
    else:
        avg_score = None
        correct = sum(1 for r in rows if r.get("score") == 1)
    errors = [r for r in rows if r.get("error")]
    timeouts = [r for r in errors if "Timeout" in (r.get("error") or "")]
    total_tokens = sum((r.get("usage_total") or {}).get("total_tokens", 0) or 0
                       for r in rows)

    # Average number of tool calls per sample — how many tools the model
    # invoked on average to reach an answer.
    avg_tool_calls = (round(sum(r.get("n_tool_calls", 0) or 0 for r in rows) / len(rows), 2)
                      if rows else 0.0)

    # avg@k: use score_avg when present (rolled out by run_one); fall back to
    # the legacy 0/1 score for older single-sample runs being re-aggregated.
    avg_k = max((r.get("avg_k", 1) or 1) for r in rows) if rows else 1
    if rows:
        score_avgs = [
            r["score_avg"] if "score_avg" in r else float(r.get("score") or 0)
            for r in rows
        ]
        accuracy_avg_k = round(sum(score_avgs) / len(score_avgs), 4)
    else:
        accuracy_avg_k = 0.0

    out = run_dir / "results.jsonl"
    tmp = out.with_suffix(".jsonl.new")
    with open(tmp, "w") as rf:
        for r in rows:
            rf.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)

    result: dict[str, Any] = {
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "avg_k": avg_k,
        "accuracy_avg_k": accuracy_avg_k,
        "errors": len(errors),
        "timeouts": len(timeouts),
        "total_tokens": total_tokens,
        "avg_tool_calls": avg_tool_calls,
        "results_path": str(out),
        "error_ids": [_record_key(r) for r in errors],
        "timeout_ids": [_record_key(r) for r in timeouts],
    }
    if avg_score is not None:
        result["avg_score"] = avg_score
    return result


def print_summary(agg: dict, total_elapsed: float | None = None) -> None:
    extra = f"  total_elapsed={total_elapsed:.1f}s" if total_elapsed is not None else ""
    if agg.get("avg_score") is not None:
        print(f"\n▶ avg_score: {agg['avg_score']}/100  "
              f"(correct≥50: {agg['correct']}/{agg['total']})  "
              f"errors={agg['errors']} (timeouts={agg['timeouts']})  "
              f"total_tokens={agg['total_tokens']}  "
              f"avg_tool_calls={agg.get('avg_tool_calls', 0)}{extra}")
    else:
        print(f"\n▶ accuracy: {agg['correct']}/{agg['total']} = {agg['accuracy']:.3f}  "
              f"errors={agg['errors']} (timeouts={agg['timeouts']})  "
              f"total_tokens={agg['total_tokens']}  "
              f"avg_tool_calls={agg.get('avg_tool_calls', 0)}{extra}")
    k = agg.get("avg_k", 1)
    if k > 1:
        print(f"▶ avg@{k}:   {agg['accuracy_avg_k']:.3f}  "
              f"(mean of per-task score_avg over {agg['total']} tasks)")
    if agg["errors"]:
        print(f"  error ids:   {agg['error_ids']}")
    print(f"  results  -> {agg['results_path']}")
