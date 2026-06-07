"""Live progress dashboard for eval runs.

Shows a rich.Live table with one row per task that has entered
RUNNING/DONE state. The footer shows running progress + accuracy.

Falls back to a no-op (plain print) when stdout is not a TTY so that
piped / redirected runs still produce normal logs.

Usage:
    with EvalDashboard(items, enabled=sys.stdout.isatty()) as dash:
        for task ...:
            dash.start(task_id)
            ...
            dash.update(task_id, tools=n)
            ...
            dash.finish(task_id, ok=True, tools=n, elapsed_s=...)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from ..core.benchmark import task_key
from ..scoring.vision2web import is_vision2web_source


_STATE_STYLE = {
    "PENDING": "dim",
    "RUNNING": "yellow",
    "DONE": "green",
    "FAILED": "red",
}


def _uses_continuous_score(source_dataset: str | None) -> bool:
    return is_vision2web_source(source_dataset)


def _score_style(score: float, continuous: bool) -> str:
    if continuous:
        if score >= 50:
            return "bold green"
        if score <= 0:
            return "bold red"
        return "yellow"
    return "bold green" if score >= 1 else "bold red"


class EvalDashboard:
    """Manages a live table of per-task state. Thread/async safe."""

    def __init__(self, items: Iterable[dict], enabled: bool = True,
                 refresh_per_second: float = 4.0,
                 question_max_width: int = 60):
        self.enabled = enabled
        self._rows: OrderedDict[str, dict] = OrderedDict()
        self._all_keys: list[str] = []
        for item in items:
            key = task_key(item)
            source_dataset = item.get("source_dataset")
            self._all_keys.append(key)
            # Keep query around but don't add to rows yet — only RUNNING/DONE
            # tasks appear in the visible table.
            self._rows[key] = {
                "id": item["id"],
                "task_key": key,
                "continuous_score": _uses_continuous_score(source_dataset),
                "query": item.get("query", ""),
                "state": "PENDING",
                "rounds": 0,      # number of LLM turns completed so far
                "tools": 0,
                "score": None,    # binary 0/1 or continuous 0-100
                "score_avg": None,   # mean per-sample score for avg@k
                "samples_done": 0,   # how many of k samples have finished
                "elapsed_s": None,
                "visible": False,
            }
        self.total = len(self._all_keys)
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._console = Console()
        self._live: Live | None = None
        self._question_max_width = question_max_width

    # ---- lifecycle -----------------------------------------------------
    def __enter__(self) -> "EvalDashboard":
        self._t0 = time.time()
        if self.enabled:
            self._live = Live(self._render(), console=self._console,
                              refresh_per_second=4, vertical_overflow="visible")
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(exc_type, exc, tb)

    # ---- mutation ------------------------------------------------------
    def start(self, task_key: str) -> None:
        with self._lock:
            row = self._rows.get(task_key)
            if row is None:
                return
            row["state"] = "RUNNING"
            row["visible"] = True
        self._refresh()

    def update(self, task_key: str, **fields) -> None:
        with self._lock:
            row = self._rows.get(task_key)
            if row is None:
                return
            row.update(fields)
        self._refresh()

    def finish(self, task_key: str, score: int | float, tools: int = 0,
               rounds: int | None = None,
               elapsed_s: float | None = None,
               failed: bool = False,
               score_avg: float | None = None) -> None:
        with self._lock:
            row = self._rows.get(task_key)
            if row is None:
                return
            row["state"] = "FAILED" if failed else "DONE"
            row["tools"] = tools
            if rounds is not None:
                row["rounds"] = rounds
            row["score"] = score
            if score_avg is not None:
                row["score_avg"] = score_avg
            row["elapsed_s"] = elapsed_s
            row["visible"] = True
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # ---- rendering -----------------------------------------------------
    def _render(self) -> Group:
        with self._lock:
            visible_rows = [r for r in self._rows.values() if r["visible"]]
            done_count = sum(1 for r in self._rows.values() if r["state"] in ("DONE", "FAILED"))
            running_count = sum(1 for r in self._rows.values() if r["state"] == "RUNNING")
            done_rows = [r for r in self._rows.values() if r["state"] in ("DONE", "FAILED")]
            continuous_mode = any(r.get("continuous_score") for r in done_rows)
            scored_rows = [r for r in done_rows if r.get("score") is not None]
            if continuous_mode:
                correct = sum(1 for r in scored_rows if float(r["score"]) >= 50)
                avg_score = (
                    sum(float(r["score"]) for r in scored_rows) / len(scored_rows)
                    if scored_rows else 0.0
                )
            else:
                correct = sum(1 for r in scored_rows if r["score"] == 1)
                avg_score = None
            avg_k_rows = [r for r in done_rows if r.get("score_avg") is not None]
            sum_score_avg = sum((r.get("score_avg") or 0.0) for r in avg_k_rows)
            has_avg_k = bool(avg_k_rows)

        table = Table(show_header=True, header_style="bold cyan",
                      expand=True, padding=(0, 1))
        table.add_column("task", style="cyan", width=18, no_wrap=True)
        table.add_column("question", overflow="fold", max_width=self._question_max_width)
        table.add_column("state", width=8, no_wrap=True)
        table.add_column("rounds", justify="right", width=6, no_wrap=True)
        table.add_column("tools", justify="right", width=5, no_wrap=True)
        table.add_column("score", justify="center", width=7, no_wrap=True)
        if has_avg_k:
            table.add_column("avg@k", justify="right", width=6, no_wrap=True)
        table.add_column("time", justify="right", width=7, no_wrap=True)

        for r in visible_rows:
            state = r["state"]
            state_text = Text(state, style=_STATE_STYLE.get(state, ""))
            score = r["score"]
            row_continuous = bool(r.get("continuous_score"))
            if score is None:
                score_text = Text("—", style="dim")
            elif row_continuous:
                score_f = float(score)
                score_text = Text(f"{score_f:.1f}", style=_score_style(score_f, True))
            elif score == 1:
                score_text = Text("✓ 1", style="bold green")
            else:
                score_text = Text("✗ 0", style="bold red")
            blank = "" if state == "PENDING" else None
            rounds_str = blank if blank is not None else str(r["rounds"])
            tools_str = blank if blank is not None else str(r["tools"])
            time_str = f"{r['elapsed_s']:.0f}s" if r["elapsed_s"] is not None else ""
            q = r["query"]
            if len(q) > self._question_max_width * 2:
                q = q[: self._question_max_width * 2 - 1] + "…"
            row_cells = [str(r["task_key"]), q, state_text, rounds_str, tools_str,
                         score_text]
            if has_avg_k:
                sa = r.get("score_avg")
                if sa is None:
                    avg_cell = Text("—", style="dim")
                else:
                    if row_continuous:
                        avg_cell = Text(
                            f"{sa:.1f}",
                            style=_score_style(float(sa), True),
                        )
                    else:
                        style = "bold green" if sa >= 0.999 else (
                            "bold red" if sa <= 0.001 else "yellow")
                        avg_cell = Text(f"{sa:.2f}", style=style)
                row_cells.append(avg_cell)
            row_cells.append(time_str)
            table.add_row(*row_cells)

        elapsed = (time.time() - self._t0) if self._t0 else 0
        footer_text = (
            f"done {done_count}/{self.total}  "
            f"running {running_count}  "
        )
        if continuous_mode:
            footer_text += f"avg_score {avg_score:.1f}/100  "
        else:
            acc = correct / done_count if done_count else 0.0
            footer_text += f"correct {correct}  acc {acc:.1%}  "
        if has_avg_k:
            avg_k_acc = sum_score_avg / len(avg_k_rows) if avg_k_rows else 0.0
            if continuous_mode:
                footer_text += f"avg@k {avg_k_acc:.1f}/100  "
            else:
                footer_text += f"avg@k {avg_k_acc:.1%}  "
        footer_text += f"elapsed {elapsed:.0f}s"
        footer = Text(footer_text, style="bold")
        return Group(table, footer)
