"""Evaluate session predictions against ground truth with JudgeLLM.

Reads session JSONL files produced by distill/run.py, collects the final
assistant-visible response for each sample, and asks JudgeLLM to decide
whether it matches the dataset ground truth.

Output: one JSONL line per (item_id, sample_index) with correctness judgement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
_TOOL_CALL_RE = re.compile(r"<tool_call>[\s\S]*?</tool_call>")


def _load_records(path: Path) -> list[dict] | None:
    try:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return None


def _parse_session(session_path: Path) -> tuple[str | None, bool]:
    """Return the final assistant-visible content and tool-call flag."""
    records = _load_records(session_path)
    if not records:
        return None, False

    tool_called = any(r.get("role") == "assistant" and r.get("tool_calls") for r in records)

    last = next((r for r in reversed(records) if r.get("role") == "assistant"), None)
    if last is None:
        return None, tool_called

    visible = _TOOL_CALL_RE.sub("", _THINK_RE.sub("", last.get("content", ""))).strip()
    if not visible:
        return None, tool_called

    return visible, tool_called


# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------


class JudgeLLM:
    """Lightweight OpenAI judge for answer equivalence, tuned for thinking models."""

    _SYSTEM = (
        "You are an answer judge. Decide whether the prediction correctly answers the question "
        "using the ground truth as reference. Respond with exactly one word only: YES or NO."
    )

    _YES_WORD_RE = re.compile(r"\bYES\b", re.IGNORECASE)
    _NO_WORD_RE = re.compile(r"\bNO\b", re.IGNORECASE)

    def __init__(self, model: str = "gemini-3.0-flash-preview", base_url: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE")
        self._thread_local = threading.local()

    def _get_client(self) -> OpenAI:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=self.base_url,
            )
            self._thread_local.client = client
        return client

    @classmethod
    def _parse_judgement(cls, content: str) -> bool | None:
        """Extract YES/NO from reply. None means undetermined (retry-worthy)."""
        if not content:
            return None
        text = content.strip()
        # Fast path: the whole reply is just YES / NO (possibly with trailing punctuation).
        head = re.split(r"[\s.,:;!?]+", text, maxsplit=1)[0].upper()
        if head == "YES":
            return True
        if head == "NO":
            return False
        # Fallback: scan the last non-empty line for a lone YES or NO.
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            has_yes = bool(cls._YES_WORD_RE.search(stripped))
            has_no = bool(cls._NO_WORD_RE.search(stripped))
            if has_yes and not has_no:
                return True
            if has_no and not has_yes:
                return False
            break
        return None

    def judge(self, predicted: str, ground_truth: str, question: str = "", retries: int = 3) -> bool:
        """Return True if predicted answer is semantically equivalent to ground truth."""
        if not predicted or not ground_truth:
            return False

        user_msg = "\n\n".join(
            [
                f"question:\n{question}",
                f"ground_truth:\n{ground_truth}",
                f"prediction:\n{predicted}",
            ]
        )

        client = self._get_client()
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=16384,
                    temperature=1.0,
                    top_p=0.95,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                    },
                    # reasoning_effort="high",
                )
                # import pdb; pdb.set_trace()
                verdict = self._parse_judgement(resp.choices[0].message.content or "")
                if verdict is not None:
                    return verdict
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(2**attempt)
        return False


# ---------------------------------------------------------------------------
# Session discovery & answer evaluation
# ---------------------------------------------------------------------------


def _find_session_trace(sessions_dir: Path, item_id, suffix: str) -> Path | None:
    session_dir = sessions_dir / f"session_{item_id}{suffix}"
    files = list(session_dir.glob("*.jsonl")) if session_dir.is_dir() else []
    return files[0] if files else None


def _process_item(
    item: dict,
    sample_indices: list[int],
    sessions_dir: Path,
    answer_field: str,
    mode: str = "tool",
) -> list[dict]:
    item_id = item.get("id")
    gt = str(item.get(answer_field, "")).strip()
    question = str(item.get("question", "")).strip()
    if not gt:
        return []

    results = []
    for si in sample_indices:
        trace_path = _find_session_trace(sessions_dir, item_id, f"_s{si}")
        if trace_path is None:
            results.append(
                {
                    "id": item_id,
                    "sample_index": si,
                    "question": question,
                    "ground_truth": gt,
                    "predicted": "",
                    "tool_called": False,
                    "trace_missing": True,
                }
            )
            continue
        predicted, tool_called = _parse_session(trace_path)
        if mode == "notool" and tool_called:
            print(
                f"[warn] notool mode but tool_called=True for id={item_id}, sample_index={si} "
                f"({trace_path}); check config_notool.json."
            )
        results.append(
            {
                "id": item_id,
                "sample_index": si,
                "question": question,
                "ground_truth": gt,
                "predicted": predicted or "",
                "tool_called": tool_called,
                "trace_missing": False,
            }
        )
    return results


def _evaluate_sample(result: dict, judge: JudgeLLM | None) -> dict:
    predicted = result["predicted"]
    ground_truth = result["ground_truth"]
    question = result.pop("question", "")
    if result.get("trace_missing") or not predicted:
        result["correct"] = False
    else:
        result["correct"] = judge.judge(predicted, ground_truth, question=question) if judge else False
    return result


def process_dataset(
    dataset_path: Path,
    sessions_dir: Path,
    sample_indices: list[int],
    answer_field: str,
    judge: JudgeLLM | None = None,
    num_workers: int = 8,
    mode: str = "tool",
) -> list[dict]:
    items = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    pending_results = []
    for item in items:
        pending_results.extend(_process_item(item, sample_indices, sessions_dir, answer_field, mode))

    if not pending_results:
        return []

    results = [None] * len(pending_results)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_evaluate_sample, result, judge): idx
            for idx, result in enumerate(pending_results)
        }
        with tqdm(total=len(futures), desc="Evaluating answers") as pbar:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                pbar.update(1)
    return [result for result in results if result is not None]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate final assistant responses against ground truth with JudgeLLM."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--sessions-dir", required=True)
    p.add_argument("--sample-indices", required=True, help="Comma-separated, e.g. 0,1,2")
    p.add_argument("--answer-field", default="answer")
    p.add_argument("--output", required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--mode",
        choices=["tool", "notool"],
        default="tool",
        help="Rollout mode. In 'notool' mode, samples with tool_called=True trigger an immediate "
        "warning (usually means the no-tool config was misconfigured).",
    )
    args = p.parse_args()

    dataset_path = Path(args.dataset).resolve()
    sessions_dir = Path(args.sessions_dir).resolve()
    sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
    output_path = Path(args.output).resolve()

    judge = JudgeLLM()
    print(
        f"Judge: {judge.model}\nDataset: {dataset_path}\nSessions: {sessions_dir}\n"
        f"Samples: {sample_indices}\nMode: {args.mode}"
    )

    results = process_dataset(
        dataset_path,
        sessions_dir,
        sample_indices,
        args.answer_field,
        judge,
        num_workers=args.num_workers,
        mode=args.mode,
    )

    missing = sum(1 for r in results if r.get("trace_missing"))
    if missing:
        print(f"[warn] {missing} sample(s) had missing session traces; counted as correct=False.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(
        f"\nTotal: {total}, Correct: {correct}, Accuracy: {correct / total:.3f}" if total else "No results."
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
