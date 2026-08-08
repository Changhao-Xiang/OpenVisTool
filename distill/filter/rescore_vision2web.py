"""Re-score saved Vision2Web HTML predictions without running the rollout model."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from distill.filter.evaluate_instructive_trajectory_ablation import (
    CONDITIONS,
    TOOL_TRAJECTORY_CONDITIONS,
    Vision2WebScorer,
    condition_order,
    iter_jsonl_allow_partial_tail,
    load_jsonl,
    preflight_endpoint,
    resolve_item_images,
    run_summary,
)


def parse_conditions(raw: str) -> tuple[str, ...]:
    conditions = tuple(name.strip() for name in raw.split(",") if name.strip())
    invalid = set(conditions) - set(CONDITIONS)
    if not conditions:
        raise ValueError("--conditions must select at least one condition")
    if invalid:
        raise ValueError(f"Unknown --conditions: {sorted(invalid)}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("--conditions must not contain duplicates")
    return conditions


def score_record(
    source: dict[str, Any],
    *,
    item: dict[str, Any],
    dataset_path: Path,
    scorer: Vision2WebScorer,
    conditions: tuple[str, ...],
    prediction_index: int,
    judge_model: str,
    judge_api_base: str,
) -> dict[str, Any]:
    index = int(source["index"])
    original_images = resolve_item_images(dataset_path, item)
    state: dict[str, dict[str, Any]] = {condition: {} for condition in conditions}
    for condition in condition_order(conditions, index, prediction_index):
        predictions = source["conditions"][condition]["predictions"]
        if prediction_index >= len(predictions):
            raise IndexError(
                f"{source['key']} condition {condition} has only {len(predictions)} predictions"
            )
        prediction = str(predictions[prediction_index])
        score, method, detail = scorer.score(
            prediction,
            item=item,
            dataset_path=dataset_path,
            original_images=original_images,
        )
        state[condition].update(
            {
                "prediction_indices": [prediction_index],
                "prediction_sha256": [
                    hashlib.sha256(prediction.encode("utf-8")).hexdigest()
                ],
                "scores": [round(score, 6)],
                "methods": [method],
                "score_details": [detail],
                "avg_k": round(score, 6),
                "correct_count": None,
            }
        )

    gains: dict[str, float] = {}
    if "no_tool" in state:
        no_tool = state["no_tool"]["avg_k"]
        for condition in TOOL_TRAJECTORY_CONDITIONS:
            if condition in state:
                gains[condition] = round(state[condition]["avg_k"] - no_tool, 6)
    if all(condition in state for condition in TOOL_TRAJECTORY_CONDITIONS):
        gains["paired_delta"] = round(
            state["correctness_instructive"]["avg_k"]
            - state["correctness_only"]["avg_k"],
            6,
        )
    return {
        "key": source["key"],
        "domain": "vision2web",
        "index": index,
        "id": source.get("id", index),
        "source_dataset": source.get("source_dataset"),
        "query": source.get("query", ""),
        "answer": source.get("answer", ""),
        "scorer": "vision2web",
        "score_scale": "0_to_1",
        "conditions": state,
        "gains": gains,
        "prefix": source.get("prefix", {}),
        "traces": source.get("traces", {}),
        "trajectory_sources": source.get("trajectory_sources", {}),
        "rescore": {
            "source_record": source["key"],
            "prediction_index": prediction_index,
            "avg_k": 1,
            "judge_model": judge_model,
            "judge_api_base": judge_api_base,
            "rollout_model_called": False,
        },
    }


def run(args: argparse.Namespace) -> None:
    if args.prediction_index < 0 or args.num_workers <= 0:
        raise ValueError("--prediction-index must be non-negative and --num-workers positive")
    conditions = parse_conditions(args.conditions)
    dataset_path = Path(args.dataset).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path in (dataset_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    items = load_jsonl(dataset_path)
    source_records = sorted(load_jsonl(input_path), key=lambda record: int(record["index"]))
    if args.sample_range:
        parts = [int(part) if part else None for part in args.sample_range.split(":")]
        source_records = source_records[slice(*parts)]

    existing = (
        list(iter_jsonl_allow_partial_tail(output_path)) if output_path.is_file() else []
    )
    for record in existing:
        config = record.get("rescore", {})
        expected = {
            "prediction_index": args.prediction_index,
            "judge_model": args.judge_model,
            "judge_api_base": args.judge_api_base,
        }
        if any(config.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Existing output has incompatible rescore config: {config}")
        if set(record.get("conditions", {})) != set(conditions):
            raise ValueError("Existing output has incompatible conditions")
    processed = {str(record["key"]) for record in existing}
    pending = [record for record in source_records if str(record["key"]) not in processed]

    if pending and not args.skip_preflight:
        preflight_endpoint(
            base_url=args.judge_api_base,
            model=args.judge_model,
            label="GPT-5.5 judge",
        )
    scorer = Vision2WebScorer(
        model=args.judge_model,
        base_url=args.judge_api_base,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        retries=args.judge_retries,
        render_timeout_ms=args.render_timeout_ms,
        render_wait_ms=args.render_wait_ms,
        render_slots=args.render_slots,
        max_tokens_parameter=args.judge_token_parameter,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "vision2web_rescore",
        "source_input": str(input_path),
        "dataset": str(dataset_path),
        "n": len(source_records),
        "conditions": list(conditions),
        "prediction_index": args.prediction_index,
        "avg_k": 1,
        "judge_model": args.judge_model,
        "judge_api_base": args.judge_api_base,
        "judge_token_parameter": args.judge_token_parameter,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_temperature": args.judge_temperature,
        "rollout_model_called": False,
    }
    manifest_path = output_path.parent / "run_config.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(f"Existing manifest is incompatible: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"Vision2Web rescore: source={len(source_records)} resumed={len(source_records) - len(pending)} "
        f"pending={len(pending)} prediction_index={args.prediction_index} avg@1\n"
        f"Judge={args.judge_model} @ {args.judge_api_base}; workers={args.num_workers}\n"
        f"Conditions={','.join(conditions)}; rollout_model_called=false\nOutput={output_path}"
    )
    failures = 0
    if pending:
        with output_path.open(
            "a", encoding="utf-8", buffering=1
        ) as output_file, ThreadPoolExecutor(
            max_workers=min(args.num_workers, len(pending))
        ) as executor:
            futures = {
                executor.submit(
                    score_record,
                    source,
                    item=items[int(source["index"])],
                    dataset_path=dataset_path,
                    scorer=scorer,
                    conditions=conditions,
                    prediction_index=args.prediction_index,
                    judge_model=args.judge_model,
                    judge_api_base=args.judge_api_base,
                ): source
                for source in pending
            }
            with tqdm(
                total=len(futures), desc="Rescore vision2web", unit="item"
            ) as progress:
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        print(
                            f"[error] index={source.get('index')} id={source.get('id')}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_file.flush()
                    progress.update(1)
                    progress.set_postfix(failures=failures)
    if failures:
        raise RuntimeError(f"{failures} items failed; rerun the same command to resume")

    run_summary(
        argparse.Namespace(
            inputs=[str(output_path)],
            output=str(Path(args.summary_output).expanduser().resolve()),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.summary_seed,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing Vision2Web ablation JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--judge-token-parameter", choices=("max_tokens", "max_completion_tokens"), default="max_completion_tokens")
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--prediction-index", type=int, default=0)
    parser.add_argument("--sample-range", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--render-timeout-ms", type=int, default=30_000)
    parser.add_argument("--render-wait-ms", type=int, default=500)
    parser.add_argument("--render-slots", type=int, default=4)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--summary-seed", type=int, default=20260723)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
