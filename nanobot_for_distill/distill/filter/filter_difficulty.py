"""Filter hard samples with no-tool vLLM rollouts.

Given a dataset JSONL and a media directory, this script performs k no-tool
rollouts per sample through an OpenAI-compatible vLLM API, evaluates each
response with either the rule matcher or JudgeLLM, computes avg@k, and keeps
only samples whose avg@k is below a threshold.

Example:
    python distill/filter/filter_difficulty.py \
        --dataset dataset/Chart/ChartVerse-SFT-600K/ChartVerse-SFT-600K.jsonl \
        --model Qwen3-VL-30B-A3B-Instruct
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.filter.evaluate_answer_judge import JudgeLLM
from distill.filter.evaluate_answer_rule import check_match
from distill.filter.evaluate_html_vlm_judge import (
    HtmlRenderer,
    VlmHtmlJudge,
    _parse_viewport,
    _reference_viewport,
    _sample_output_stem,
    rule_check_html,
)

DEFAULT_QUERY_FALLBACKS = ("query", "problem")
DEFAULT_MEDIA_FALLBACKS = ("images",)
DEFAULT_RETRIES = 3


def infer_media_dir(dataset_path: Path) -> Path:
    return (dataset_path.parent / "images").resolve()


def infer_output_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_difficulty_filtered.jsonl").resolve()


def infer_progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(".progress.jsonl").resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            items.append(item)
    return items


def parse_sample_range(expr: str) -> slice:
    raw_expr = expr.strip()
    if not raw_expr:
        return slice(None)

    if ":" not in raw_expr:
        try:
            index = int(raw_expr)
        except ValueError as exc:
            raise ValueError(f"Invalid --sample-range: {expr!r}") from exc
        return slice(index, index + 1)

    parts = raw_expr.split(":")
    if len(parts) > 3:
        raise ValueError(
            "--sample-range must follow Python slice syntax like '100:500', ':500', '100:', or '::2'"
        )

    parsed_parts: list[int | None] = []
    for part in parts:
        if not part:
            parsed_parts.append(None)
            continue
        try:
            parsed_parts.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid --sample-range component: {part!r}") from exc

    return slice(*parsed_parts)


def get_text_field(item: dict[str, Any], field: str, fallbacks: tuple[str, ...] = ()) -> str:
    for candidate in (field, *fallbacks):
        value = item.get(candidate)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    raise ValueError(f"Missing non-empty text field '{field}'")


def resolve_media_paths(item: dict[str, Any], media_field: str, media_dir: Path) -> list[Path]:
    raw_value = item.get(media_field)
    if raw_value is None:
        for fallback in DEFAULT_MEDIA_FALLBACKS:
            raw_value = item.get(fallback)
            if raw_value is not None:
                break
    if raw_value is None:
        return []

    raw_paths = raw_value if isinstance(raw_value, list) else [raw_value]
    resolved: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            raise ValueError(f"Field '{media_field}' must be a string or list of strings")
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = (media_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        resolved.append(path)
    return resolved


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def load_system_prompt_markdown(path_str: str) -> tuple[str, Path | None]:
    raw_path = path_str.strip()
    if not raw_path:
        return "", None

    prompt_path = Path(raw_path).expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt markdown file not found: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"System prompt markdown file is empty: {prompt_path}")
    return system_prompt, prompt_path


def build_message_content(query: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    content.append({"type": "text", "text": query})
    return content


def response_to_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif hasattr(part, "type") and getattr(part, "type") == "text":
                parts.append(str(getattr(part, "text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def get_item_key(item: dict[str, Any], index: int) -> str:
    item_id = item.get("id")
    if item_id is not None:
        return f"id:{item_id}"
    return f"index:{index}"


def build_output_payload(result: dict[str, Any], keep_metadata: bool) -> dict[str, Any]:
    payload = dict(result["item"])
    payload["avg_k_notool"] = result["avg_k"]
    if keep_metadata:
        payload["_difficulty"] = {
            "avg_k_notool": result["avg_k"],
            "correct_count": result["correct_count"],
            "corrects": result["corrects"],
            "methods": result["methods"],
            "predictions": result["predictions"],
        }
        if result.get("scores"):
            payload["_difficulty"]["scores"] = result["scores"]
        if result.get("rule_results"):
            payload["_difficulty"]["rule_results"] = result["rule_results"]
        if result.get("rendered_images"):
            payload["_difficulty"]["rendered_images"] = result["rendered_images"]
    return payload


def _iter_jsonl_records_allow_partial_tail(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            if line_no == len(lines):
                break
            raise ValueError(f"Invalid JSON in {path} line {line_no}")
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path} line {line_no}")
        records.append(record)
    return records


def load_processed_keys(progress_path: Path) -> set[str]:
    processed_keys: set[str] = set()
    if not progress_path.is_file():
        return processed_keys

    for record in _iter_jsonl_records_allow_partial_tail(progress_path):
        key = str(record.get("key", "")).strip()
        if key:
            processed_keys.add(key)
    return processed_keys


def load_written_keys_from_output(output_path: Path) -> set[str]:
    written_keys: set[str] = set()
    if not output_path.is_file():
        return written_keys

    for record in _iter_jsonl_records_allow_partial_tail(output_path):
        item_id = record.get("id")
        if item_id is not None:
            written_keys.add(f"id:{item_id}")
    return written_keys


def evaluate_prediction(
    prediction: str,
    answer: str,
    *,
    accuracy_backend: str,
    query: str,
    tolerance: float,
    match_mode: str,
    judge: JudgeLLM | None,
    judge_retries: int,
) -> tuple[bool, str]:
    if accuracy_backend == "rule":
        return check_match(prediction, answer, tolerance=tolerance, mode=match_mode)
    if accuracy_backend == "judge":
        if judge is None:
            raise ValueError("Judge backend requested but JudgeLLM is not initialized")
        return judge.judge(prediction, answer, question=query, retries=judge_retries), "judge_llm"
    raise ValueError(f"Unsupported accuracy backend: {accuracy_backend}")


def evaluate_html_prediction(
    prediction: str,
    answer: str,
    *,
    query: str,
    reference_image: Path,
    renderer: HtmlRenderer,
    html_judge: VlmHtmlJudge,
    judge_retries: int,
    score_threshold: float,
    render_dir: Path,
    dataset_stem: str,
    item_id: Any,
    index: int,
    rollout_index: int,
    keep_rendered_images: bool,
    default_viewport: tuple[int, int],
    min_tag_ratio: float,
    min_structural_jaccard: float,
    min_text_recall: float,
    require_css: bool,
    require_rick_image: bool,
) -> tuple[bool, str, float, dict[str, Any], str]:
    rule = rule_check_html(
        prediction,
        answer,
        min_tag_ratio=min_tag_ratio,
        min_structural_jaccard=min_structural_jaccard,
        min_text_recall=min_text_recall,
        require_css=require_css,
        require_rick_image=require_rick_image,
    )
    rule_payload = {
        "passed": bool(rule["passed"]),
        "errors": rule["errors"],
        "metrics": rule["metrics"],
    }
    if not rule["passed"]:
        method = "html_rule:" + ",".join(rule["errors"] or ["failed"])
        return False, method, 0.0, rule_payload, ""

    stem = _sample_output_stem(item_id, index, rollout_index)
    rendered_path = (
        render_dir / f"{dataset_stem}_{stem}.png"
        if keep_rendered_images
        else Path(tempfile.gettempdir()) / f"{dataset_stem}_{stem}_{threading.get_ident()}.png"
    )
    viewport = _reference_viewport(reference_image, default_viewport)
    try:
        renderer.render(str(rule["extracted_html"]), rendered_path, viewport)
        score, reason, _ = html_judge.judge(
            reference_image=reference_image,
            rendered_image=rendered_path,
            retries=judge_retries,
        )
    finally:
        if not keep_rendered_images:
            rendered_path.unlink(missing_ok=True)

    rule_payload["judge_reason"] = reason
    is_correct = score >= score_threshold
    return (
        is_correct,
        f"html_vlm_score:{score:.2f}",
        round(score, 4),
        rule_payload,
        str(rendered_path) if keep_rendered_images else "",
    )


class VllmRolloutClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str = "",
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
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

    def rollout(self, content: list[dict[str, Any]], retries: int = DEFAULT_RETRIES) -> str:
        last_error: Exception | None = None
        client = self._get_client()
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response_to_text(response)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        if last_error is not None:
            raise last_error
        return ""


def process_one_item(
    item: dict[str, Any],
    *,
    index: int,
    client: VllmRolloutClient,
    judge: JudgeLLM | None,
    html_judge: VlmHtmlJudge | None,
    html_renderer: HtmlRenderer | None,
    media_dir: Path,
    query_field: str,
    answer_field: str,
    media_field: str,
    k: int,
    accuracy_backend: str,
    tolerance: float,
    match_mode: str,
    judge_retries: int,
    html_score_threshold: float,
    html_render_dir: Path,
    dataset_stem: str,
    keep_rendered_images: bool,
    html_default_viewport: tuple[int, int],
    html_min_tag_ratio: float,
    html_min_structural_jaccard: float,
    html_min_text_recall: float,
    html_require_css: bool,
    html_require_rick_image: bool,
) -> dict[str, Any]:
    query = get_text_field(item, query_field, DEFAULT_QUERY_FALLBACKS)
    answer = get_text_field(item, answer_field)
    image_paths = resolve_media_paths(item, media_field, media_dir)
    content = build_message_content(query, image_paths)

    predictions: list[str] = []
    corrects: list[int] = []
    methods: list[str] = []
    scores: list[float] = []
    rule_results: list[dict[str, Any]] = []
    rendered_images: list[str] = []
    for rollout_index in range(k):
        prediction = client.rollout(content)
        if accuracy_backend == "html_vlm":
            if html_judge is None or html_renderer is None:
                raise ValueError("HTML VLM backend requested but judge/renderer is not initialized")
            if not image_paths:
                raise ValueError("HTML VLM backend requires at least one reference image")
            is_correct, method, score, rule_payload, rendered_image = evaluate_html_prediction(
                prediction,
                answer,
                query=query,
                reference_image=image_paths[0],
                renderer=html_renderer,
                html_judge=html_judge,
                judge_retries=judge_retries,
                score_threshold=html_score_threshold,
                render_dir=html_render_dir,
                dataset_stem=dataset_stem,
                item_id=item.get("id", index),
                index=index,
                rollout_index=rollout_index,
                keep_rendered_images=keep_rendered_images,
                default_viewport=html_default_viewport,
                min_tag_ratio=html_min_tag_ratio,
                min_structural_jaccard=html_min_structural_jaccard,
                min_text_recall=html_min_text_recall,
                require_css=html_require_css,
                require_rick_image=html_require_rick_image,
            )
            scores.append(score)
            rule_results.append(rule_payload)
            rendered_images.append(rendered_image)
        else:
            is_correct, method = evaluate_prediction(
                prediction,
                answer,
                accuracy_backend=accuracy_backend,
                query=query,
                tolerance=tolerance,
                match_mode=match_mode,
                judge=judge,
                judge_retries=judge_retries,
            )
        predictions.append(prediction)
        corrects.append(int(is_correct))
        methods.append(method)

    avg_k = sum(corrects) / k if k else 0.0
    return {
        "key": get_item_key(item, index),
        "index": index,
        "id": item.get("id", index),
        "item": item,
        "avg_k": round(avg_k, 4),
        "correct_count": sum(corrects),
        "corrects": corrects,
        "methods": methods,
        "predictions": predictions,
        "scores": scores,
        "rule_results": rule_results,
        "rendered_images": rendered_images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter difficult samples with no-tool vLLM rollouts.")
    parser.add_argument("--dataset", "--input", dest="dataset", required=True, help="Input dataset JSONL.")
    parser.add_argument(
        "--media-dir",
        default="",
        help="Directory that stores dataset images. Defaults to <dataset_dir>/images.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output filtered JSONL. Defaults to <dataset_stem>_difficulty_filtered.jsonl.",
    )
    parser.add_argument(
        "--model", required=True, help="vLLM model name exposed by the OpenAI-compatible API."
    )
    parser.add_argument(
        "--rollout-api-base",
        default=os.environ.get("OPENAI_API_BASE", ""),
        help="OpenAI-compatible API base URL for rollout model. Defaults to OPENAI_API_BASE.",
    )
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--media-field", default="images")
    parser.add_argument("--k", type=int, default=5, help="Number of no-tool rollouts per sample.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Keep samples with avg@k < threshold.")
    parser.add_argument(
        "--accuracy-backend",
        choices=("rule", "judge", "html_vlm"),
        default="rule",
        help="How to judge rollout correctness. Use html_vlm for screenshot-to-HTML tasks.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Rule-match tolerance. In generic mode it is relative error for non-zero numeric refs; in point mode it is the normalized coordinate tolerance. Ignored in bbox mode.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("generic", "point", "bbox"),
        default="generic",
        help="Ground-truth format for rule matching. Used only when --accuracy-backend=rule.",
    )
    parser.add_argument("--judge-model", default="gemini-3.0-flash-preview", help="JudgeLLM model name.")
    parser.add_argument(
        "--judge-api-base",
        default="",
        help="OpenAI-compatible API base URL for judge model. Defaults to --rollout-api-base.",
    )
    parser.add_argument("--judge-retries", type=int, default=3, help="Retry count for JudgeLLM calls.")
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument(
        "--html-score-threshold",
        type=float,
        default=80.0,
        help="For --accuracy-backend=html_vlm, VLM score >= this value counts as correct.",
    )
    parser.add_argument(
        "--html-render-dir",
        default="",
        help="Directory for rendered prediction screenshots. Defaults to <output_dir>/rendered_html_difficulty.",
    )
    parser.add_argument(
        "--keep-rendered-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For html_vlm, keep rendered prediction screenshots on disk after judging.",
    )
    parser.add_argument("--html-render-timeout-ms", type=int, default=10000)
    parser.add_argument("--html-render-wait-ms", type=int, default=500)
    parser.add_argument("--html-default-viewport", default="1280x720")
    parser.add_argument("--html-min-tag-ratio", type=float, default=0.35)
    parser.add_argument("--html-min-structural-jaccard", type=float, default=0.25)
    parser.add_argument("--html-min-text-recall", type=float, default=0.25)
    parser.add_argument("--html-no-require-css", action="store_true")
    parser.add_argument("--html-no-require-rick-image", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--system-prompt-md",
        "--system-prompt-file",
        dest="system_prompt_md",
        default="",
        help="Path to a Markdown file whose contents will be sent as the system prompt.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers over dataset items.")
    parser.add_argument(
        "--sample-range",
        default="",
        help="Python slice syntax for selecting dataset items, e.g. '100:500', ':1000', '100::2', or '-500:'.",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Write avg@k and rollout details together with the original sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0")
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required")
    if not args.rollout_api_base:
        raise EnvironmentError("--rollout-api-base or OPENAI_API_BASE is required")
    if args.accuracy_backend in {"judge", "html_vlm"} and not (args.judge_api_base or args.rollout_api_base):
        raise EnvironmentError("--judge-api-base or --rollout-api-base is required for judge backend")

    dataset_path = Path(args.dataset).resolve()
    media_dir = Path(args.media_dir).resolve() if args.media_dir else infer_media_dir(dataset_path)
    output_path = Path(args.output).resolve() if args.output else infer_output_path(dataset_path)
    html_render_dir = (
        Path(args.html_render_dir).resolve()
        if args.html_render_dir
        else (output_path.parent / "rendered_html_difficulty").resolve()
    )
    system_prompt, system_prompt_path = load_system_prompt_markdown(args.system_prompt_md)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not media_dir.is_dir():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")

    items = load_jsonl(dataset_path)
    sample_slice = parse_sample_range(args.sample_range)
    selected_pairs = list(enumerate(items))[sample_slice]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = infer_progress_path(output_path)
    processed_keys = load_processed_keys(progress_path)
    if processed_keys:
        resume_source = progress_path
    else:
        processed_keys = load_written_keys_from_output(output_path)
        resume_source = output_path if processed_keys else None
    pending_pairs = [
        (index, item) for index, item in selected_pairs if get_item_key(item, index) not in processed_keys
    ]
    skipped_count = len(selected_pairs) - len(pending_pairs)
    client = VllmRolloutClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt=system_prompt,
        base_url=args.rollout_api_base,
    )
    judge_api_base = args.judge_api_base or args.rollout_api_base
    judge = (
        JudgeLLM(model=args.judge_model, base_url=judge_api_base)
        if args.accuracy_backend == "judge"
        else None
    )
    html_judge = (
        VlmHtmlJudge(
            model=args.judge_model,
            base_url=judge_api_base,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
        )
        if args.accuracy_backend == "html_vlm"
        else None
    )
    html_renderer = (
        HtmlRenderer(timeout_ms=args.html_render_timeout_ms, wait_ms=args.html_render_wait_ms)
        if args.accuracy_backend == "html_vlm"
        else None
    )
    html_default_viewport = _parse_viewport(args.html_default_viewport)
    if args.accuracy_backend == "html_vlm" and args.keep_rendered_images:
        html_render_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Dataset: {dataset_path}\nMedia dir: {media_dir}\nModel: {args.model}\n"
        f"k={args.k}, threshold={args.threshold}, workers={args.num_workers}\n"
        f"Accuracy backend: {args.accuracy_backend}"
        + (
            f" ({args.judge_model})"
            if args.accuracy_backend in {"judge", "html_vlm"}
            else f" ({args.match_mode})"
        )
        + "\n"
        + f"Match mode: {args.match_mode}, tolerance={args.tolerance}\n"
        + (
            f"HTML score threshold: {args.html_score_threshold}, render dir: {html_render_dir}, "
            f"keep rendered: {args.keep_rendered_images}\n"
            if args.accuracy_backend == "html_vlm"
            else ""
        )
        + f"System prompt md: {system_prompt_path or '-'}\n"
        + f"Sample range: {args.sample_range or ':'} -> {len(selected_pairs)} selected from {len(items)} total\n"
        + f"Resume source: {resume_source or '-'}\n"
        + f"Already processed in selection: {skipped_count}\n"
        + f"Pending: {len(pending_pairs)}"
    )

    failed = 0
    processed_count = 0
    kept_count = 0
    mean_sum = 0.0
    with (
        output_path.open("a", encoding="utf-8", buffering=1) as output_file,
        progress_path.open("a", encoding="utf-8", buffering=1) as progress_file,
        ThreadPoolExecutor(max_workers=min(args.num_workers, len(pending_pairs) or 1)) as executor,
    ):
        futures = {
            executor.submit(
                process_one_item,
                item,
                index=index,
                client=client,
                judge=judge,
                html_judge=html_judge,
                html_renderer=html_renderer,
                media_dir=media_dir,
                query_field=args.query_field,
                answer_field=args.answer_field,
                media_field=args.media_field,
                k=args.k,
                accuracy_backend=args.accuracy_backend,
                tolerance=args.tolerance,
                match_mode=args.match_mode,
                judge_retries=args.judge_retries,
                html_score_threshold=args.html_score_threshold,
                html_render_dir=html_render_dir,
                dataset_stem=dataset_path.stem,
                keep_rendered_images=args.keep_rendered_images,
                html_default_viewport=html_default_viewport,
                html_min_tag_ratio=args.html_min_tag_ratio,
                html_min_structural_jaccard=args.html_min_structural_jaccard,
                html_min_text_recall=args.html_min_text_recall,
                html_require_css=not args.html_no_require_css,
                html_require_rick_image=not args.html_no_require_rick_image,
            ): index
            for index, item in pending_pairs
        }

        with tqdm(total=len(futures), desc="Filtering difficulty", unit="item") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += 1
                    mean_sum += result["avg_k"]
                    kept = result["avg_k"] < args.threshold
                    if kept:
                        output_file.write(
                            json.dumps(
                                build_output_payload(result, keep_metadata=args.keep_metadata),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        output_file.flush()
                        kept_count += 1
                    progress_file.write(
                        json.dumps(
                            {
                                "key": result["key"],
                                "index": result["index"],
                                "id": result["id"],
                                "avg_k": result["avg_k"],
                                "kept": kept,
                                **(
                                    {"score_avg": round(sum(result["scores"]) / len(result["scores"]), 4)}
                                    if result.get("scores")
                                    else {}
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    progress_file.flush()
                except Exception as exc:
                    failed += 1
                    tqdm.write(f"[WARN] item #{futures[future]} failed: {exc}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix(kept=kept_count, failed=failed, refresh=False)

    mean_avg = mean_sum / processed_count if processed_count else 0.0
    print(f"\nTotal items: {len(items)}")
    print(f"Selected:    {len(selected_pairs)}")
    print(f"Resumed:     {skipped_count}")
    print(f"Processed:   {processed_count}")
    print(f"Failed:      {failed}")
    print(f"Mean avg@k:  {mean_avg:.4f}")
    print(f"Kept:        {kept_count} (avg@k < {args.threshold})")
    print(f"Output:      {output_path}")
    print(f"Progress:    {progress_path}")


if __name__ == "__main__":
    main()
