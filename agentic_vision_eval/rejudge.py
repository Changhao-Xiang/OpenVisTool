"""Re-run extractor + judge (or the geometric grounding / Vision2Web scorers)
on an existing eval run, without re-calling the test model.

Reads workspace/<run>/task_*/result.json, re-scores the stored `pred`, and
rewrites each task's result.json plus the aggregate results.jsonl + meta.json.
Old values are preserved as `score_prev` / `extracted_prev` / `vision2web_prev`.

Config resolution (important): the run-defining fields — bench_path, eval_mode,
avg_k, grounding pixel budgets, and the test model family — are read from the
FROZEN snapshots in the run dir (general_config.json / model_config.json), so the
right scorer and benchmark are always used even if the live configs drifted. The
judge / extractor ENDPOINTS come from the live `--general-config` (default
configs/general_config.json) — that's the point of rejudge: re-score with a new
or edited judge.

Usage:
    uv run python rejudge.py workspace/run_20260424_105335
    uv run python rejudge.py workspace/run_... --workers 16
    uv run python rejudge.py workspace/run_... --rerender   # Vision2Web only
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

from harness.core import PathMapper, load_benchmark, load_config, task_key
from harness.scoring import (
    DEVICES, Extractor, Judge,
    is_grounding_item, is_vision2web_source,
    judge_vision2web_viewports, render_html_to_viewport_images,
    resolve_submitted_html, score_grounding, score_grounding_pyautogui,
    vision2web_viewports,
)
from harness.runtime import aggregate_results


def _load_turns(trace_dir: Path) -> list[dict]:
    """Read agent turns from <trace_dir>/trace.jsonl (skips the step=-1 header).
    Returns [] on any error."""
    path = trace_dir / "trace.jsonl"
    if not path.exists():
        return []
    turns: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("step", 0) == -1:
                continue
            turns.append(obj)
    except (OSError, json.JSONDecodeError):
        return []
    return turns


def _resolve_config(run_dir: Path, cli_general_config: str):
    """Build the AppConfig for re-judging `run_dir`.

    - bench_path / eval_mode / avg_k / grounding budgets: from the frozen
      `run_dir/general_config.json` snapshot (what the run actually used).
    - judge / extractor endpoints + n_votes: from `cli_general_config` (live),
      so you can re-score with a different judge.
    - test model family (Qwen2.5-VL?): from the frozen `run_dir/model_config.json`
      snapshot, which selects the grounding scorer.

    Falls back to the live config alone for legacy runs lacking snapshots.
    """
    frozen_g = run_dir / "general_config.json"
    frozen_m = run_dir / "model_config.json"

    overrides: dict = {}
    if frozen_g.exists():
        try:
            snap = json.loads(frozen_g.read_text())
        except json.JSONDecodeError:
            snap = {}
        for key in ("bench_path", "eval_mode", "avg_k",
                    "grounding_max_pixels", "grounding_min_pixels"):
            if key in snap:
                overrides[key] = snap[key]

    return load_config(
        cli_general_config,
        model_config_path=str(frozen_m) if frozen_m.exists() else None,
        require_test=False,
        general_overrides=overrides or None,
    )


def _load_bench_index(bench_path: str) -> dict[str, dict]:
    """Map task_key (and unambiguous id) -> benchmark item dict.

    We need the full item (not just `query`) so the gui rejudge path
    can read `answer` + `img_size` to redo the point-in-bbox check.
    """
    items = load_benchmark(bench_path)
    idx: dict[str, dict] = {task_key(item): item for item in items}
    id_counts = Counter(item["id"] for item in items)
    for item in items:
        if id_counts[item["id"]] == 1:
            idx[str(item["id"])] = item
    return idx


async def _redo_vision2web_sample(sample_dir: Path, judge: Judge,
                                  pred: str | None = None,
                                  rerender: bool = False) -> dict | None:
    """Re-score one Vision2Web sample against the prototype refs (`<device>.jpg`).

    By default reuses the already-rendered viewport images (`generated_<device>.png`),
    since they are deterministic given the stored pred.

    With `rerender=True` and a `pred`, applies the agentic HTML fallback
    (`resolve_submitted_html`): if the final turn had no HTML code but the model
    wrote one via the file tools (write_file/edit_file/render_html), that file is
    re-rendered to the viewports (overwriting `generated_*.png` / `generated.html`)
    before judging. Samples whose pred already yielded HTML are left untouched.

    Returns {scores, avg, details, raw, [html_source]} or None when no viewport
    images exist.
    """
    html_source: str | None = None
    if rerender and pred is not None:
        turns = _load_turns(sample_dir)
        try:
            html, src = resolve_submitted_html(
                pred, turns, sample_dir, PathMapper(sample_dir))
        except Exception:
            html, src = "", "none"
        # Only re-render when the fallback recovered a file the final turn omitted;
        # "pred" means the stored render already reflects the submitted HTML.
        if src.startswith("file:"):
            try:
                await render_html_to_viewport_images(
                    html, sample_dir, vision2web_viewports(sample_dir))
                html_source = src
            except Exception as e:
                html_source = f"rerender-error:{str(e)[:80]}"

    refs = {d: str(sample_dir / f"{d}.jpg")
            for d in DEVICES if (sample_dir / f"{d}.jpg").exists()}
    gens = {d: str(sample_dir / f"generated_{d}.png")
            for d in DEVICES if (sample_dir / f"generated_{d}.png").exists()}

    res = await judge_vision2web_viewports(judge, refs, gens)
    if res is None:
        return None
    out = {"scores": res["scores"], "avg": res["avg"],
           "details": res["details"], "raw": res["raw"]}
    if html_source is not None:
        out["html_source"] = html_source
    return out


async def _rejudge_vision2web(task_dir: Path, rec: dict, result_path: Path,
                              judge: Judge | None, rerender: bool = False) -> dict:
    """Re-judge a Vision2Web record in place, preserving previous values.

    avg@k samples live in `sample_<j>/`; the top-level record mirrors the
    primary sample (sample 0), exactly as eval.py writes it. `score_avg` is the
    mean of the per-sample viewport averages.
    """
    tid = rec.get("id")
    key = rec.get("task_key") or task_dir.name.removeprefix("task_")
    if judge is None:
        return {"id": tid, "task_key": key, "skipped": True,
                "reason": "judge not configured"}

    def _apply(target: dict, res: dict) -> None:
        target["score_prev"] = target.get("score")
        target["score"] = res["avg"]
        target.setdefault("vision2web_prev", target.get("vision2web"))
        v2w = {"scores": res["scores"], "avg": res["avg"], "details": res["details"]}
        if "html_source" in res:
            v2w["html_source"] = res["html_source"]
        target["vision2web"] = v2w
        target["judge_raw"] = res["raw"]

    samples = rec.get("samples")
    if isinstance(samples, list) and samples:
        sample_avgs: list[float] = []
        for j, s in enumerate(samples):
            sample_dir = task_dir / f"sample_{j}"
            if not sample_dir.exists():
                sample_dir = task_dir
            res = await _redo_vision2web_sample(
                sample_dir, judge, pred=s.get("pred"), rerender=rerender)
            if res is None:
                s["rejudge_error"] = "no rendered viewport images"
                if s.get("score") is not None:
                    sample_avgs.append(s["score"])
                continue
            _apply(s, res)
            sample_avgs.append(res["avg"])

        # Top-level mirrors the primary sample (sample 0).
        primary = samples[0]
        rec["score_prev"] = rec.get("score")
        rec["score"] = primary.get("score")
        rec.setdefault("vision2web_prev", rec.get("vision2web"))
        rec["vision2web"] = primary.get("vision2web")
        rec["judge_raw"] = primary.get("judge_raw")
        if sample_avgs:
            rec["score_avg_prev"] = rec.get("score_avg")
            rec["score_avg"] = round(sum(sample_avgs) / len(sample_avgs), 4)
    else:
        # Single rollout: rendered images live at the task root.
        res = await _redo_vision2web_sample(
            task_dir, judge, pred=rec.get("pred"), rerender=rerender)
        if res is None:
            return {"id": tid, "task_key": key, "skipped": True,
                    "reason": "no rendered viewport images"}
        _apply(rec, res)
        rec["score_avg_prev"] = rec.get("score_avg")
        rec["score_avg"] = res["avg"]

    rec.setdefault("task_key", key)
    tmp = result_path.with_suffix(".json.new")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    tmp.replace(result_path)

    old = rec.get("score_avg_prev")
    new = rec.get("score_avg")
    n_s = len(samples) if isinstance(samples, list) and samples else 1
    old_str = f"{old:.2f}" if isinstance(old, (int, float)) else str(old)
    new_str = f"{new:.2f}" if isinstance(new, (int, float)) else str(new)
    print(f"task={str(key):<24} v2w score_avg: {old_str} → {new_str}  "
          f"(samples={n_s}, viewports={rec.get('vision2web', {}).get('scores')})",
          flush=True)
    return rec


async def _rejudge_one(task_dir: Path, bench_index: dict[str, dict],
                       judge: Judge | None, extractor: Extractor | None,
                       cfg,
                       sem: asyncio.Semaphore,
                       rerender: bool = False) -> dict:
    async with sem:
        result_path = task_dir / "result.json"
        if not result_path.exists():
            return {"id": None, "skipped": True, "reason": f"no result.json in {task_dir}"}

        rec = json.loads(result_path.read_text())
        tid = rec["id"]
        key = rec.get("task_key") or task_dir.name.removeprefix("task_")

        eval_mode = cfg.eval_mode

        # Vision2Web: continuous component-fidelity score over rendered viewport
        # images — bypass the generic extractor + YES/NO judge (whose `gt` is an
        # image path, not a text answer). No benchmark item lookup needed.
        if eval_mode != "gui" and is_vision2web_source(rec.get("source_dataset")):
            return await _rejudge_vision2web(task_dir, rec, result_path, judge, rerender)

        item = bench_index.get(str(key)) or bench_index.get(str(tid))
        if item is None:
            return {"id": tid, "task_key": key, "skipped": True,
                    "reason": "task key not in benchmark"}
        query = item["query"]

        # Read old score; tolerate legacy bool-`ok` and old-style `score_prev` records.
        old_score = rec.get("score")
        if old_score is None and "ok" in rec:
            old_score = int(bool(rec["ok"]))
        old_extracted = rec.get("extracted")
        old_score_avg = rec.get("score_avg")
        pred = rec.get("pred", "")

        async def _redo(pred_text: str, prev_extracted: str | None,
                        sample_dir: Path | None = None
                        ) -> tuple[str, int, str] | tuple[None, None, str]:
            """Run extract + judge (or point-in-bbox, for gui mode) for one pred;
            return (extracted, score, raw) or (None, None, error_reason) on failure.
            """
            if eval_mode == "gui" and is_grounding_item(item):
                # Match eval.py's scorer choice: the Qwen2.5-VL family (native +
                # Thyme/DeepEyesV2 code agents) commit an ABSOLUTE-pixel (x, y)
                # scored via smart_resize; other models emit 0-1000 computer_use.
                if cfg.is_qwen25vl() or cfg.is_code_agent():
                    gr = score_grounding_pyautogui(
                        pred_text, item["answer"], item.get("img_size"),
                        max_pixels=cfg.grounding_max_pixels,
                        min_pixels=cfg.grounding_min_pixels)
                else:
                    turns = _load_turns(sample_dir or task_dir)
                    gr = score_grounding(turns, pred_text, item["answer"],
                                         item.get("img_size"))
                return gr["extracted"], int(gr["score"]), gr["judge_raw"]

            if extractor is not None:
                try:
                    extracted = await extractor.extract(query, pred_text)
                except Exception as e:
                    return None, None, f"extract error: {e}"
            else:
                extracted = pred_text
            if judge is None:
                return None, None, "judge not configured"
            try:
                ok, raw = await judge.judge(query, rec["gt"], extracted)
            except Exception as e:
                return None, None, f"judge error: {e}"
            return extracted, int(bool(ok)), raw

        # --- top-level pred: always re-judged (back-compat) ---
        new_extracted, new_score, raw_or_err = await _redo(pred, old_extracted,
                                                           task_dir)
        if new_extracted is None:
            return {"id": tid, "skipped": True, "reason": raw_or_err}

        rec["score"] = new_score
        rec["score_prev"] = old_score
        rec["extracted"] = new_extracted
        if old_extracted is not None and old_extracted != new_extracted:
            rec["extracted_prev"] = old_extracted
        rec["judge_raw"] = raw_or_err
        rec.pop("ok", None)
        rec.pop("ok_prev", None)

        # --- avg@k samples: re-judge each individual sample if present ---
        samples = rec.get("samples")
        if isinstance(samples, list) and samples:
            for j, s in enumerate(samples):
                s_pred = s.get("pred", "")
                s_old_extracted = s.get("extracted")
                s_old_score = s.get("score")
                # Sample 0 mirrors the top-level pred — reuse the verdict to
                # save one extract+judge round-trip.
                if j == 0 and s_pred == pred:
                    s["score_prev"] = s_old_score
                    s["score"] = new_score
                    if s_old_extracted is not None and s_old_extracted != new_extracted:
                        s["extracted_prev"] = s_old_extracted
                    s["extracted"] = new_extracted
                    s["judge_raw"] = raw_or_err
                    continue

                # avg@k samples each live in their own sample_<j>/ subdir.
                sample_dir = task_dir / f"sample_{j}"
                s_ext, s_score, s_raw = await _redo(s_pred, s_old_extracted,
                                                    sample_dir)
                if s_ext is None:
                    # Keep the old verdict but flag the sample.
                    s["rejudge_error"] = s_raw
                    continue
                s["score_prev"] = s_old_score
                s["score"] = s_score
                if s_old_extracted is not None and s_old_extracted != s_ext:
                    s["extracted_prev"] = s_old_extracted
                s["extracted"] = s_ext
                s["judge_raw"] = s_raw

            # Recompute score_avg over all samples (use new score where present,
            # falling back to old per-sample score for any that errored).
            scores = [s.get("score") for s in samples if s.get("score") is not None]
            if scores:
                rec["score_avg_prev"] = old_score_avg
                rec["score_avg"] = round(sum(scores) / len(scores), 4)
        else:
            # avg_k==1 (no per-sample list): keep score_avg in sync with the
            # re-judged score, otherwise aggregate_results would compute
            # accuracy_avg_k from a stale score_avg that disagrees with accuracy.
            rec["score_avg_prev"] = old_score_avg
            rec["score_avg"] = float(new_score)

        # Atomic write (avoid truncating on crash).
        tmp = result_path.with_suffix(".json.new")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        tmp.replace(result_path)

        flipped = (old_score is not None and new_score != old_score)
        mark = "✓" if new_score else "✗"
        old_mark = ("✓" if old_score else "✗") if old_score is not None else "?"
        change = f"  (was {old_mark} → {mark})" if flipped else ""
        avg_change = ""
        if rec.get("avg_k", 1) > 1 and old_score_avg is not None:
            avg_change = f"  avg@k: {old_score_avg:.2f} → {rec['score_avg']:.2f}"
        raw_one_line = (raw_or_err or "").replace("\n", " ⏎ ")
        rec.setdefault("task_key", key)
        print(f"task={key:<18} {mark}{change}{avg_change}  pred={pred[:60]!r}\n"
              f"          gt={rec['gt']!r}\n"
              f"          extracted={new_extracted[:80]!r}\n"
              f"          judge={raw_one_line[:300]!r}", flush=True)
        return rec


async def rejudge(run_dir: str, general_config: str, workers_override: int | None,
                  rerender: bool = False):
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise SystemExit(f"run dir not found: {run_path}")

    task_dirs = sorted(
        [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda d: d.name,
    )
    if not task_dirs:
        raise SystemExit(f"no task_* subdirs found in {run_path}")

    if not os.path.exists(general_config):
        raise SystemExit(f"general config not found: {general_config}")

    # Run-defining fields from the frozen snapshot; judge/extractor from the live
    # --general-config; model family from the frozen model_config snapshot.
    cfg = _resolve_config(run_path, general_config)

    workers = workers_override or cfg.workers
    bench_index = _load_bench_index(cfg.bench_path)

    if cfg.eval_mode == "gui":
        judge = None
        extractor = None
    else:
        judge_raw_cfg = cfg.general.get("judge", {})
        judge = Judge(cfg.judge, retry=cfg.retry,
                      request_timeout_s=cfg.request_timeout_s,
                      max_connections=max(8, workers * 4),
                      n_votes=int(judge_raw_cfg.get("n_votes", 3)),
                      vote_temperature=float(judge_raw_cfg.get("vote_temperature", 0.6)))
        extractor = (Extractor(cfg.extractor, retry=cfg.retry,
                               request_timeout_s=cfg.request_timeout_s,
                               max_connections=max(8, workers * 4))
                     if cfg.extractor else None)

    snapshot = "frozen snapshot" if (run_path / "general_config.json").exists() else "live config (no snapshot)"
    print(f"▶ run dir:   {run_path}")
    print(f"▶ tasks:     {len(task_dirs)}")
    print(f"▶ eval_mode: {cfg.eval_mode}  (run fields from {snapshot})")
    print(f"▶ bench:     {cfg.bench_path}")
    if cfg.eval_mode == "gui":
        scorer = ("pyautogui/smart_resize (Qwen2.5-VL family)"
                  if cfg.is_qwen25vl() or cfg.is_code_agent()
                  else "computer_use 0-1000")
        print(f"▶ judge:     (gui mode — point-in-bbox, no LLM; scorer: {scorer})")
        print(f"▶ extractor: (gui mode — disabled)")
    else:
        print(f"▶ judge:     {cfg.judge.model}  @ {cfg.judge.base_url}")
        print(f"▶ extractor: "
              + (f"{cfg.extractor.model}  @ {cfg.extractor.base_url}"
                 if cfg.extractor else "(disabled)"))
    print(f"▶ workers:   {workers}\n")

    if rerender:
        print("▶ rerender:  on (Vision2Web — recover model's last-written HTML "
              "for samples whose final turn had no code, then re-render + re-judge)")
    sem = asyncio.Semaphore(workers)
    tasks = [_rejudge_one(d, bench_index, judge, extractor, cfg, sem, rerender)
             for d in task_dirs]

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    valid = [r for r in results if not r.get("skipped")]
    flipped = sum(1 for r in valid
                  if r.get("score_prev") is not None
                  and r.get("score") != r.get("score_prev"))

    # Re-aggregate from the freshly-written result.json files using the canonical
    # aggregator (handles continuous Vision2Web scores correctly: avg_score,
    # correct = #(score>=50), accuracy_avg_k = mean(score_avg)).
    agg = aggregate_results(run_path)

    # Refresh meta.json headline metrics, preserving the originals as *_prev.
    meta_path = run_path / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        for k in ("accuracy", "accuracy_avg_k", "avg_score", "correct"):
            if k in meta and f"{k}_prev" not in meta:
                meta[f"{k}_prev"] = meta[k]
        meta["accuracy"] = agg["accuracy"]
        meta["accuracy_avg_k"] = agg["accuracy_avg_k"]
        if "avg_score" in agg:
            meta["avg_score"] = agg["avg_score"]
        meta["correct"] = agg["correct"]
        meta["rejudged_with"] = general_config
        mtmp = meta_path.with_suffix(".json.new")
        mtmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        mtmp.replace(meta_path)

    skipped = [r for r in results if r.get("skipped")]
    if skipped:
        print("\nskipped:")
        for r in skipped:
            print(f"  task={r.get('task_key') or r.get('id')}: {r['reason']}")

    print(f"\n▶ re-judged: {len(valid)} tasks  flipped={flipped}  elapsed={elapsed:.1f}s")
    if agg.get("avg_score") is not None:
        print(f"▶ avg_score: {agg['avg_score']}/100  (correct≥50: {agg['correct']}/{agg['total']})")
    else:
        print(f"▶ accuracy:  {agg['correct']}/{agg['total']} = {agg['accuracy']:.3f}")
    if agg.get("avg_k", 1) > 1:
        print(f"▶ avg@{agg['avg_k']}:   {agg['accuracy_avg_k']:.4f}  "
              f"(mean of per-task score_avg over {agg['total']} tasks)")
    print(f"  updated  -> {agg['results_path']}")
    print(f"           -> meta.json (accuracy/accuracy_avg_k/avg_score refreshed, *_prev kept)")
    print(f"           -> task_<source_dataset>_<id>/result.json (score_prev / vision2web_prev kept)")


def _maybe_unset_proxies_for_internal():
    import re
    for key in ("JUDGE_BASE_URL", "TEST_BASE_URL", "OPENAI_BASE_URL"):
        url = os.environ.get(key, "")
        if re.search(r"//(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)", url):
            print(f"▶ detected internal endpoint ({url}) — unsetting proxies")
            for pk in ("http_proxy", "https_proxy", "all_proxy",
                       "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                os.environ.pop(pk, None)
            return


def main() -> None:
    """Console-script entry point. Registered in pyproject.toml as `rejudge`."""
    _maybe_unset_proxies_for_internal()
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="workspace/run_* directory to re-judge")
    p.add_argument("--general-config", default="configs/general_config.json",
                   help="Live general config supplying the judge/extractor "
                        "endpoints (run-defining fields come from the run's "
                        "frozen snapshot).")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--rerender", action="store_true",
                   help="Vision2Web: for samples whose final turn had no HTML, "
                        "recover the last HTML file the model wrote/edited and "
                        "re-render it before judging (agentic openvistool models). "
                        "Overwrites those samples' generated.html / generated_*.png.")
    args = p.parse_args()
    asyncio.run(rejudge(args.run_dir, args.general_config, args.workers, args.rerender))


if __name__ == "__main__":
    main()
