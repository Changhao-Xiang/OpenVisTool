"""Eval entry point.

CLI args + per-task orchestration. All reusable helpers live under harness/.

Modes:
  - default:     fresh run -> workspace/<model>/run_<timestamp>/
  - --resume R:  resume an existing run at R. Skips tasks whose result.json
                 has score set and `error` null; re-runs anything that's
                 missing or has `error != null` (timeouts, agent crashes,
                 partial runs from a Ctrl-C). Re-aggregates results.jsonl
                 + meta.json across all tasks afterward.
"""

import argparse
import asyncio
import copy
import datetime as dt
import glob
import json
import sys
import time
from pathlib import Path

from harness.agents import (
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS,
    SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL,
    SYSTEM_PROMPT_VISION2WEB_WEBPAGE,
    build_gui_grounding_coord_prompt,
    build_gui_grounding_prompt,
    resize_grounding_images,
    run,
    run_code_agent,
    run_qwen25_grounding,
)
from harness.core import (
    AppConfig,
    LLMClient,
    PathMapper,
    build_content,
    load_benchmark,
    load_config,
    strip_base64,
    task_key,
)
from harness.core import (
    summary as config_summary,
)
from harness.runtime import (
    EvalDashboard,
    aggregate_results,
    build_registry,
    print_summary,
    setup_sample_sandbox,
    setup_sandbox,
)
from harness.scoring import (
    Extractor,
    Judge,
    is_grounding_item,
    is_vision2web_item,
    judge_vision2web_viewports,
    render_html_to_viewport_images,
    resolve_submitted_html,
    score_grounding,
    score_grounding_pyautogui,
    vision2web_ref_images,
    vision2web_viewports,
)
from harness.tools.code_sandbox import CodeSandboxTool


async def _run_one_sample(item: dict, sample_dir: Path,
                          cfg: AppConfig,
                          llm: LLMClient, judge: Judge | None, extractor: Extractor | None,
                          sample_images: list[str],
                          on_step=None) -> dict:
    """Run a single rollout (agent → extractor → judge) inside `sample_dir`.

    Returns the per-sample record dict. Persists sample-level trace.jsonl
    alongside it; the caller owns aggregation into result.json.
    """
    t0 = time.time()
    # Code-as-tool baselines (Thyme / DeepEyesV2) use a separate text-protocol
    # driver instead of the native function-calling registry.
    is_code_agent = cfg.agent_mode in ("thyme", "deepeyesv2")
    is_grounding = cfg.eval_mode == "gui" and is_grounding_item(item)
    # Qwen2.5-VL family (native + the Thyme/DeepEyesV2 code agents) emit ABSOLUTE
    # pixel coordinates in the smart-resized input space and commit a bare
    # `(x, y)` point, so they bypass the 0-1000 `computer_use` channel.
    is_qwen25 = is_code_agent or cfg.is_qwen25vl()
    # Base Qwen2.5-VL grounding runs the official single-shot computer_use
    # recipe (no agent loop, no inspection tools), so it skips the registry.
    is_qwen25_singleshot = (is_grounding and cfg.is_qwen25vl()
                            and not is_code_agent)
    include_cu = cfg.eval_mode == "gui" and not is_qwen25
    registry = (
        build_registry(sample_dir, cfg.exec_timeout,
                       include_computer_use=include_cu,
                       allowed_tools=cfg.allowed_tools)
        if cfg.tools_enabled and not is_code_agent and not is_qwen25_singleshot
        else None
    )
    path_mapper = PathMapper(sample_dir) if registry is not None else None

    # Determine system prompt override based on task type. The Vision2Web
    # code-gen task gets a code-generation prompt; GUI grounding has a stricter
    # completion protocol where only computer_use terminates.
    is_v2w_static = is_vision2web_item(item)

    sys_override: str | None = None
    if is_v2w_static:
        sys_override = SYSTEM_PROMPT_VISION2WEB_WEBPAGE
    elif cfg.eval_mode == "gui" and is_grounding_item(item):
        if cfg.is_qwen25vl():
            # Qwen2.5-VL commits a bare absolute-pixel (x, y) — no computer_use.
            # With a registry it may crop/zoom first; the (x, y) is its final
            # text answer either way, scored back via smart_resize.
            sys_override = build_gui_grounding_coord_prompt(
                cfg.allowed_tools if registry is not None else None)
        elif registry is not None:
            sys_override = build_gui_grounding_prompt(cfg.allowed_tools)
        else:
            # The model has no native function-calling tools available, so we
            # inline a computer_use schema and parse the click out of pred
            # text downstream. Qwen3-VL chat template expects a JSON object
            # directly inside <tool_call>, unlike Hermes-style XML.
            if cfg.is_qwen3vl():
                sys_override = SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS_QWEN3VL
            else:
                sys_override = SYSTEM_PROMPT_GUI_GROUNDING_NO_TOOLS

    meta: dict = {}
    error: str | None = None
    try:
        if is_code_agent:
            # For grounding, feed the code agent a pre-resized screenshot (same
            # smart_resize the scorer uses) so its absolute-pixel click is in the
            # resized space the scorer maps back from — correct even for the
            # screenshots the server's 12.8M cap would downscale.
            ca_images = (
                resize_grounding_images(sample_images, sample_dir,
                                        cfg.grounding_max_pixels,
                                        cfg.grounding_min_pixels)
                if is_grounding else sample_images
            )
            sandbox_tool = CodeSandboxTool(
                workspace=sample_dir, variant=cfg.agent_mode,
                image_paths=ca_images, exec_timeout=cfg.exec_timeout,
            )
            pred, meta = await run_code_agent(
                question=item["query"],
                image_paths=ca_images,
                llm=llm,
                sandbox=sandbox_tool,
                variant=cfg.agent_mode,
                max_steps=cfg.max_steps,
                on_step=on_step,
                grounding=is_grounding,
                # Align decoding with the official harness: grounding is greedy.
                greedy=is_grounding,
            )
        elif is_qwen25_singleshot:
            # Base Qwen2.5-VL: official ScreenSpot-Pro single-shot recipe
            # (computer_use prompt declaring the resized resolution + prefilled
            # left_click + greedy). The model emits an absolute-pixel (x, y) in
            # the resized space; scored back via smart_resize below.
            pred, meta = await run_qwen25_grounding(
                question=item["query"],
                image_path=sample_images[0],
                llm=llm,
                max_pixels=cfg.grounding_max_pixels,
                min_pixels=cfg.grounding_min_pixels,
                on_step=on_step,
            )
        else:
            pred, meta = await run(
                build_content(item,
                              script_image_paths=sample_images if registry else None,
                              workspace=sample_dir if registry else None),
                llm=llm,
                registry=registry,
                max_steps=cfg.max_steps,
                sandbox_dir=str(sample_dir),
                path_mapper=path_mapper,
                on_step=on_step,
                system_prompt=sys_override,
            )
    except Exception as e:
        pred = ""
        error = f"{type(e).__name__}: {e}"
        print(f"[id={item['id']}] agent error: {error}",
              file=sys.stderr, flush=True)

    pred = pred or ""

    # --- Scoring path selection ---
    # Binary 0/1 for QA + GUI grounding; continuous float (0-100) for the
    # Vision2Web code-gen benchmark.
    score_value: int | float = 0
    gui_extras = None
    vision2web_extras = None

    if is_v2w_static:
        # Vision2Web: resolve HTML → render per viewport → component judge.
        # Agentic openvistool models iterate via render_html and may end on a
        # plain-text turn; resolve_submitted_html falls back to the last HTML
        # file they wrote/edited.
        html_code, html_src = resolve_submitted_html(
            pred, meta.get("turns"), sample_dir, path_mapper)
        extracted = html_code

        try:
            generated = await render_html_to_viewport_images(
                html_code, sample_dir, vision2web_viewports(sample_dir))
        except Exception as e:
            generated = {}
            error = error or f"render error: {e}"
            print(f"[id={item['id']}] vision2web render error: {e}",
                  file=sys.stderr, flush=True)

        ref_images = vision2web_ref_images(item, sample_images)
        v2w = (await judge_vision2web_viewports(judge, ref_images, generated)
               if generated and ref_images and judge is not None else None)
        if v2w is not None:
            score_value = v2w["avg"]
            judge_raw = v2w["raw"]
        else:
            score_value = 0.0
            judge_raw = "<skipped: missing generated/ref images or judge>"
        vision2web_extras = {
            "scores": v2w["scores"] if v2w else {},
            "avg": score_value,
            "details": v2w["details"] if v2w else {},
            "html_source": html_src,  # "pred" | "file:<name>" | "none"
        }

    elif cfg.eval_mode == "gui" and is_grounding_item(item):
        # GUI grounding bypasses LLM extractor + judge: score the click against
        # the GT bbox geometrically. Qwen2.5-VL family (native + Thyme /
        # DeepEyesV2) commits a bare `(x, y)` in absolute pixels of the
        # smart-resized input space, scaled back via smart_resize; other models
        # (e.g. Qwen3-VL) emit a 0-1000 `computer_use` coordinate.
        if is_qwen25:
            gr = score_grounding_pyautogui(
                pred, item["answer"], item.get("img_size"),
                max_pixels=cfg.grounding_max_pixels,
                min_pixels=cfg.grounding_min_pixels)
        else:
            gr = score_grounding(meta.get("turns"), pred, item["answer"],
                                 item.get("img_size"))
        extracted = gr["extracted"]
        score_value = int(bool(gr["score"]))
        judge_raw = gr["judge_raw"]
        gui_extras = {
            "point_norm": gr.get("point"),
            "point_pixels": gr.get("point_pixels"),
            "bbox": gr.get("bbox"),
            "img_size": gr.get("img_size"),
            "click_source": gr.get("source"),
        }
    else:
        extracted = pred
        if extractor is not None:
            try:
                extracted = await extractor.extract(item["query"], pred)
            except Exception as e:
                print(f"[id={item['id']}] extract error: {e}",
                      file=sys.stderr, flush=True)
                extracted = pred

        # judge is None only in gui eval_mode; the else-branch is normally
        # unreachable there (gui items are grounding items), but guard anyway
        # so a stray non-grounding item in a gui run degrades cleanly instead
        # of raising AttributeError. Mirrors rejudge.py's explicit check.
        if judge is None:
            judge_raw = "<judge not configured>"
        else:
            try:
                correct, judge_raw = await judge.judge(item["query"], item["answer"], extracted)
            except Exception as e:
                correct, judge_raw = False, f"<judge error: {e}>"
                print(f"[id={item['id']}] judge error: {e}",
                      file=sys.stderr, flush=True)
            score_value = int(bool(correct))

    elapsed = round(time.time() - t0, 2)

    with open(sample_dir / "trace.jsonl", "w") as tf:
        tf.write(json.dumps(
            {"step": -1,
             "input_messages": strip_base64(meta.get("input_messages", []))},
            ensure_ascii=False) + "\n")
        for turn in strip_base64(meta.get("turns", [])):
            tf.write(json.dumps(turn, ensure_ascii=False) + "\n")

    return {
        "pred": pred,
        "extracted": extracted,
        "score": score_value,
        "judge_raw": judge_raw,
        "error": error,
        "usage_total": meta.get("usage_total"),
        "n_steps": meta.get("n_steps", 0),
        "n_tool_calls": meta.get("n_tool_calls", 0),
        "stopped_reason": meta.get("stopped_reason"),
        "sandbox": str(sample_dir),
        "elapsed_s": elapsed,
        **({"gui": gui_extras} if gui_extras is not None else {}),
        **({"vision2web": vision2web_extras}
           if vision2web_extras is not None else {}),
    }


async def run_one(item: dict, run_dir: Path,
                  cfg: AppConfig,
                  llm: LLMClient, judge: Judge | None, extractor: Extractor | None,
                  semaphore: asyncio.Semaphore,
                  dashboard: EvalDashboard | None = None) -> dict:
    """Run a single benchmark item end-to-end for avg@k samples (k = cfg.avg_k).

    Each sample runs the full agent → extractor → judge pipeline independently.
    When k == 1 we preserve the legacy layout: the agent runs directly in
    `task_<id>/` (no sample_<j>/ subdir) so existing resume/rejudge tooling is
    unaffected. When k > 1, each rollout gets its own `task_<id>/sample_<j>/`.

    The top-level `task_<id>/result.json` carries:
      - the primary-sample fields (pred/extracted/score/...) for back-compat
      - `score_avg`  : mean of per-sample scores in [0, 1]
      - `avg_k`      : number of samples actually attempted
      - `samples`    : list of per-sample records (present iff avg_k > 1)
    """
    async with semaphore:
        t0 = time.time()
        key = task_key(item)
        task_sandbox, task_images = setup_sandbox(run_dir, item)

        if dashboard:
            dashboard.start(key)

        k = max(1, cfg.avg_k)
        samples: list[dict] = []
        for j in range(k):
            if k == 1:
                sample_dir, sample_images = task_sandbox, task_images
            else:
                sample_dir, sample_images = setup_sample_sandbox(
                    task_sandbox, item, j)

            # Live progress: dashboard tracks the LATEST sample's rounds/tools.
            on_step = (
                (lambda rounds, n_tools, _key=key:
                    dashboard.update(_key, rounds=rounds, tools=n_tools))
                if dashboard else None
            )

            rec = await _run_one_sample(
                item, sample_dir, cfg, llm, judge, extractor,
                sample_images, on_step=on_step,
            )
            samples.append(rec)

            if dashboard and k > 1:
                # Partial progress: show avg-so-far as we go.
                running_avg = sum(s["score"] for s in samples) / len(samples)
                dashboard.update(key,
                                 score_avg=running_avg,
                                 samples_done=len(samples))

        # Aggregate across k samples. The "primary" sample for back-compat
        # fields is the first one (sample_0).
        primary = samples[0]
        score_avg = round(sum(s["score"] for s in samples) / len(samples), 4)
        any_error = next((s["error"] for s in samples if s["error"]), None)
        total_tokens = {
            "prompt_tokens": sum(
                (s.get("usage_total") or {}).get("prompt_tokens", 0) or 0
                for s in samples),
            "completion_tokens": sum(
                (s.get("usage_total") or {}).get("completion_tokens", 0) or 0
                for s in samples),
            "total_tokens": sum(
                (s.get("usage_total") or {}).get("total_tokens", 0) or 0
                for s in samples),
        }

        elapsed = round(time.time() - t0, 2)
        result: dict = {
            "id": item["id"],
            "source_dataset": item.get("source_dataset"),
            "task_key": key,
            "gt": item["answer"],
            # Back-compat: top-level fields mirror the first sample so legacy
            # consumers (rejudge.py, results aggregation, etc.) keep working.
            "pred": primary["pred"],
            "extracted": primary["extracted"],
            "score": primary["score"],
            "judge_raw": primary["judge_raw"],
            "error": any_error,
            "usage_total": total_tokens if k > 1 else primary.get("usage_total"),
            "n_steps": primary.get("n_steps", 0),
            "n_tool_calls": primary.get("n_tool_calls", 0),
            "stopped_reason": primary.get("stopped_reason"),
            "sandbox": str(task_sandbox),
            "elapsed_s": elapsed,
            # avg@k additions.
            "avg_k": k,
            "score_avg": score_avg,
        }
        if "gui" in primary:
            result["gui"] = primary["gui"]
        if "vision2web" in primary:
            result["vision2web"] = primary["vision2web"]
        if k > 1:
            result["samples"] = samples

        (task_sandbox / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))

        if dashboard:
            dashboard.finish(key,
                             score=primary["score"],
                             score_avg=score_avg,
                             tools=result["n_tool_calls"],
                             rounds=result["n_steps"],
                             elapsed_s=elapsed,
                             failed=any_error is not None)
        return result


def build_clients(cfg: AppConfig) -> tuple[LLMClient, Judge | None, Extractor | None]:
    """Create the agent / judge / extractor LLM clients with shared pool size.
    Public so `rejudge.py` can reuse it.

    In `eval_mode == "gui"` the judge + extractor are not used (point-in-bbox
    eval is purely geometric), so we skip building them.
    """
    pool = max(8, cfg.workers * 4)
    llm = LLMClient(cfg.test, retry=cfg.retry, label="agent",
                    request_timeout_s=cfg.request_timeout_s,
                    max_connections=pool)
    if cfg.eval_mode != "judge":
        return llm, None, None
    judge_raw_cfg = cfg.general.get("judge", {})
    judge = Judge(cfg.judge, retry=cfg.retry,
                  request_timeout_s=cfg.request_timeout_s,
                  max_connections=pool,
                  n_votes=int(judge_raw_cfg.get("n_votes", 3)),
                  vote_temperature=float(judge_raw_cfg.get("vote_temperature", 0.6)))
    extractor = (Extractor(cfg.extractor, retry=cfg.retry,
                           request_timeout_s=cfg.request_timeout_s,
                           max_connections=pool)
                 if cfg.extractor else None)
    return llm, judge, extractor


async def close_clients(llm: LLMClient, judge: Judge | None,
                        extractor: Extractor | None) -> None:
    await llm.close()
    if judge is not None and hasattr(judge, "_llm"):
        await judge._llm.close()
    if extractor is not None and hasattr(extractor, "_llm"):
        await extractor._llm.close()


def freeze_configs(run_dir: Path, cfg: AppConfig) -> tuple[Path, Path | None]:
    """Snapshot general_config.json + model_config.json into the run dir.

    Subsequent --resume calls load these snapshots so that the run is
    insulated from later config edits. Returns (general_snapshot,
    model_snapshot or None).

    The general snapshot is the *effective* config, not a verbatim copy of
    cfg.general_path: CLI overrides (--bench / --eval-mode / --workers /
    --max-steps / --avg-k / --no-tools) are folded back in so that --resume
    and --rejudge see the bench/eval_mode this run actually used. (Without
    this, e.g. multi_eval.sh runs every bench against the same on-disk
    general_config and all run dirs froze an identical, wrong snapshot.)
    """
    g_dst = run_dir / "general_config.json"
    frozen = copy.deepcopy(cfg.general)
    frozen.update(
        bench_path=cfg.bench_path,
        eval_mode=cfg.eval_mode,
        workers=cfg.workers,
        max_steps=cfg.max_steps,
        avg_k=cfg.avg_k,
        tools_enabled=cfg.tools_enabled,
        dashboard_enabled=cfg.dashboard_enabled,
    )

    # Never persist credentials or private endpoint details in a run. Resume
    # expands these placeholders from the current environment.
    for section_name in ("judge", "extractor"):
        section = frozen.get(section_name)
        if not isinstance(section, dict):
            continue
        if section.get("model"):
            section["model"] = "${JUDGE_MODEL}"
        if section.get("base_url"):
            section["base_url"] = "${JUDGE_BASE_URL}"
        if section.get("api_key"):
            section["api_key"] = "${JUDGE_API_KEY}"

    g_dst.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    m_dst = None
    if cfg.model_config_path and cfg.model_raw:
        m_dst = run_dir / "model_config.json"
        model_frozen = copy.deepcopy(cfg.model_raw)
        model_frozen["model"] = "${TEST_MODEL_ID}"
        model_frozen["base_url"] = "${TEST_MODEL_BASE_URL}"
        model_frozen["api_key"] = "${TEST_MODEL_API_KEY}"
        m_dst.write_text(json.dumps(model_frozen, ensure_ascii=False, indent=2))
    return g_dst, m_dst


def write_meta(run_dir: Path, cfg: AppConfig, n: int) -> dict:
    """Initial meta.json — only the run-time facts that aren't already in
    the frozen general_config.json / model_config.json snapshots living
    next to it. Run-completion stats are merged in by `update_meta`.
    """
    meta = {
        "start_time": dt.datetime.now().isoformat(timespec="seconds"),
        # Where the snapshots were copied from. The actual config values
        # live in `<run_dir>/general_config.json` and model_config.json.
        "general_config_origin": cfg.general_path,
        "model_config_origin": cfg.model_config_path,
        # `n` may be smaller than the bench size when -n was passed.
        "n": n,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def update_meta(run_dir: Path, agg: dict, total_elapsed: float) -> None:
    """Merge end-of-run aggregates into meta.json (preserving original
    start_time, model info, etc)."""
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    meta.update({
        "end_time": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(total_elapsed, 2),
        "total": agg["total"],
        "correct": agg["correct"],
        "accuracy": agg["accuracy"],
        "avg_k": agg.get("avg_k", 1),
        "accuracy_avg_k": agg.get("accuracy_avg_k", agg["accuracy"]),
        "errors": agg["errors"],
        "timeouts": agg["timeouts"],
        "total_tokens": agg["total_tokens"],
        "avg_tool_calls": agg.get("avg_tool_calls", 0),
        "error_ids": agg["error_ids"],
        "timeout_ids": agg["timeout_ids"],
    })
    if agg.get("avg_score") is not None:
        meta["avg_score"] = agg["avg_score"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _completed_keys(run_dir: Path) -> set[str]:
    """Task keys whose result.json exists with `error` null — they're done.
    Anything else (missing result.json, or error set) needs (re-)running."""
    done: set[str] = set()
    for f in glob.glob(str(run_dir / "task_*" / "result.json")):
        try:
            r = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            continue
        if not r.get("error"):
            done.add(r.get("task_key") or Path(f).parent.name.removeprefix("task_"))
    return done


async def evaluate(cfg: AppConfig, n: int | None = None,
                   out_dir: str | None = None,
                   resume_dir: str | None = None) -> float:
    all_items = load_benchmark(cfg.bench_path)
    if n:
        all_items = all_items[:n]

    if resume_dir is not None:
        run_dir = Path(resume_dir).resolve()
        if not run_dir.exists():
            raise SystemExit(f"resume dir not found: {run_dir}")
        done_keys = _completed_keys(run_dir)
        items = [it for it in all_items if task_key(it) not in done_keys]
        # Only create meta.json if it's missing (recovery from an interrupted
        # very-early run). Snapshots are already present (we read them above
        # in main() to load cfg), so do NOT re-freeze — that would self-copy
        # and raise SameFileError.
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            write_meta(run_dir, cfg, len(all_items))
        skipped_n = len(all_items) - len(items)
    else:
        if out_dir is None:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(cfg.workspace_root) / cfg.model_dir_name() / f"run_{stamp}"
        else:
            run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Freeze the configs into the run dir BEFORE write_meta so meta.json's
        # `general_config` / `model_config` fields point at the snapshots.
        freeze_configs(run_dir, cfg)
        write_meta(run_dir, cfg, len(all_items))
        items = all_items
        skipped_n = 0

    print(config_summary(cfg))
    if resume_dir is not None:
        print(f"\n▶ resume: {run_dir}")
        print(f"  bench total: {len(all_items)}  already done: {skipped_n}  to run: {len(items)}\n")
    else:
        print(f"\n▶ n={len(items)}\n▶ run dir -> {run_dir}\n")

    if not items:
        # Nothing to do — just re-aggregate and print.
        agg = aggregate_results(run_dir)
        print_summary(agg)
        update_meta(run_dir, agg, 0.0)
        return agg["accuracy"]

    llm, judge, extractor = build_clients(cfg)

    sem = asyncio.Semaphore(cfg.workers)
    t_all = time.time()
    with EvalDashboard(items, enabled=sys.stdout.isatty(),
                       details_enabled=cfg.dashboard_enabled) as dash:
        tasks = [run_one(item, run_dir, cfg, llm, judge, extractor, sem, dashboard=dash)
                 for item in items]
        await asyncio.gather(*tasks, return_exceptions=False)
    total_elapsed = time.time() - t_all

    # Re-aggregate from disk so resumed runs include the previously-done tasks.
    agg = aggregate_results(run_dir)
    print_summary(agg, total_elapsed)
    update_meta(run_dir, agg, total_elapsed)

    if resume_dir is not None:
        meta_path = run_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            meta = {}
        meta.setdefault("resumes", []).append({
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "ran_tasks": sorted(task_key(it) for it in items),
            "elapsed_s": round(total_elapsed, 2),
            "post_correct": agg["correct"],
            "post_errors": agg["errors"],
            "post_timeouts": agg["timeouts"],
        })
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    await close_clients(llm, judge, extractor)
    return agg["accuracy"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--general-config", default="configs/general_config.json")
    p.add_argument("--model-config", default=None,
                   help="Per-model JSON config. With --resume, defaults to "
                        "the original run's `model_config` from meta.json.")
    p.add_argument("--bench", default=None, help="Override bench path")
    p.add_argument("-n", type=int, default=None, help="Limit to first N items")
    p.add_argument("--workers", type=int, default=None, help="Override workers")
    p.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--avg-k", type=int, default=None,
                   help="Override avg_k: run each task k times and compute "
                        "avg@k in addition to plain accuracy.")
    p.add_argument("--eval-mode", default=None, choices=("judge", "gui"),
                   help="Override eval_mode from general config (judge | gui).")
    p.add_argument("--out-dir", default=None, help="Explicit run dir")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="Resume an existing run. Skips tasks with score and "
                        "no error; re-runs anything with `error != null` or "
                        "missing result.json.")
    tools_grp = p.add_mutually_exclusive_group()
    tools_grp.add_argument("--tools", dest="tools", action="store_true", default=None,
                           help="Force tools ON (overrides general config)")
    tools_grp.add_argument("--no-tools", dest="tools", action="store_false", default=None,
                           help="Force tools OFF (overrides general config)")
    dashboard_grp = p.add_mutually_exclusive_group()
    dashboard_grp.add_argument("--dashboard", dest="dashboard", action="store_true", default=None,
                               help="Show the live per-task dashboard table")
    dashboard_grp.add_argument("--no-dashboard", dest="dashboard", action="store_false", default=None,
                               help="Hide per-task dashboard details; keep only the live aggregate avg@k/progress line")
    return p.parse_args()


def _resume_config_paths(run_dir: Path) -> tuple[str, str | None]:
    """Return (general_config_path, model_config_path) for a resume.

    Reads the snapshot files at <run_dir>/general_config.json (required)
    and <run_dir>/model_config.json (optional) that were frozen at the
    original run's start.
    """
    g = run_dir / "general_config.json"
    if not g.exists():
        raise SystemExit(
            f"resume dir {run_dir} has no general_config.json snapshot — "
            f"cannot determine config to use")
    m = run_dir / "model_config.json"
    return str(g), str(m) if m.exists() else None


def main() -> None:
    args = _parse_args()

    if args.resume:
        # Resume mode: snapshot configs in the run dir are authoritative;
        # ignore --general-config / --model-config if they were also passed.
        run_dir = Path(args.resume)
        general_path, model_path = _resume_config_paths(run_dir)
        if args.general_config != "configs/general_config.json" or args.model_config:
            print("note: --general-config / --model-config ignored in resume mode; "
                  f"using snapshots from {run_dir}", file=sys.stderr)
    else:
        general_path = args.general_config
        model_path = args.model_config

    overrides: dict = {}
    if args.eval_mode is not None:
        overrides["eval_mode"] = args.eval_mode
    cfg = load_config(general_path, model_path, general_overrides=overrides or None)
    if args.bench:
        cfg.bench_path = args.bench
    if args.workers:
        cfg.workers = args.workers
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.avg_k is not None:
        cfg.avg_k = max(1, args.avg_k)
    if args.tools is not None:
        cfg.tools_enabled = args.tools
    if args.dashboard is not None:
        cfg.dashboard_enabled = args.dashboard

    asyncio.run(evaluate(cfg, n=args.n, out_dir=args.out_dir,
                         resume_dir=args.resume))


if __name__ == "__main__":
    main()
