"""Config loader.

Two layers (both pure JSON, secrets included):
  - general_config.json: shared defaults (workspace, retry, judge, extractor, ...)
  - <model>_config.json:  per-tested-model params (model name, endpoint, sampling)

Each LLM endpoint spec must contain `model`, `base_url`, `api_key` directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .image import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS


DEFAULT_GENERAL_CONFIG = "configs/general_config.json"


# Supported eval modes. "judge" = LLM extractor + majority-vote LLM judge
# (chart/QA-style benches). "gui" = GUI grounding via point-in-bbox: extract
# the click coordinate from the agent's `computer_use` tool_call and verify
# it lands inside the GT pixel bbox. No LLM extractor/judge is invoked.
EVAL_MODES = ("judge", "gui")

# Agent drivers. "function_call" = native OpenAI tool-calling loop (default).
# "thyme" / "deepeyesv2" = code-as-tool text-protocol baselines.
AGENT_MODES = ("function_call", "thyme", "deepeyesv2")


@dataclass
class Endpoint:
    """Fully-resolved LLM endpoint (model + base_url + api_key + sampling)."""
    model: str
    base_url: str
    api_key: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None

    def sampling_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            out["presence_penalty"] = self.presence_penalty
        extra_body: dict[str, Any] = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            # vLLM exposes this OpenAI-compatible extension via extra_body.
            extra_body["repetition_penalty"] = self.repetition_penalty
        if extra_body:
            out["extra_body"] = extra_body
        return out


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 30.0


@dataclass
class AppConfig:
    """Top-level config consumed by eval.py / rejudge.py."""
    general_path: str
    general: dict[str, Any]
    model_config_path: str | None
    model_raw: dict[str, Any] | None

    workspace_root: str = "workspace"
    bench_path: str = "dataset/table/chart_tool_benchmark_union.jsonl"
    workers: int = 8
    max_steps: int = 50
    exec_timeout: int = 60
    tools_enabled: bool = True
    # Optional allowlist of tool names. When set, build_registry registers
    # ONLY these workspace/exec tools (computer_use is governed separately by
    # eval_mode). None = register all tools (default).
    allowed_tools: list[str] | None = None
    request_timeout_s: float = 90.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    # Number of independent rollouts per task. avg_k=1 reproduces the
    # previous single-sample acc; avg_k>1 enables the avg@k metric — for
    # each task we run agent+extract+judge k times, score_avg is the mean
    # of the per-sample 0/1 LLM-judge verdicts, and the run-level metric
    # accuracy_avg_k is the mean of score_avg across all tasks.
    avg_k: int = 1

    # "judge" = LLM extractor + LLM judge (chart/QA benches).
    # "gui"   = GUI grounding (point-in-bbox on `computer_use` args).
    eval_mode: str = "judge"

    # Agent driver. "function_call" = native OpenAI tool-calling loop
    # (utils/agent.run, the default). "thyme" / "deepeyesv2" = code-as-tool
    # text-protocol baselines (utils/code_agent.run_code_agent): the model
    # emits Python in <code> blocks executed by a sandbox kernel. Resolved
    # from the model config (preferred) or general config.
    agent_mode: str = "function_call"

    # GUI grounding (ScreenSpot-Pro) coordinate scaling for Qwen2.5-VL-style
    # baselines: their pyautogui click is in absolute pixels of the smart-resized
    # input space, so scoring rescales by the resize ratio. Defaults match
    # vLLM's Qwen2.5-VL serving defaults.
    grounding_max_pixels: int = DEFAULT_MAX_PIXELS
    grounding_min_pixels: int = DEFAULT_MIN_PIXELS

    extractor_enabled: bool = True
    _judge_raw: dict[str, Any] = field(default_factory=dict)
    _extractor_raw: dict[str, Any] = field(default_factory=dict)

    # Resolved endpoints
    test: Endpoint | None = None
    judge: Endpoint | None = None
    extractor: Endpoint | None = None

    def model_dir_name(self) -> str:
        """Directory-safe label used for `workspace/<name>/run_*/` layout."""
        if self.model_raw and self.model_raw.get("name"):
            return self.model_raw["name"]
        if self.test:
            return self.test.model.replace("/", "_").replace(":", "_")
        return "unknown"

    def _model_name(self) -> str:
        return str((self.model_raw or {}).get("name", "")).lower()

    def is_qwen25vl(self) -> bool:
        """Detect the Qwen2.5-VL family from the model config's `name`.

        Qwen2.5-VL emits ABSOLUTE pixel coordinates in its smart-resized input
        space (scored via score_grounding_pyautogui), unlike the 0-1000
        computer_use convention. Both eval.py and rejudge.py key the grounding
        scorer choice off this so they stay consistent.
        """
        name = self._model_name()
        return "qwen25vl" in name or "qwen2.5-vl" in name

    def is_qwen3vl(self) -> bool:
        """Detect Qwen3-VL from the model config's `name` field."""
        name = self._model_name()
        return "qwen3vl" in name or "qwen3-vl" in name

    def is_code_agent(self) -> bool:
        """True for the code-as-tool baselines (Thyme / DeepEyesV2)."""
        return self.agent_mode in ("thyme", "deepeyesv2")


def _resolve_endpoint(spec: dict[str, Any],
                      fallback_temperature: float | None = None,
                      label: str = "endpoint") -> Endpoint:
    """Build an Endpoint from a JSON spec.

    Required keys: model, base_url, api_key.
    """
    for key in ("model", "base_url", "api_key"):
        if not spec.get(key):
            raise RuntimeError(f"{label} config missing {key!r}")
    return Endpoint(
        model=spec["model"],
        base_url=spec["base_url"],
        api_key=spec["api_key"],
        temperature=spec.get("temperature", fallback_temperature),
        top_p=spec.get("top_p"),
        top_k=spec.get("top_k"),
        min_p=spec.get("min_p"),
        max_tokens=spec.get("max_tokens"),
        presence_penalty=spec.get("presence_penalty"),
        repetition_penalty=spec.get("repetition_penalty"),
    )


def load_config(general_path: str = DEFAULT_GENERAL_CONFIG,
                model_config_path: str | None = None,
                require_test: bool = True,
                general_overrides: dict[str, Any] | None = None) -> AppConfig:
    general = json.loads(Path(general_path).read_text())
    if general_overrides:
        general.update(general_overrides)
    model_raw = json.loads(Path(model_config_path).read_text()) if model_config_path else None

    eval_mode = str(general.get("eval_mode", "judge")).lower()
    if eval_mode not in EVAL_MODES:
        raise RuntimeError(
            f"eval_mode must be one of {EVAL_MODES}, got {eval_mode!r}"
        )

    # agent_mode lives on the model config (it's a property of the checkpoint),
    # falling back to general config, then the native function-calling default.
    agent_mode = str(
        (model_raw or {}).get("agent_mode")
        or general.get("agent_mode")
        or "function_call"
    ).lower()
    if agent_mode not in AGENT_MODES:
        raise RuntimeError(
            f"agent_mode must be one of {AGENT_MODES}, got {agent_mode!r}"
        )

    retry_raw = general.get("retry", {}) or {}
    cfg = AppConfig(
        general_path=general_path,
        general=general,
        model_config_path=model_config_path,
        model_raw=model_raw,
        workspace_root=general.get("workspace_root", "workspace"),
        bench_path=general.get("bench_path", "dataset/table/chart_tool_benchmark_union.jsonl"),
        workers=general.get("workers", 8),
        max_steps=general.get("max_steps", 50),
        exec_timeout=general.get("exec_timeout", 60),
        tools_enabled=bool(general.get("tools_enabled", True)),
        allowed_tools=general.get("allowed_tools"),
        request_timeout_s=float(general.get("request_timeout_s", 90)),
        avg_k=max(1, int(general.get("avg_k", 1))),
        eval_mode=eval_mode,
        agent_mode=agent_mode,
        grounding_max_pixels=int(general.get("grounding_max_pixels", DEFAULT_MAX_PIXELS)),
        grounding_min_pixels=int(general.get("grounding_min_pixels", DEFAULT_MIN_PIXELS)),
        retry=RetryConfig(
            max_attempts=retry_raw.get("max_attempts", 3),
            base_delay=retry_raw.get("base_delay", 2.0),
            max_delay=retry_raw.get("max_delay", 30.0),
        ),
        extractor_enabled=bool(general.get("extractor", {}).get("enabled", True)),
        _judge_raw=general.get("judge", {}) or {},
        _extractor_raw=general.get("extractor", {}) or {},
    )

    if model_raw is not None:
        cfg.test = _resolve_endpoint(model_raw, label="test model")
    elif require_test:
        raise RuntimeError(
            "no model config provided. Pass --model-config configs/<name>_config.json"
        )
    else:
        cfg.test = None

    cfg.judge = (
        _resolve_endpoint(cfg._judge_raw, label="judge")
        if cfg.eval_mode == "judge" and cfg._judge_raw else None
    )
    cfg.extractor = (
        _resolve_endpoint(cfg._extractor_raw, label="extractor")
        if cfg.eval_mode == "judge" and cfg.extractor_enabled else None
    )
    return cfg


def summary(cfg: AppConfig) -> str:
    lines = [
        f"general config : {cfg.general_path}",
        f"model config   : {cfg.model_config_path or '(none)'}",
        f"bench          : {cfg.bench_path}",
        f"workspace root : {cfg.workspace_root}",
        f"model dir      : {cfg.model_dir_name()}",
        f"workers        : {cfg.workers}",
        f"max_steps      : {cfg.max_steps}",
        f"exec_timeout   : {cfg.exec_timeout}s",
        f"req timeout    : {cfg.request_timeout_s}s",
        f"tools          : {'ON' if cfg.tools_enabled else 'OFF'}",
        f"avg_k          : {cfg.avg_k}",
        f"eval_mode      : {cfg.eval_mode}",
        f"agent_mode     : {cfg.agent_mode}",
        f"retry          : {cfg.retry.max_attempts}x base={cfg.retry.base_delay}s max={cfg.retry.max_delay}s",
        f"test   model   : "
        + (f"{cfg.test.model}  @ {cfg.test.base_url}" if cfg.test else "(not loaded)"),
        f"judge  model   : "
        + (f"{cfg.judge.model}  @ {cfg.judge.base_url}" if cfg.judge else "(disabled — gui mode)"),
        f"extractor      : "
        + (f"{cfg.extractor.model}  @ {cfg.extractor.base_url}" if cfg.extractor else "(disabled)"),
    ]
    return "\n".join(lines)
