"""Code-as-tool sandbox for the DeepEyesV2 / Thyme baselines.

`CodeSandboxTool` runs a Python cell in an isolated subprocess (see
``harness/tools/code_kernel.py``) and returns the stdout/stderr plus any generated
images as OpenAI multimodal content blocks, so the agent loop can feed the
result straight back to the model. State persists across cells within a task
(picklable locals + re-imported modules), approximating a Jupyter kernel.

This tool is invoked by ``harness/agents/code_agent.py`` (the text-protocol loop) rather
than the native function-calling registry — the baselines emit code inside
``<code>...</code>`` text blocks, not OpenAI ``tool_calls``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from harness.tools.base import Tool
from harness.tools.filesystem import detect_image_mime
from harness.tools.shell import sandbox_subprocess_env

_KERNEL = str(Path(__file__).resolve().parent / "code_kernel.py")


class CodeSandboxTool(Tool):
    """Execute a Python cell in a per-task code sandbox.

    Args:
        workspace:    the task sandbox dir (cwd + image output dir).
        variant:      "thyme" or "deepeyesv2" — selects the kernel's namespace
                      setup (image_path/temp_output_dir vs. image_1..N).
        image_paths:  resolved absolute paths of the task's input images.
        exec_timeout: per-cell wall-clock limit (seconds).
        max_images:   cap on images returned to the model per cell.
    """

    def __init__(self, workspace: Path, variant: str,
                 image_paths: list[str] | None = None,
                 exec_timeout: int = 120, max_images: int = 6):
        self._workspace = Path(workspace).resolve()
        self._variant = variant
        # Resolve to absolute paths: the kernel subprocess runs with cwd set to
        # the sandbox, so relative image paths would not resolve there.
        self._image_paths = [str(Path(p).resolve()) for p in (image_paths or [])]
        self._timeout = exec_timeout
        self._max_images = max_images
        # Output dir for generated images + persisted kernel state. Distinct
        # subdir so it never collides with symlinked input images.
        self._out_dir = self._workspace / "_code_out"
        self._ctx_path = self._workspace / ".kernel_ctx.pkl"
        # Fresh kernel per task: drop any stale context from a prior run.
        try:
            self._ctx_path.unlink()
        except OSError:
            pass

    @property
    def name(self) -> str:
        return "python"

    @property
    def description(self) -> str:
        return (
            "Execute a Python code cell in a persistent sandbox kernel and "
            "return its stdout/stderr and any images it produces."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
            },
            "required": ["code"],
        }

    async def execute(self, code: str, **kwargs: Any) -> list[dict[str, Any]]:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        req_path = self._workspace / f".kernel_req_{token}.json"
        res_path = self._workspace / f".kernel_res_{token}.json"
        req_path.write_text(json.dumps({
            "code": code,
            "variant": self._variant,
            "image_paths": self._image_paths,
            "output_dir": str(self._out_dir),
            "context_path": str(self._ctx_path),
        }))

        result: dict[str, Any] = {"stdout": "", "stderr": "",
                                  "error": None, "images": []}
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, _KERNEL,
                "--req", str(req_path), "--out", str(res_path),
                cwd=str(self._workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**sandbox_subprocess_env(str(self._workspace)), "MPLBACKEND": "Agg"},
            )
            try:
                _, kernel_stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return self._blocks(stdout="", stderr="",
                                    error=f"Execution timed out after {self._timeout}s",
                                    images=[])

            if res_path.exists():
                result = json.loads(res_path.read_text())
            else:
                err = (kernel_stderr or b"").decode("utf-8", "replace")
                result["error"] = f"Kernel produced no result. {err[:500]}"
        except Exception as e:  # noqa: BLE001
            result["error"] = f"Sandbox launch error: {e}"
        finally:
            for p in (req_path, res_path):
                try:
                    p.unlink()
                except OSError:
                    pass

        return self._blocks(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            error=result.get("error"),
            images=result.get("images", []) or [],
        )

    # -- result formatting --------------------------------------------------
    def _blocks(self, stdout: str, stderr: str, error: str | None,
                images: list[str]) -> list[dict[str, Any]]:
        """Render the kernel result as a `[text, image, image, ...]` content
        list. Wrapper text is variant-specific to match each baseline's
        training format."""
        stdout = (stdout or "").strip()
        stderr = (stderr or "").strip()
        if error:
            stderr = (stderr + "\n" + error).strip() if stderr else error

        max_len = 8000
        if len(stdout) > max_len:
            stdout = stdout[:max_len] + f"\n... (truncated {len(stdout) - max_len} chars)"

        img_blocks: list[dict[str, Any]] = []
        for path in images[: self._max_images]:
            block = self._encode_image(path)
            if block is not None:
                img_blocks.append(block)

        if self._variant == "thyme":
            body = stdout if stdout else "(no stdout)"
            if stderr:
                body += f"\n[stderr]\n{stderr}"
            n = len(img_blocks)
            tag = ("<sandbox_output>" + ("<image>" * n) + f"\n{body}\n</sandbox_output>")
            text = tag
        else:  # deepeyesv2
            img_note = ("Images:\n" + "<image>" * len(img_blocks)
                        if img_blocks else "")
            text = (
                "Code execution result:\n"
                f"stdout:\n```\n{stdout}\n```\n\n"
                f"stderr:\n```\n{stderr}\n```\n\n"
                f"{img_note}"
            ).strip()

        return [{"type": "text", "text": text}, *img_blocks]

    @staticmethod
    def _encode_image(path: str) -> dict[str, Any] | None:
        try:
            raw = Path(path).read_bytes()
        except OSError:
            return None
        mime = detect_image_mime(raw) or "image/png"
        b64 = base64.b64encode(raw).decode()
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}}
