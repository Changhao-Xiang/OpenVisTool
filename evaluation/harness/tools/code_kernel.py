"""Standalone code-execution kernel for the DeepEyesV2 / Thyme baselines.

Both baselines drive the model through a *code-as-tool* text protocol rather
than native function calling: the model writes a Python cell, an external
sandbox runs it, and the stdout / generated images are fed back. This script
is that sandbox. It is intentionally self-contained (no project imports) so it
can be launched as a plain subprocess:

    python harness/tools/code_kernel.py --req <request.json> --out <result.json>

Request JSON:
    {
      "code":         "<python source>",
      "variant":      "thyme" | "deepeyesv2",
      "image_paths":  ["/abs/img1.png", ...],
      "output_dir":   "/abs/sandbox/<dir for generated images>",
      "context_path": "/abs/sandbox/.kernel_ctx.pkl"   # persisted kernel state
    }

Result JSON:
    {"stdout": str, "stderr": str, "error": str|null, "images": [paths...]}

State persistence mirrors the reference implementations: picklable locals are
saved to ``context_path`` and restored on the next cell, and imported modules
are re-imported by name. This approximates a long-lived Jupyter kernel without
keeping a process alive across async workers.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import pickle
import re
import types

# matplotlib must be headless and configured before pyplot import.
os.environ.setdefault("MPLBACKEND", "Agg")


# --- Lightweight static safety guard (mirrors Thyme's sandbox) -------------
_DANGEROUS_PATTERNS = [
    r"\bsubprocess\.",
    r"\bsocket\.",
    r"\bos\.(remove|unlink|rmdir|system|popen|kill|rename|renames)\b",
    r"\bshutil\.(rmtree|move)\b",
    r"\b__import__\(\s*['\"](subprocess|socket)['\"]",
]


def _is_dangerous(code: str) -> bool:
    return any(re.search(p, code, re.IGNORECASE) for p in _DANGEROUS_PATTERNS)


_VALID_IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
# Training-time path prefix the model hard-codes for its output images; Thyme's
# reference sandbox rewrites it to the real per-task output dir.
_THYME_OUT_PREFIX = "/mnt/data/temp_processed_images/"


class _ImagePathTransformer(ast.NodeTransformer):
    """Rewrite ``image_path = "<literal>"`` to the real input image when the
    literal is not an existing image file. Mirrors Thyme's sandbox: the model
    is trained to hard-code a /mnt/data/... path that does not exist here."""

    def __init__(self, replacement: str):
        self.replacement = replacement
        self.changed = False

    def visit_Assign(self, node):  # noqa: N802
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "image_path"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            val = node.value.value
            valid = (os.path.isfile(val)
                     and val.lower().endswith(_VALID_IMG_EXT))
            if not valid:
                node.value = ast.Constant(value=self.replacement)
                self.changed = True
        return self.generic_visit(node)


def _thyme_rewrite_code(code: str, image_paths: list[str], output_dir: str) -> str:
    """Apply Thyme's sandbox code rewrites: redirect a hard-coded ``image_path``
    to the real input image, and the /mnt/data/temp_processed_images/ output
    prefix to the real output dir (creating implied subdirs)."""
    if image_paths:
        try:
            tree = ast.parse(code)
            tf = _ImagePathTransformer(image_paths[0])
            new_tree = tf.visit(tree)
            if tf.changed:
                ast.fix_missing_locations(new_tree)
                code = ast.unparse(new_tree)
        except SyntaxError:
            pass

    if _THYME_OUT_PREFIX in code:
        repl = output_dir.rstrip("/") + "/"
        for m in re.finditer(re.escape(_THYME_OUT_PREFIX) + r"([^'\"\s]*)", code):
            sub = os.path.dirname(m.group(1))
            if sub:
                os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
        code = code.replace(_THYME_OUT_PREFIX, repl)
    return code


def _list_images(directory: str) -> set[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    out: set[str] = set()
    if not os.path.isdir(directory):
        return out
    for name in os.listdir(directory):
        if name.lower().endswith(exts):
            out.add(os.path.join(directory, name))
    return out


def _build_namespace(variant: str, image_paths: list[str], output_dir: str) -> dict:
    """Construct the execution globals with the modules + image variables the
    baseline's training data assumes are already present."""
    import math
    import random
    import string

    import numpy as np
    from PIL import Image

    ns: dict = {
        "__name__": "__sandbox__",
        "os": os,
        "math": math,
        "random": random,
        "string": string,
        "json": json,
        "re": re,
        "io": io,
        "np": np,
        "numpy": np,
        "Image": Image,
    }
    try:  # cv2 is optional but expected by both baselines
        import cv2

        ns["cv2"] = cv2
    except Exception:  # noqa: BLE001
        pass
    try:
        import matplotlib.pyplot as plt

        ns["plt"] = plt
    except Exception:  # noqa: BLE001
        plt = None

    if variant == "deepeyesv2":
        # DeepEyesV2: every input image is pre-opened as image_1, image_2, ...
        for i, p in enumerate(image_paths, start=1):
            try:
                ns[f"image_{i}"] = Image.open(p)
            except Exception:  # noqa: BLE001
                ns[f"image_{i}"] = None
        if image_paths:
            ns["image"] = ns.get("image_1")
    else:  # thyme
        # Thyme: the first image path + an output dir are bound as locals.
        ns["image_path"] = image_paths[0] if image_paths else ""
        ns["temp_output_dir"] = output_dir

    return ns


def _patch_plt_show(ns: dict, output_dir: str, saved: list[str]) -> None:
    """Redirect ``plt.show()`` to save the current figures (DeepEyesV2 returns
    figures rendered with plt.show())."""
    plt = ns.get("plt")
    if plt is None:
        return

    def _show(*_a, **_k):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            path = os.path.join(output_dir, f"_kernel_fig_{len(saved)}.png")
            try:
                fig.savefig(path, bbox_inches="tight")
                saved.append(path)
            except Exception:  # noqa: BLE001
                pass
        plt.close("all")

    plt.show = _show


def _restore_context(ns: dict, context_path: str) -> None:
    if not context_path or not os.path.isfile(context_path):
        return
    try:
        with open(context_path, "rb") as f:
            ctx = pickle.load(f)
    except Exception:  # noqa: BLE001
        return
    for alias, mod_name in (ctx.get("modules") or {}).items():
        try:
            import importlib

            ns[alias] = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            pass
    ns.update(ctx.get("locals") or {})


def _save_context(ns: dict, base_keys: set[str], context_path: str) -> None:
    if not context_path:
        return
    modules: dict[str, str] = {}
    picklable: dict[str, object] = {}
    for name, value in ns.items():
        if name in base_keys or name.startswith("__"):
            continue
        if isinstance(value, types.ModuleType):
            modules[name] = value.__name__
            continue
        try:
            pickle.dumps(value)
            picklable[name] = value
        except Exception:  # noqa: BLE001
            continue
    try:
        with open(context_path, "wb") as f:
            pickle.dump({"modules": modules, "locals": picklable}, f)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.req) as f:
        req = json.load(f)

    code = req.get("code", "")
    variant = req.get("variant", "deepeyesv2")
    image_paths = req.get("image_paths", []) or []
    output_dir = req.get("output_dir", ".")
    context_path = req.get("context_path", "")

    result = {"stdout": "", "stderr": "", "error": None, "images": []}

    if _is_dangerous(code):
        result["error"] = (
            "Code contains a disallowed operation (file deletion / subprocess / "
            "socket). Execution denied."
        )
        with open(args.out, "w") as f:
            json.dump(result, f)
        return

    os.makedirs(output_dir, exist_ok=True)
    if variant == "thyme":
        code = _thyme_rewrite_code(code, image_paths, output_dir)
    pre_existing = _list_images(output_dir)

    ns = _build_namespace(variant, image_paths, output_dir)
    base_keys = set(ns.keys())
    saved_figs: list[str] = []
    _patch_plt_show(ns, output_dir, saved_figs)
    _restore_context(ns, context_path)

    # Syntax check first so we can surface a clean error.
    try:
        ast.parse(code)
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        with open(args.out, "w") as f:
            json.dump(result, f)
        return

    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(compile(code, "<cell>", "exec"), ns, ns)
    except Exception as e:  # noqa: BLE001
        import traceback

        result["error"] = f"{type(e).__name__}: {e}"
        err_buf.write(traceback.format_exc())

    result["stdout"] = out_buf.getvalue()
    result["stderr"] = err_buf.getvalue()

    # --- Collect produced images ------------------------------------------
    images: list[str] = list(saved_figs)
    stdout_text = result["stdout"]

    # Thyme: the model is told to print `processed_path`; also honour a
    # `processed_path` local variable when present.
    if variant == "thyme":
        pp = ns.get("processed_path")
        if isinstance(pp, str) and os.path.isfile(pp) and pp not in images:
            images.append(pp)

    # Any newly created image files in the output dir (both variants).
    for p in sorted(_list_images(output_dir) - pre_existing):
        if p not in images:
            images.append(p)

    # Image paths explicitly printed to stdout (point to real files).
    for m in re.finditer(r"[^\s'\"]+\.(?:png|jpg|jpeg|bmp|gif|tiff|webp)", stdout_text):
        cand = m.group(0)
        if os.path.isfile(cand) and cand not in images:
            images.append(cand)

    result["images"] = images

    _save_context(ns, base_keys, context_path)

    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
