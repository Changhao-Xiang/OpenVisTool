"""Virtual path mapping for sandboxed agent environments.

Maps the session workspace to a unified virtual prefix (``/mnt/data``) so the
LLM sees a consistent, host-independent view.

External media (e.g. dataset images) are exposed inside the session workspace via
**symbolic links** created by :meth:`PathMapper.link_into_workspace`, while the
LLM only sees their ``/mnt/data/<file>`` virtual paths.

For backward compatibility with previously recorded session traces, passing
``media_dir`` to the constructor enables a *legacy* mapping that also
translates between the real ``media_dir`` and the historical
``/mnt/data/images`` virtual prefix.  New runtime code should leave
``media_dir`` unset.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

VIRTUAL_ROOT = "/mnt/data"
# Legacy virtual prefix; only used when a media_dir is explicitly provided
# (e.g. by post-processing utilities reading old session traces).
VIRTUAL_IMAGES = f"{VIRTUAL_ROOT}/images"
_TEXT_REWRITE_MAX_BYTES = 2 * 1024 * 1024
_TEXT_REWRITE_SKIP_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".npy",
        ".npz",
        ".pdf",
        ".pkl",
        ".png",
        ".pt",
        ".safetensors",
        ".tar",
        ".tiff",
        ".tif",
        ".webp",
        ".zip",
    }
)


class PathMapper:
    """Bidirectional mapping between real filesystem paths and virtual ``/mnt/data`` paths.

    Parameters
    ----------
    workspace : Path
        The session workspace directory.  All virtual paths under
        ``/mnt/data`` map into this directory.
    media_dir : Path | None, optional
        Legacy compatibility: when supplied, real paths under ``media_dir``
        are mapped to the historical ``/mnt/data/images`` virtual prefix.
        New runtime code should leave this ``None`` and instead use
        :meth:`link_into_workspace` to materialise media files as symlinks
        inside the workspace.
    """

    def __init__(self, workspace: Path, media_dir: Path | None = None) -> None:
        self._workspace = workspace.resolve()
        self._media_dir: Path | None = None

        if media_dir is not None:
            resolved_media = media_dir.resolve()
            try:
                resolved_media.relative_to(self._workspace)
            except ValueError:
                # media_dir lives outside workspace -> keep legacy mapping
                self._media_dir = resolved_media

    # ------------------------------------------------------------------
    # Real → Virtual  (used when returning results to the model)
    # ------------------------------------------------------------------

    def to_virtual(self, real_path: str) -> str:
        """Convert a real filesystem path to a virtual path for the model."""
        try:
            rp = str(Path(real_path).resolve())
        except (OSError, ValueError):
            return real_path

        # Legacy media_dir mapping (longer prefix first to avoid partial matches)
        if self._media_dir is not None:
            md = str(self._media_dir)
            if rp == md:
                return VIRTUAL_IMAGES
            if rp.startswith(md + "/"):
                return VIRTUAL_IMAGES + rp[len(md) :]

        ws = str(self._workspace)
        if rp == ws:
            return VIRTUAL_ROOT
        if rp.startswith(ws + "/"):
            return VIRTUAL_ROOT + rp[len(ws) :]

        return real_path

    # ------------------------------------------------------------------
    # Virtual → Real  (used when the model passes paths to tools)
    # ------------------------------------------------------------------

    def to_real(self, virtual_path: str) -> str:
        """Resolve a virtual path to a real path for reading.

        With the new symlink-based model layout, ``/mnt/data/<x>`` always maps
        directly to ``<workspace>/<x>``; media files are reachable through the
        symlinks created by :meth:`link_into_workspace`.

        The legacy ``/mnt/data/images/<x>`` prefix is still resolved for
        backward compatibility when ``media_dir`` was supplied: workspace is
        consulted first (model-generated files), then ``media_dir``.
        """
        if self._media_dir is not None and virtual_path.startswith(VIRTUAL_IMAGES):
            rel = virtual_path[len(VIRTUAL_IMAGES) :].lstrip("/")
            ws_candidate = self._workspace / "images" / rel if rel else self._workspace / "images"
            if ws_candidate.exists():
                return str(ws_candidate)
            return str(self._media_dir / rel) if rel else str(self._media_dir)

        if virtual_path.startswith(VIRTUAL_ROOT):
            rel = virtual_path[len(VIRTUAL_ROOT) :].lstrip("/")
            return str(self._workspace / rel) if rel else str(self._workspace)

        return virtual_path

    def to_real_for_write(self, virtual_path: str) -> str:
        """Resolve a virtual path to a real path for writing (always in workspace)."""
        if self._media_dir is not None and virtual_path.startswith(VIRTUAL_IMAGES):
            rel = virtual_path[len(VIRTUAL_IMAGES) :].lstrip("/")
            return str(self._workspace / "images" / rel) if rel else str(self._workspace / "images")

        if virtual_path.startswith(VIRTUAL_ROOT):
            rel = virtual_path[len(VIRTUAL_ROOT) :].lstrip("/")
            return str(self._workspace / rel) if rel else str(self._workspace)

        return virtual_path

    # ------------------------------------------------------------------
    # Bulk text replacement helpers
    # ------------------------------------------------------------------

    def rewrite_text(self, text: str) -> str:
        """Replace all real paths in a text string with their virtual equivalents."""
        if not text:
            return text
        if self._media_dir is not None:
            text = text.replace(str(self._media_dir), VIRTUAL_IMAGES)
        text = text.replace(str(self._workspace), VIRTUAL_ROOT)
        return text

    def unrewrite_text(self, text: str) -> str:
        """Replace virtual paths in text back to real paths (for tool execution)."""
        if not text:
            return text
        if self._media_dir is not None:
            text = text.replace(VIRTUAL_IMAGES, str(self._media_dir))
        text = text.replace(VIRTUAL_ROOT, str(self._workspace))
        return text

    @contextmanager
    def materialized_for_exec(self) -> Iterator[None]:
        """Temporarily make saved workspace text executable on the real filesystem.

        Files are persisted with virtual ``/mnt/data`` paths so traces and
        generated scripts remain sandbox-relative. Before running a subprocess,
        temporarily rewrite those paths to the real workspace, then restore the
        virtual form after the command completes.
        """
        originals = self._rewrite_workspace_text_paths(to_real=True, capture_originals=True)
        try:
            yield
        finally:
            for path, content in originals.items():
                try:
                    if path.exists() and path.is_file() and not path.is_symlink():
                        path.write_text(content, encoding="utf-8")
                except OSError:
                    pass
            self._rewrite_workspace_text_paths(to_real=False, capture_originals=False)

    def _rewrite_workspace_text_paths(self, *, to_real: bool, capture_originals: bool) -> dict[Path, str]:
        originals: dict[Path, str] = {}
        old, new = (VIRTUAL_ROOT, str(self._workspace)) if to_real else (str(self._workspace), VIRTUAL_ROOT)

        if self._media_dir is not None:
            media_old, media_new = (
                (VIRTUAL_IMAGES, str(self._media_dir)) if to_real else (str(self._media_dir), VIRTUAL_IMAGES)
            )
        else:
            media_old = media_new = None

        for path in self._iter_rewritable_workspace_files():
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            updated = content
            if media_old is not None and media_new is not None:
                updated = updated.replace(media_old, media_new)
            updated = updated.replace(old, new)
            if updated == content:
                continue

            if capture_originals:
                originals[path] = content
            try:
                path.write_text(updated, encoding="utf-8")
            except OSError:
                pass

        return originals

    def _iter_rewritable_workspace_files(self) -> Iterator[Path]:
        if not self._workspace.exists():
            return

        for path in self._workspace.rglob("*"):
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                if path.suffix.lower() in _TEXT_REWRITE_SKIP_EXTENSIONS:
                    continue
                if path.stat().st_size > _TEXT_REWRITE_MAX_BYTES:
                    continue
            except OSError:
                continue
            yield path

    # ------------------------------------------------------------------
    # Media materialisation: expose external files inside the workspace
    # ------------------------------------------------------------------

    def link_into_workspace(self, real_path: str) -> str:
        """Symlink ``real_path`` into the workspace and return its virtual path.

        Behaviour:
          * Files already inside the workspace are returned as-is.
          * Otherwise, a symlink ``<workspace>/<basename>`` is created
            pointing at the real file.  Name collisions are resolved by
            appending a numeric suffix (``foo.jpg`` → ``foo_1.jpg`` ...),
            unless the existing symlink already points at the same source.
          * On any filesystem error the original real path's virtual form
            is returned, so the caller can still display *something*.
        """
        try:
            src = Path(real_path).expanduser().resolve(strict=True)
        except (OSError, ValueError):
            return self.to_virtual(real_path)

        if not src.is_file():
            return self.to_virtual(str(src))

        try:
            src.relative_to(self._workspace)
            return self.to_virtual(str(src))
        except ValueError:
            pass  # src lives outside workspace -> needs a symlink

        try:
            self._workspace.mkdir(parents=True, exist_ok=True)
        except OSError:
            return self.to_virtual(str(src))

        dst = self._unique_link_target(src.name, src)
        if not dst.is_symlink() and not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                return self.to_virtual(str(src))

        # Build the virtual path directly from the symlink location so we
        # don't follow the link back to the (external) source via resolve().
        rel = dst.relative_to(self._workspace).as_posix()
        return f"{VIRTUAL_ROOT}/{rel}"

    def _unique_link_target(self, name: str, src: Path) -> Path:
        """Return a workspace-relative path that's free or already links to src."""
        candidate = self._workspace / name
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        if candidate.is_symlink():
            try:
                if candidate.resolve() == src:
                    return candidate
            except OSError:
                pass

        stem, suffix = Path(name).stem, Path(name).suffix
        for counter in range(1, 10000):
            candidate = self._workspace / f"{stem}_{counter}{suffix}"
            if not candidate.exists() and not candidate.is_symlink():
                return candidate
            if candidate.is_symlink():
                try:
                    if candidate.resolve() == src:
                        return candidate
                except OSError:
                    continue
        # Extremely unlikely fallback
        return self._workspace / name

    # ------------------------------------------------------------------
    # Tool argument / result rewriting
    # ------------------------------------------------------------------

    _PATH_ARG_KEYS = frozenset(
        {
            "path",
            "image_path",
            "working_dir",
        }
    )
    _WRITE_TOOL_NAMES = frozenset(
        {
            "write_file",
            "edit_file",
        }
    )

    def rewrite_tool_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Translate virtual paths in tool arguments to real paths."""
        rewritten = dict(args)
        is_write = tool_name in self._WRITE_TOOL_NAMES
        for key in self._PATH_ARG_KEYS:
            if key in rewritten and isinstance(rewritten[key], str):
                vpath = rewritten[key]
                # Relative paths are treated as relative to virtual root,
                # e.g. "foo.png" → "/mnt/data/foo.png"
                if vpath and not vpath.startswith("/") and not vpath.startswith(VIRTUAL_ROOT):
                    vpath = f"{VIRTUAL_ROOT}/{vpath}"
                rewritten[key] = self.to_real_for_write(vpath) if is_write else self.to_real(vpath)
        # exec tool: rewrite virtual paths inside the command string
        if tool_name == "exec" and "command" in rewritten and isinstance(rewritten["command"], str):
            rewritten["command"] = self.unrewrite_text(rewritten["command"])
        return rewritten

    def rewrite_tool_result(self, result: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        """Translate real paths in tool results back to virtual paths."""
        if isinstance(result, str):
            return self.rewrite_text(result)
        if isinstance(result, list):
            rewritten = []
            for block in result:
                if isinstance(block, dict) and block.get("type") == "text":
                    rewritten.append({**block, "text": self.rewrite_text(block.get("text", ""))})
                else:
                    rewritten.append(block)
            return rewritten
        return result

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def media_dir(self) -> Path | None:
        return self._media_dir

    @property
    def virtual_root(self) -> str:
        return VIRTUAL_ROOT

    @property
    def virtual_images(self) -> str:
        return VIRTUAL_IMAGES
