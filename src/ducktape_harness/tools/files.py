"""File tools: read_file and write_file.

WHY: read caps at 1 MB with offset/limit hint (prevents context blow-up);
write is uncapped because user explicitly requested the write.
"""

from __future__ import annotations

import base64
import difflib
import itertools
import os
from typing import Any, cast

from ducktape_harness.config import (
    DIFF_PREVIEW_LINES,
    IMAGE_MAX_BYTES,
    IMAGE_TYPES,
    OUTPUT_CAP,
    READ_MAX,
    READ_MAX_LINES,
)
from ducktape_harness.tools.base import ToolError, register


def resolve_path(path: Any, cwd: str) -> str | None:
    """Absolute target for a tool path arg, or None when unusable."""
    if not isinstance(path, str) or not path:
        return None
    return path if os.path.isabs(path) else os.path.join(cwd, path)


class ReadFileTool:
    name = "read_file"
    description = "Read a file. Supports offset/limit for pagination. Fails if >1 MB."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "offset": {"type": "integer", "description": "Line offset (0-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to read"},
            "image": {
                "type": "boolean",
                "description": "Return the file as an image content block",
            },
        },
        "required": ["path"],
    }

    def sandbox_target(self, input: dict[str, Any], cwd: str) -> str | None:
        # WHY: lets the engine bound read_file's auto-allow to the workspace.
        return resolve_path(input.get("path"), cwd)

    def _as_image(self, path: str) -> list[dict[str, Any]]:
        # WHY returns blocks: the provider widened ToolResultBlock.content to
        # accept text/image blocks (gap #2 fixed), so the model can SEE files.
        media = IMAGE_TYPES.get(os.path.splitext(path)[1].lower())
        if media is None:
            raise ToolError("image read requires png/jpg/jpeg/gif/webp")
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            raise ToolError(str(e)) from e
        if size > IMAGE_MAX_BYTES:
            raise ToolError(f"image too large ({size} bytes > {IMAGE_MAX_BYTES})")
        if not data:
            raise ToolError(f"{path} is empty")
        return [
            {
                "type": "image",
                "source": "base64",
                "media_type": media,
                "data": base64.b64encode(data).decode("ascii"),
            }
        ]

    def run(self, input: dict[str, Any], cwd: str) -> Any:
        path = input.get("path", "")
        if not isinstance(path, str):
            raise ToolError("path must be a string")
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        if input.get("image"):
            return self._as_image(path)
        offset = input.get("offset", 0)
        limit = input.get("limit")
        try:
            offset_i = int(offset) if offset is not None else 0
            limit_i = int(limit) if limit is not None else None
        except (ValueError, TypeError):
            raise ToolError("offset/limit must be integers") from None
        if offset_i < 0:
            raise ToolError("offset must be >= 0")
        if limit_i is not None and limit_i < 0:
            raise ToolError("limit must be >= 0")
        explicit_limit = input.get("limit") is not None
        if explicit_limit:
            # WHY clamp model-supplied limits: an unbounded `limit` would make
            # pagination a RAM balloon / drip-feed exfil channel.
            limit_i = min(cast(int, limit_i), READ_MAX_LINES)
        paginated = offset_i > 0 or explicit_limit
        if paginated and limit_i is None:
            # offset-only must not slurp the whole tail into RAM
            limit_i = READ_MAX_LINES
        try:
            size = os.path.getsize(path)
        except OSError as e:
            raise ToolError(str(e)) from e
        # WHY the pagination exemption: rejecting before slicing made the
        # "use offset/limit" hint unsatisfiable for any large file.
        if size > READ_MAX and not paginated:
            raise ToolError(
                f"file too large ({size} bytes > {READ_MAX}); use offset/limit to read in chunks"
            )
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                end = None if limit_i is None else offset_i + limit_i
                sliced = list(itertools.islice(f, offset_i, end))
        except OSError as e:
            raise ToolError(str(e)) from e
        content = "".join(sliced)
        if len(content.encode("utf-8")) > OUTPUT_CAP:
            b = content.encode("utf-8")[:OUTPUT_CAP].decode("utf-8", errors="ignore")
            return b + "\n[truncated]"
        return content


class WriteFileTool:
    name = "write_file"
    description = "Write content to a file (overwrites)."
    preview_multiline = True  # gate keeps the diff's newlines
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "File content"},
        },
        "required": ["path", "content"],
    }

    def preview(self, input: dict[str, Any], cwd: str) -> str:
        """Unified diff old→new shown at the approval prompt (diff approval)."""
        target = resolve_path(input.get("path"), cwd)
        content = input.get("content", "")
        if target is None:
            return str(input)
        if not isinstance(content, str):
            content = str(content)
        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                old = f.read().splitlines()
        except OSError:
            old = []
        diff = list(
            difflib.unified_diff(
                old,
                content.splitlines(),
                fromfile=f"{target} (current)",
                tofile=f"{target} (new)",
            )
        )
        if not diff:
            return f"{target}: no change"
        shown = diff[:DIFF_PREVIEW_LINES]
        out = "\n".join(line.rstrip() for line in shown)
        if len(diff) > len(shown):
            out += f"\n… [diff truncated, {len(diff)} lines total]"
        return out

    def sandbox_target(self, input: dict[str, Any], cwd: str) -> str | None:
        return resolve_path(input.get("path"), cwd)

    def run(self, input: dict[str, Any], cwd: str) -> str:
        path = input.get("path", "")
        content = input.get("content", "")
        if not isinstance(path, str):
            raise ToolError("path must be a string")
        if not isinstance(content, str):
            content = str(content)
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            raise ToolError(str(e)) from e
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"


register(ReadFileTool())
register(WriteFileTool())

# Ensure REGISTRY populated for tests that import files module
__all__ = ["ReadFileTool", "WriteFileTool"]
