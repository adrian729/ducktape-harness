"""Permission gate for tool calls.

WHY: per-call y/N/always with session memory; 'always' is per tool name
and survives /clear, deliberately coarse for bash (documented).
"""

from __future__ import annotations


class PermissionGate:
    """Gate holding always-allowed entries and prompting.

    WHY: for `bash`, "always" is keyed by the exact command, not the tool
    name — remembering "bash" would ungated every future command. A bare
    name-level entry still works if seeded programmatically (tests).
    """

    def __init__(self, always_allowed: set[str] | None = None) -> None:
        self.always_allowed: set[str] = set(always_allowed) if always_allowed else set()

    def _keys(self, name: str, key: str) -> set[str]:
        # WHY key != display: the identity used for "always" is canonical
        # (engine passes the raw command for bash), never the pretty preview.
        if name == "bash":
            return {name, f"bash::{key}"}
        return {name}

    def is_always_allowed(self, name: str, key: str = "") -> bool:
        return not self._keys(name, key).isdisjoint(self.always_allowed)

    def ask(
        self,
        name: str,
        args_preview: str,
        key: str | None = None,
        multiline: bool = False,
    ) -> str:
        """Prompt user; returns 'allow', 'deny', or 'always'.

        WHY: raises KeyboardInterrupt/EOFError to let engine apply the
        2-layer pairing invariant (interrupted block + batch-denied remainder).
        """
        key = args_preview if key is None else key
        if self.is_always_allowed(name, key):
            return "allow"
        # WHY mode-aware: bash previews must not split the prompt (collapse to
        # one line), but diff previews ARE their structure — newlines kept.
        if multiline:
            from ducktape_harness.config import PREVIEW_DISPLAY_MULTILINE

            shown = args_preview
            if len(shown) > PREVIEW_DISPLAY_MULTILINE:
                shown = shown[:PREVIEW_DISPLAY_MULTILINE] + "\n…[truncated]"
        else:
            from ducktape_harness.config import PREVIEW_DISPLAY_CHARS

            shown = " ".join(args_preview.split())
            if len(shown) > PREVIEW_DISPLAY_CHARS:
                shown = shown[:PREVIEW_DISPLAY_CHARS] + " …[truncated]"
        args = f" {shown}" if shown else ""
        prompt = f"Allow tool {name}{args}? [y/N/always]: "
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return "allow"
        if ans == "always":
            self.always_allowed.add(f"bash::{key}" if name == "bash" else name)
            return "allow"
        return "deny"
