"""Harness-wide constants.

WHY: single source for caps, timeouts, and window table so engine, tools,
and footer agree without duplication; CONTEXT_WINDOWS is centralized here
because footer and usage both need the same lookup.
"""

from __future__ import annotations

DEFAULT_MODEL: str = "claude-sonnet-4"

TOOL_TIMEOUT: float = 30.0

HTTP_TIMEOUT: float | None = None

# WHY mirrors provider _FIRST_CONTENT_EVENTS (streaming.py): tool_use_delta never
# arrives without a preceding tool_use_start, so parity with ttft semantics is exact.
CONTENT_EVENT_TYPES: frozenset[str] = frozenset({"text_delta", "thinking_delta", "tool_use_start"})

OUTPUT_CAP: int = 100 * 1024

READ_MAX: int = 1024 * 1024

MAX_TOOL_ITERATIONS: int = 24

SUB_MAX_ITERATIONS: int = 12

PAUSE_CAP: int = 3

# Policy: tools whose calls skip the permission prompt entirely. Reads are
# safe to expose because they cannot mutate state; empty this set to gate
# every tool again.
POLICY_AUTO_ALLOW: frozenset[str] = frozenset({"read_file"})

# Reads whose basename matches these patterns are NEVER auto-allowed —
# confidentiality (not mutation) is what they threaten.
SECRET_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_*",
    "*credentials*",
    "*.p12",
    "*.pfx",
    ".netrc",
    ".pgpass",
    "shadow",
    "passwd",
)

# Tools refused outright (before the gate) when their path leaves the sandbox;
# other path tools just lose auto-allow and get the normal prompt.
SANDBOX_DENY_TOOLS: frozenset[str] = frozenset({"write_file"})

# Ceiling per paginated read_file call: model-supplied limit is clamped here
# so "offset/limit for huge files" can neither balloon RAM nor drip-feed GBs.
READ_MAX_LINES: int = 2000

# Gate prompt display caps: single-line (collapsed) vs multiline (diff) — the
# stored "always" keys are always the raw strings, never the display form.
PREVIEW_DISPLAY_CHARS: int = 500
PREVIEW_DISPLAY_MULTILINE: int = 4000

# How many unified-diff lines the write_file approval preview shows.
DIFF_PREVIEW_LINES: int = 40

# Project-context file, read from the project root (git toplevel) or the
# launch dir when not inside a repo; cap keeps a giant file from eating the
# context window.
AGENTS_FILE: str = "AGENTS.md"
AGENTS_MAX_BYTES: int = 32 * 1024

# Hard bound on the upward .git search: launch dir + this many ancestors.
# WHY capped: an uncapped walk to filesystem root could stat hundreds of
# directories on deep trees, every turn, for a "project" far above the user's.
PROJECT_ROOT_MAX_LEVELS: int = 5

# Image attachments (/attach): extension→media type and a size bound roughly
# matching the largest per-image limit any built-in vendor accepts.
IMAGE_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
IMAGE_MAX_BYTES: int = 5 * 1024 * 1024

COMPACT_KEEP_TAIL: int = 6

BUDGET_WARN_FRACTION: float = 0.85

# WHY heuristic: images are token-heavy; 1600 approximates a typical
# vision-token cost when no usage data is available.
IMAGE_EST_TOKENS: int = 1600

CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku": 200_000,
    "claude": 200_000,
    "gpt-4o": 128_000,
    "gpt-4": 8192,
    "gpt-3.5": 16385,
    "o1": 200_000,
    "o3": 200_000,
    "llama3": 128_000,
    "llama": 128_000,
}
