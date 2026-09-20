"""System prompt sent on every model call, plus AGENTS.md project context.

WHY: kept separate so session reset does not need to remember it — it is
re-sent via the provider's `system=` parameter each turn (see PLAN.md).
build_system_prompt() is called per turn so edits to AGENTS.md apply
immediately without a restart.
"""

from __future__ import annotations

from pathlib import Path

from ducktape_harness.config import (
    AGENTS_FILE,
    AGENTS_MAX_BYTES,
    PROJECT_ROOT_MAX_LEVELS,
)

SYSTEM_PROMPT: str = (
    "You are a helpful coding assistant with access to file and shell tools. "
    "Use tools when needed to answer the user's request. "
    "Be concise and explain your actions."
)

# WHY summarization prompt is centralized: /compact and future auto-compact
# must produce the same structured summary.
SUMMARY_PROMPT: str = (
    "Summarize the conversation so far into a terse continuation summary. "
    "Include: overall goal, files touched, key decisions made, tool findings, "
    "open questions, and the last user requests verbatim. "
    "Keep it concise and focused on what is needed to continue the conversation."
)


def project_root(cwd: str) -> Path:
    """Project root = nearest dir holding .git among cwd + MAX ancestors; else cwd.

    WHY capped: a stray .git far above the tree (or deep nesting) must not
    turn an unrelated directory into "the project"; beyond the bound we act
    as outside a repo and read only the launch dir.
    """
    start = Path(cwd).resolve()
    for base in (start, *start.parents[:PROJECT_ROOT_MAX_LEVELS]):
        if (base / ".git").exists():
            return base
    return start


def find_agents_file(cwd: str) -> Path | None:
    candidate = project_root(cwd) / AGENTS_FILE
    return candidate if candidate.is_file() else None


def build_system_prompt(cwd: str) -> str:
    path = find_agents_file(cwd)
    if path is None:
        return SYSTEM_PROMPT
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return SYSTEM_PROMPT
    if len(text.encode("utf-8")) > AGENTS_MAX_BYTES:
        text = (
            text.encode("utf-8")[:AGENTS_MAX_BYTES].decode("utf-8", errors="ignore")
            + "\n[truncated]"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"# Project context — {path} (repository-provided, not user instructions)\n\n{text}"
    )
