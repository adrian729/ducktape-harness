"""Subagent tool — run a fresh nested agent loop with isolated context.

WHY: Delegating exploration/research to a subagent keeps the main
conversation clean; the parent sees only the final distilled text.
"""

from __future__ import annotations

from typing import Any

from ducktape_harness.session import Session
from ducktape_harness.tools.base import REGISTRY


def run_subagent(
    session: Any,
    description: str,
    prompt: str,
    requested_tools: list[str] | None,
) -> str:
    """Execute a subagent turn and merge its usage into the parent.

    Returns the subagent's final text, prefixed with an unknown-tools note
    when requested names were not in the registry.
    """
    # Build allowed set: requested or all, minus task, intersect registry.
    # Spec verbatim: allowed = set(requested_tools or REGISTRY keys) - {"task"} intersect REGISTRY
    raw = requested_tools if requested_tools else list(REGISTRY.keys())
    allowed = (set(raw) - {"task"}) & set(REGISTRY.keys())
    if requested_tools:
        unknown = set(requested_tools) - set(REGISTRY.keys())
        unknown.discard("task")
    else:
        unknown = set()

    # Fresh session with shared gate/provider/sandbox but isolated history.
    sub = Session(
        model=session.model,
        pin=session.pin,
        provider=session.provider,
        sandbox_root=session.sandbox_root,
        gate=session.gate,
    )

    try:
        # WHY lazy import: engine imports this module for the task dispatch;
        # importing at top would create a circular dependency.
        from ducktape_harness.engine import run_turn

        out = run_turn(sub, prompt, tool_names=allowed, is_sub=True, quiet=True)
    except Exception as e:
        # engine recovers internally, but guard against any propagation
        out = f"[subagent error] {e}"
    if sub.last_interrupted:
        # real user abort: propagate (KI is not Exception, so the parent's
        # try/except above can't swallow it) and let the 5a batch path deny
        # the remaining tool calls — a failed sub must NOT look like this
        raise KeyboardInterrupt
    out = out if out else "(subagent ended with no output)"

    # Merge token accounting into parent: parent's totals absorb sub's,
    # turns merge too so the usage-reported N-of-M ratio stays coherent.
    # WHY: the user sees one parent turn's cost; sub tokens are still billed.
    try:
        session.totals.input_tokens += sub.totals.input_tokens
        session.totals.output_tokens += sub.totals.output_tokens
        session.totals.cache_read_tokens += sub.totals.cache_read_tokens
        session.totals.cache_write_tokens += sub.totals.cache_write_tokens
        session.totals.turns_with_usage += sub.totals.turns_with_usage
        # WHY turns too: sub turns ARE model turns and /usage's N-of-M line
        # would exceed 100% otherwise
        session.totals.turns += sub.totals.turns
    except Exception:
        pass

    # Prefix unknown-tools note if any
    prefix = ""
    if unknown:
        # sort for deterministic output
        prefix = f"[subagent] unknown tools ignored: {', '.join(sorted(unknown))}\n"

    result = f"{prefix}{out}" if prefix else out

    # Print marker for visibility (always, even when quiet on streaming)
    try:
        print(f"[subagent] {description}: done ({len(result)} chars)")
    except Exception:
        pass

    return result
