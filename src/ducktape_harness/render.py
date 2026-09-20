"""Rendering helpers — streaming deltas and footer.

WHY: keeps engine focused on control flow; footer logic centralizes
context-window lookup and segment-drop rules.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from ducktape_provider.types import Usage

from ducktape_harness.config import CONTEXT_WINDOWS


def load_windows() -> dict[str, int]:
    """Merge built-in table with optional user toml."""
    table = dict(CONTEXT_WINDOWS)
    path = Path(os.path.expanduser("~/.config/ducktape/windows.toml"))
    try:
        if path.is_file():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            # Expect { "windows": { "prefix": int } } or flat { prefix = int }
            windows = data.get("windows", data)
            if isinstance(windows, dict):
                for k, v in windows.items():
                    try:
                        table[str(k)] = int(v)  # type: ignore[arg-type]
                    except (ValueError, TypeError):
                        continue
    except Exception:
        pass
    return table


_WINDOW_CACHE: dict[tuple[str, str | None], int | None] = {}


def clear_window_cache() -> None:
    _WINDOW_CACHE.clear()


def provider_window(model: str, provider: object | None, pin: str | None) -> int | None:
    """Context window from provider metadata (gap #1 fixed in b0ad0bb).

    WHY cache: model_info hits live vendors; windows change so rarely that
    caching per (model, pin) is the honest trade. None results cached too,
    so openai/unknown never re-probes each footer.
    """
    if provider is None:
        return None
    key = (model, pin)
    if key not in _WINDOW_CACHE:
        try:
            mi = getattr(provider, "model_info", None)
            if callable(mi):
                info = mi(model, provider=pin) if pin else mi(model)
                win = info.get("context_window") if isinstance(info, dict) else None
            else:
                win = None
            # explicit answers (incl. None = vendor knows nothing) are sticky;
            # exceptions are NOT cached so a transient blip retries next turn
            _WINDOW_CACHE[key] = win if isinstance(win, int) and win > 0 else None
        except Exception:
            return None
    return _WINDOW_CACHE[key]


def context_window_for(
    model: str,
    windows: dict[str, int] | None = None,
    provider: object | None = None,
    pin: str | None = None,
) -> int | None:
    win = provider_window(model, provider, pin)
    if win is not None:
        return win
    if windows is None:
        windows = load_windows()
    best: tuple[int, int | None] = (-1, None)
    for prefix, win_ in windows.items():
        if model == prefix or model.startswith(prefix + "-"):
            if len(prefix) > best[0]:
                best = (len(prefix), win_)
    return best[1]


def format_footer(
    totals_in: int,
    totals_out: int,
    last_usage: Usage | None,
    model: str,
    latency_ms: float | None,
    provider: object | None = None,
    pin: str | None = None,
) -> str:
    """Build footer string per spec segment rules."""
    # ↑↓ cumulative
    parts: list[str] = [f"↑{totals_in} ↓{totals_out}"]
    # ctx segment
    if last_usage is None:
        parts.append("ctx —")
    else:
        ctx_tokens = int(last_usage.get("input_tokens", 0)) + int(
            last_usage.get("output_tokens", 0)
        )
        win = context_window_for(model, provider=provider, pin=pin)
        if win is None:
            parts.append(f"ctx {ctx_tokens}tok")
        else:
            pct = (ctx_tokens / win * 100) if win else 0
            parts.append(f"ctx {ctx_tokens}tok / {win} ({pct:.1f}%)")
    footer = " · ".join(parts)
    if latency_ms is not None:
        footer += f" · {int(latency_ms)}ms"
    return footer


def dim(text: str) -> str:
    return f"\x1b[2m{text}\x1b[0m"
