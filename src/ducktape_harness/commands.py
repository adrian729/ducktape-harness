"""Slash commands for the REPL.

WHY: each command is a pure function on Session so REPL dispatch stays
trivial and commands are testable without I/O loops.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from ducktape_harness.config import COMPACT_KEEP_TAIL, IMAGE_MAX_BYTES, IMAGE_TYPES


def cmd_clear(session: Any) -> str:
    session.reset()
    return "[cleared]"


def cmd_help() -> str:
    return (
        "Commands:\n"
        "  /model                   — list providers and their models\n"
        "  /model <name>            — switch model by bare id\n"
        "  /model <provider> <name> — switch and pin that provider (names may contain ':')\n"
        "  /model <provider>        — list one provider's models\n"
        "  /clear                   — clear conversation (keeps totals & permissions)\n"
        "  /compact [note]          — summarize older turns, keeps last ~3 exchanges + totals; keeps tool pairing\n"
        "  /export [path]           — save transcript as markdown, images as metadata only\n"
        "  /attach <path>           — send an image (png/jpg/gif/webp) with the next message\n"
        "  /detach                  — drop staged images\n"
        "  /usage                   — show token totals and context usage\n"
        "  /help                    — this help\n"
        "  /quit                    — exit (or Ctrl+D)\n"
        "\n"
        "Permissions: writes and shell prompts take y/N/always; 'always' for bash\n"
        "remembers that exact command, other tools remember the tool name.\n"
        "read_file is auto-allowed; write_file outside the workspace is denied\n"
        "(relaunch with --no-sandbox to change).\n"
        "Context: AGENTS.md at the project root (git toplevel, else this folder)\n"
        "is appended to the system prompt each turn.\n"
        "Tools: read_file, write_file, bash, task (subagent, isolated context, shares gate+workspace, totals roll up).\n"
        "Limits: tool loop 24 iterations (12 in subagents), pause continuations 3, read 1 MB, output 100 KB.\n"
    )


def cmd_usage(session: Any) -> str:
    t = session.totals
    lines = [
        f"Totals: ↑{t.input_tokens} ↓{t.output_tokens} (cache read {t.cache_read_tokens}, write {t.cache_write_tokens})",
        f"Turns: {t.turns} (with usage: {t.turns_with_usage})",
    ]
    if session.last_usage is None:
        lines.append("ctx —")
    else:
        from ducktape_harness.render import context_window_for

        u = session.last_usage
        ctx = int(u.get("input_tokens", 0) or 0) + int(u.get("output_tokens", 0) or 0)
        win = context_window_for(session.model, provider=session.provider, pin=session.pin)
        tail = f" / {win} ({ctx / win * 100:.1f}%)" if win else ""
        lines.append(
            f"ctx {ctx}tok{tail} (last turn: in {u.get('input_tokens', 0)} out {u.get('output_tokens', 0)})"
        )
        lines.append(f"usage reported in {t.turns_with_usage} of {t.turns} turns")
    return "\n".join(lines)


def cmd_attach(session: Any, arg: str | None) -> str:
    """Stage an image for the next user message (feature 9, provider-ready).

    WHY fail-fast at attach: a bad path/type/size surfaces as a user error
    now, not as an opaque is_error tool round several turns later.
    """
    if not arg or not arg.strip():
        return "[error] usage: /attach <path>"
    raw = arg.strip().strip("\"'")
    path = Path(os.path.expanduser(raw))
    media = IMAGE_TYPES.get(path.suffix.lower())
    if media is None:
        ext = path.suffix or "(no extension)"
        return f"[error] unsupported image type {ext} — use png/jpg/jpeg/gif/webp"
    try:
        data = path.read_bytes()
    except OSError as e:
        return f"[error] {e}"
    if not data:
        return f"[error] {path} is empty"
    if len(data) > IMAGE_MAX_BYTES:
        return (
            f"[error] image too large ({len(data)} bytes > {IMAGE_MAX_BYTES}) — downscale it first"
        )
    block: Any = {
        "type": "image",
        "source": "base64",
        "media_type": media,
        "data": base64.b64encode(data).decode("ascii"),
    }
    session.pending_images.append(block)
    return f"[attach] staged {path.name} ({media}, {len(data)} bytes) — {len(session.pending_images)} pending"


def cmd_detach(session: Any, arg: str | None) -> str:
    n = len(session.pending_images)
    session.pending_images.clear()
    return f"[detach] cleared {n} pending image(s)" if n else "[detach] nothing pending"


def cmd_compact(session: Any, arg: str | None) -> str:
    """Summarize older turns, keep tail.

    WHY atomic: summarization may fail; we must never leave history
    half-compacted. Clean-boundary scan preserves tool pairing invariant.
    """
    provider = getattr(session, "provider", None)
    if provider is None:
        return "[compact] summarization failed: no provider configured — history untouched"
    msgs: list[dict[str, Any]] = getattr(session, "messages", [])
    n = len(msgs)
    if n <= COMPACT_KEEP_TAIL + 1:
        return "[compact] nothing worth compacting yet"

    # WHY scan earlier: naive split may land inside a tool_use/tool_result pair
    b = n - COMPACT_KEEP_TAIL
    if b < 1:
        b = 1
    if b >= n:
        b = n - 1

    def _clean(idx: int) -> bool:
        if idx <= 0 or idx >= n:
            return False
        cur = msgs[idx]
        prev = msgs[idx - 1]
        if cur.get("role") != "user" or prev.get("role") != "assistant":
            return False
        for blk in prev.get("content", []) or []:
            if blk.get("type") == "tool_use":
                return False
        for blk in cur.get("content", []) or []:
            if blk.get("type") == "tool_result":
                return False
        return True

    while b > 0 and not _clean(b):
        b -= 1
    if b <= 0 or not _clean(b):
        return "[compact] no safe split found — /clear instead"

    # Build transcript of msgs[:b]
    lines: list[str] = []
    for msg in msgs[:b]:
        role = msg.get("role", "user")
        lines.append(f"{role}:")
        for blk in msg.get("content", []) or []:
            t = blk.get("type")
            if t == "text":
                lines.append(blk.get("text", ""))
            elif t == "thinking":
                th = blk.get("thinking", "")
                # one line: keep first line-ish, truncate to keep transcript tight
                th_one = " ".join(th.split())
                lines.append(f"(thinking) {th_one}")
            elif t == "tool_use":
                name = blk.get("name", "")
                inp = blk.get("input", {})
                try:
                    j = json.dumps(inp, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    j = str(inp)
                if len(j) > 300:
                    j = j[:300]
                lines.append(f"(call {name} {j})")
            elif t == "tool_result":
                name = blk.get("name", "")
                is_err = blk.get("is_error")
                suffix = " [error]" if is_err else ""
                content = blk.get("content", "")
                if isinstance(content, str):
                    cstr = content
                    if len(cstr) > 300:
                        cstr = cstr[:300]
                elif isinstance(content, list):
                    parts: list[str] = []
                    for sub in content:
                        st = sub.get("type")
                        if st == "text":
                            txt = sub.get("text", "")
                            if len(txt) > 300:
                                txt = txt[:300]
                            parts.append(txt)
                        elif st == "image":
                            parts.append(f"[image {sub.get('media_type', '')}]")
                        else:
                            parts.append(f"[{st}]")
                    cstr = " ".join(parts)
                    if len(cstr) > 300:
                        cstr = cstr[:300]
                else:
                    cstr = str(content)[:300]
                lines.append(f"(result {name}{suffix}: {cstr})")
            elif t == "image":
                media = blk.get("media_type", "")
                data = blk.get("data", "")
                # WHY metadata only: transcript must not carry base64
                lines.append(f"[image {media} ~{len(data)} b64 chars]")
            elif t == "document":
                media = blk.get("media_type", "")
                data = blk.get("data", "")
                lines.append(f"[document {media} ~{len(data)} b64 chars]")
            else:
                lines.append(f"[{t}]")
    transcript = "\n".join(lines)

    from ducktape_harness.system_prompt import SUMMARY_PROMPT

    note = ""
    if arg and arg.strip():
        note = f"\n\nFocus: {arg.strip()[:500]}"
    prompt_text = SUMMARY_PROMPT + note + "\n\nTranscript:\n" + transcript

    try:
        resp: dict[str, Any] = provider.chat(
            session.model,
            [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
            provider=getattr(session, "pin", None),
        )
    except KeyboardInterrupt:
        return "[compact] interrupted — history untouched"
    except Exception as e:
        return f"[compact] summarization failed: {e} — history untouched"

    # WHY collect only text blocks: summary is prose, other blocks are not expected
    parts: list[str] = []
    for blk in resp.get("content", []) or []:
        if blk.get("type") == "text":
            txt = blk.get("text", "")
            if txt:
                parts.append(txt)
    summary_text = " ".join(parts).strip()
    if not summary_text:
        return "[compact] summarization failed: empty summary — history untouched"

    # atomic mutation.
    # WHY merge, not prepend: clean(b) guarantees tail[0] is user — a second
    # user message would break Claude's role-alternation rule outright.
    tail = list(msgs[b:])
    summary_block: dict[str, Any] = {
        "type": "text",
        "text": "[Compacted conversation summary]\n" + summary_text,
    }
    if tail and tail[0].get("role") == "user":
        tail[0] = {**tail[0], "content": [summary_block, *tail[0]["content"]]}
    else:  # defensive: shouldn't happen given clean(b)
        tail = [{"role": "user", "content": [summary_block]}, *tail]
    session.messages = tail
    session.last_usage = None
    return f"[compact] summarized {b} messages (kept last {len(tail)})"


def cmd_export(session: Any, arg: str | None) -> str:
    """Save transcript as markdown.

    WHY metadata-only for images: base64 would bloat the export and leak
    binary data into a text file.
    """
    raw = arg.strip() if arg and arg.strip() else ""
    if raw:
        raw_exp = os.path.expanduser(raw.strip().strip("\"'"))
        path = Path(raw_exp)
        # if relative, resolve against cwd
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
    else:
        name = f"ducktape-session-{time.strftime('%Y%m%d-%H%M%S')}.md"
        path = Path(os.getcwd()) / name

    if path.exists():
        return f"[error] {path} exists — choose another"

    # Build markdown
    model = getattr(session, "model", "unknown")
    pin = getattr(session, "pin", None)
    pinned = f" pinned {pin}" if pin else ""
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    t = getattr(session, "totals", None)
    if t is not None:
        totals_line = (
            f"Totals: ↑{t.input_tokens} ↓{t.output_tokens} "
            f"(cache read {t.cache_read_tokens}, write {t.cache_write_tokens}) "
            f"· turns {t.turns} (with usage {t.turns_with_usage})"
        )
    else:
        totals_line = "Totals: —"

    out_lines: list[str] = []
    out_lines.append(
        f"> warning: this file is a verbatim transcript — tool outputs may contain secrets that were approved into the session; store and share accordingly.\n\n# ducktape session — {model}{pinned} {iso}"
    )
    out_lines.append("")
    out_lines.append(totals_line)
    out_lines.append("")

    msgs: list[dict[str, Any]] = getattr(session, "messages", [])
    for msg in msgs:
        role = msg.get("role", "user")
        out_lines.append(f"## {role}")
        out_lines.append("")
        for blk in msg.get("content", []) or []:
            btype = blk.get("type")
            if btype == "text":
                out_lines.append(blk.get("text", ""))
                out_lines.append("")
            elif btype == "thinking":
                th = blk.get("thinking", "")[:500]
                out_lines.append(f"> thinking: {th}")
                out_lines.append("")
            elif btype == "tool_use":
                name = blk.get("name", "")
                inp = blk.get("input", {})
                try:
                    j = json.dumps(inp, ensure_ascii=False, indent=2)
                except Exception:
                    j = str(inp)
                if len(j) > 1000:
                    j = j[:1000] + "\n…[truncated]"
                out_lines.append(f"call `{name}`:")
                out_lines.append("```json")
                out_lines.append(j)
                out_lines.append("```")
                out_lines.append("")
            elif btype == "tool_result":
                name = blk.get("name", "")
                is_err = " [error]" if blk.get("is_error") else ""
                out_lines.append(f"result {name}{is_err}:")
                content = blk.get("content", "")
                if isinstance(content, str):
                    txt = content
                    trunc = False
                    if len(txt) > 2000:
                        txt = txt[:2000]
                        trunc = True
                    out_lines.append(txt)
                    if trunc:
                        out_lines.append("…[truncated]")
                    out_lines.append("")
                elif isinstance(content, list):
                    for sub in content:
                        st = sub.get("type")
                        if st == "text":
                            txt = sub.get("text", "")
                            trunc = False
                            if len(txt) > 2000:
                                txt = txt[:2000]
                                trunc = True
                            out_lines.append(txt)
                            if trunc:
                                out_lines.append("…[truncated]")
                        elif st == "image":
                            media = sub.get("media_type", "")
                            data = sub.get("data", "")
                            approx = int(len(data) * 0.75) if data else 0
                            out_lines.append(f"[image {media}, {approx} bytes — b64 omitted]")
                        else:
                            out_lines.append(f"[{st}]")
                    out_lines.append("")
                else:
                    out_lines.append(str(content))
                    out_lines.append("")
            elif btype == "image":
                media = blk.get("media_type", "")
                data = blk.get("data", "")
                approx = int(len(data) * 0.75) if data else 0
                out_lines.append(f"[image {media}, {approx} bytes — b64 omitted]")
                out_lines.append("")
            elif btype == "document":
                media = blk.get("media_type", "")
                data = blk.get("data", "")
                approx = int(len(data) * 0.75) if data else 0
                out_lines.append(f"[document {media}, {approx} bytes — b64 omitted]")
                out_lines.append("")
            else:
                out_lines.append(f"[{btype}]")
                out_lines.append("")

    content_str = "\n".join(out_lines)
    try:
        # Ensure parent exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # WHY "x": the exists() pre-check is friendly UX, O_EXCL is the
        # atomic guard — two exports in the same second can't clobber each other
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(content_str + "\n")
    except FileExistsError:
        return f"[error] {path} exists — choose another"
    except OSError as e:
        return f"[error] {e}"
    return f"[export] wrote {path} ({len(msgs)} messages)"


def cmd_model(session: Any, arg: str | None) -> str:
    """List or switch model.

    WHY spaces between provider and name: model ids contain ':' themselves
    (ollama tags like "qwen3:8b"), so a colon separator would be ambiguous.
    """
    provider = getattr(session, "provider", None)
    if provider is None:
        return "[error] no provider configured"
    if arg is None or arg.strip() == "":
        print("checking providers…")
        try:
            models = provider.models()
        except Exception as e:
            return f"[error] {e}"
        # models is dict[str, list[str]]
        lines = [_provider_line(p, ml) for p, ml in sorted(models.items())]
        lines.append(_current_line(session))
        lines.append("switch: /model <name> | /model <provider> <name> | /model <provider>")
        return "\n".join(lines) if lines else "(no models available)"
    words = arg.strip().split(maxsplit=1)
    first = words[0]
    pin = getattr(session, "pin", None)
    print("checking providers…")
    try:
        models = provider.models()
    except Exception as e:
        return f"[error] {e}"
    if first in models or first in _configured_providers(provider, models):
        if len(words) == 1:
            # /model <provider> — list just that provider
            ml = models.get(first)
            head = _provider_line(first, ml) if ml is not None else f"{first}: (unavailable)"
            return f"{head}\n{_current_line(session)}\nswitch: /model {first} <name>"
        name = words[1].strip()
        avail = models.get(first)
        if avail is None:
            return f"[error] provider unavailable: {first!r} — keeping {session.model}"
        if name not in avail:
            return f"[error] model {name!r} not in provider {first!r} — keeping {session.model}"
        session.model = name
        session.pin = first
        return f"[model] now {name} (provider {first})"
    if len(words) == 2:
        # model ids never contain spaces in practice; a two-word arg whose head is
        # not a provider means the provider name was mistyped
        return f"[error] unknown provider {first!r} — try /model to list — keeping {session.model}"
    target = arg.strip()
    if pin is not None:
        # pinned → validate against models()[pin] only
        if pin not in models:
            return f"[error] provider unavailable: {pin!r} — keeping {session.model}"
        if target not in models[pin]:
            return f"[error] model {target!r} not in provider {pin!r} — keeping {session.model}"
        session.model = target
        return f"[model] now {target} (pinned to {pin})"
    # unpinned → union, but a same-id-across-providers hit must be disambiguated
    matches = sorted(p for p, lst in models.items() if target in lst)
    if not matches:
        return f"[error] model {target!r} not found — keeping {session.model}"
    if len(matches) > 1:
        return (
            f"[error] model {target!r} served by {', '.join(matches)} — "
            f"pick one: /model {matches[0]} {target}"
        )
    session.model = target
    return f"[model] now {target}"


def _provider_line(prov: str, mlist: list[str]) -> str:
    return f"{prov}: {', '.join(sorted(mlist)) if mlist else '(no models)'}"


def _current_line(session: Any) -> str:
    return "current: " + session.model + (f" (pinned to {session.pin})" if session.pin else "")


def _configured_providers(provider: Any, models: dict[str, list[str]]) -> set[str]:
    """Provider names known to the Provider even when currently unavailable."""
    try:
        return set(provider.providers())
    except Exception:
        return set(models)
