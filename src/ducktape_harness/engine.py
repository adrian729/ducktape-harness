"""Internal loop — run_turn and helpers.

WHY: single place for the 7-step engine from PLAN.md; every path keeps
the tool_result pairing invariant and caps ordering exactly as specified.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from ducktape_provider.types import ToolResultBlock

from ducktape_harness.config import (
    BUDGET_WARN_FRACTION,
    CONTENT_EVENT_TYPES,
    IMAGE_EST_TOKENS,
    MAX_TOOL_ITERATIONS,
    OUTPUT_CAP,
    PAUSE_CAP,
    POLICY_AUTO_ALLOW,
    SANDBOX_DENY_TOOLS,
    SECRET_PATTERNS,
    SUB_MAX_ITERATIONS,
)
from ducktape_harness.render import context_window_for, dim
from ducktape_harness.system_prompt import build_system_prompt
from ducktape_harness.tools.base import REGISTRY, TASK_SPEC

try:
    from ducktape_provider import (
        APIError,
        AuthError,
        ContextOverflowError,
        MalformedResponseError,
        RateLimitError,
        RequestTimeoutError,
        ServerError,
        UnsupportedBlockError,
    )
except ImportError:  # pragma: no cover
    from ducktape_provider.types import (  # type: ignore[no-redef]
        APIError,
        AuthError,
        ContextOverflowError,
        MalformedResponseError,
        RateLimitError,
        RequestTimeoutError,
        ServerError,
        UnsupportedBlockError,
    )


def _tool_specs(tool_names: set[str] | None = None, is_sub: bool = False) -> list[dict[str, Any]]:
    specs = []
    for tool in REGISTRY.values():
        if tool_names is not None and tool.name not in tool_names:
            continue
        specs.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        )
    # task is parent-only; sub sessions never see it via tool_names exclusion + nested guard
    if not is_sub:
        # Only add task if parent's tool_names allows it (None=all, or explicitly includes task)
        if tool_names is None or "task" in tool_names:
            specs.append(TASK_SPEC)
    return specs


def _truncate_output(text: str) -> str:
    b = text.encode("utf-8")
    if len(b) > OUTPUT_CAP:
        return b[:OUTPUT_CAP].decode("utf-8", errors="ignore") + "\n[truncated]"
    return text


def _result(
    tid: str, tname: str, content: str | list[Any], is_error: bool = False
) -> ToolResultBlock:
    """One ToolResultBlock with the id+name copied off the ToolUseBlock (types.py requires both)."""
    blk: ToolResultBlock = {
        "type": "tool_result",
        "tool_use_id": tid,
        "name": tname,
        "content": content,
    }
    if is_error:
        blk["is_error"] = True  # type: ignore[typeddict-unknown-key]
    return blk


def _tool_preview(tool: Any, tinput: dict[str, Any], cwd: str) -> str:
    fn = getattr(tool, "preview", None)
    if callable(fn):
        try:
            out = fn(tinput, cwd)
            return str(out) if out else "(no arguments)"
        except Exception:
            pass
    # cap: preview is display text; never let a giant args dict spam the prompt
    text = str(tinput) if tinput else "(no arguments)"
    return text if len(text) <= 500 else text[:500] + " …[truncated]"


def _within(path: str, root: str) -> bool:
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    return rp == rr or rp.startswith(rr + os.sep)


def _sensitive_read(tname: str, tinput: dict[str, Any], cwd: str) -> bool:
    if tname != "read_file":
        return False
    path = tinput.get("path")
    if not isinstance(path, str):
        return False
    import fnmatch

    # WHY match requested basename AND resolved one: an in-workspace symlink
    # innocent.txt -> .env must not read as innocent; normalization stops
    # " .env ", ".env/" dodging; lowercase-vs-lowercase stops .ENV/.Id_Rsa.
    cleaned = os.path.normpath(path.strip())
    candidates = {os.path.basename(cleaned)}
    joined = cleaned if os.path.isabs(cleaned) else os.path.join(cwd, cleaned)
    # realpath(strict=False) never raises; broken links resolve to their own path
    candidates.add(os.path.basename(os.path.realpath(joined)))
    lowered = {c.lower() for c in candidates if c}
    return any(fnmatch.fnmatchcase(c, pat.lower()) for c in lowered for pat in SECRET_PATTERNS)


def _sandbox_outside(
    tool: Any, sandbox_root: str | None, tinput: dict[str, Any], cwd: str
) -> str | None:
    """Resolved path if the tool's target escapes the sandbox, else None."""
    if not sandbox_root:
        return None
    st = getattr(tool, "sandbox_target", None)
    if not callable(st):
        return None
    try:
        target = st(tinput, cwd)
    except Exception:
        return None
    if isinstance(target, str) and target and not _within(target, sandbox_root):
        return target
    return None


def _gate_and_run(
    session: Any,
    tid: str,
    tname: str,
    tinput: dict[str, Any],
    tool_names: set[str] | None = None,
    is_sub: bool = False,
) -> ToolResultBlock:
    """Policy → gate → run one tool call. Raises KI/EOFError/Exception for the caller's stub logic.

    WHY this order: cheap rejection (unknown tool, out-of-workspace write path)
    happens before the prompt so the user never sees a question about a call
    that would be refused anyway. For subagents, tool_names exclusion and
    nested-task guard fire before the gate as well.
    """
    cwd = os.getcwd()
    gate_obj = session.gate
    sandbox_root = session.sandbox_root

    # Task special-case — must be before generic unknown/tool_names checks so
    # nested-task guard takes precedence over "tool not available" for subagents.
    if tname == "task":
        if is_sub:
            return _result(tid, tname, "nested task not allowed", is_error=True)
        # Parent task: gate normally with preview f"{description} — {prompt[:300]}"
        description = tinput.get("description", "")
        prompt = tinput.get("prompt", "")
        if not isinstance(description, str):
            description = str(description)
        if not isinstance(prompt, str):
            prompt = str(prompt)
        preview = f"{description} — {prompt[:300]}"
        key = preview
        # WHY ask outside the try: a KI/EOFError at the task prompt must
        # propagate so the 5a batch-interrupt path denies remaining blocks —
        # catching it here would silently keep prompting past the interrupt.
        if (
            not gate_obj.is_always_allowed(tname, key)
            and gate_obj.ask(tname, preview, key, multiline=False) == "deny"
        ):
            return _result(tid, tname, "user denied", is_error=True)
        # Execute subagent path (lazy import to dodge cycle)
        try:
            from ducktape_harness.tools.subagent import run_subagent  # noqa: WPS433

            requested = tinput.get("tools")
            req_list: list[str] | None = None
            if isinstance(requested, list):
                req_list = [str(x) for x in requested if isinstance(x, (str, bytes))]
            # malformed tools param falls through as None (= all allowed);
            # unknown-name surfacing lives in run_subagent
            out = run_subagent(session, description, prompt, req_list)
        except Exception as e:
            return _result(tid, tname, _truncate_output(str(e)), is_error=True)
        text = out
        content: Any = _truncate_output(text)
        return _result(tid, tname, content)

    tool = REGISTRY.get(tname)
    if tool is None:
        return _result(tid, tname, f"unknown tool {tname}", is_error=True)
    if tool_names is not None and tname not in tool_names:
        return _result(tid, tname, "tool not available to this subagent", is_error=True)
    outside = _sandbox_outside(tool, sandbox_root, tinput, cwd)
    if outside and tname in SANDBOX_DENY_TOOLS:
        return _result(
            tid,
            tname,
            f"denied: path outside workspace — {outside} is not under {sandbox_root} "
            "(relaunch with --no-sandbox to allow)",
            is_error=True,
        )
    # WHY auto-allow is bounded: "reads are safe" covers mutability, not
    # confidentiality — out-of-workspace and secret-named reads still prompt.
    auto = (
        tname in POLICY_AUTO_ALLOW and outside is None and not _sensitive_read(tname, tinput, cwd)
    )
    if not auto:
        preview = _tool_preview(tool, tinput, cwd)
        cmd = tinput.get("command")
        key = cmd if tname == "bash" and isinstance(cmd, str) else preview
        ask_multiline = bool(getattr(tool, "preview_multiline", False))
        if (
            not gate_obj.is_always_allowed(tname, key)
            and gate_obj.ask(tname, preview, key, multiline=ask_multiline) == "deny"
        ):
            return _result(tid, tname, "user denied", is_error=True)
    if _sandbox_outside(tool, sandbox_root, tinput, cwd) and tname in SANDBOX_DENY_TOOLS:
        # symlink swapped toward an escape during the prompt wait
        return _result(
            tid, tname, "denied: path resolves outside workspace at execution time", is_error=True
        )
    out = tool.run(tinput, cwd)
    # tools may return text (truncated) or content blocks (e.g. images,
    # enabled by the provider widening ToolResultBlock.content)
    content_any: Any = out if isinstance(out, list) else _truncate_output(str(out))
    return _result(tid, tname, content_any)


def _estimate_tokens(session: Any, user_text: str) -> float:
    """Heuristic token estimate for budget warning.

    WHY separate: keeps run_turn readable; estimate never raises.
    """
    lu = getattr(session, "last_usage", None)
    if isinstance(lu, dict) and lu.get("input_tokens") is not None:
        try:
            it = int(lu.get("input_tokens", 0) or 0)
            ot = int(lu.get("output_tokens", 0) or 0)
            return it + ot + len(user_text) / 4
        except Exception:
            pass
    # Fallback: system + all text blocks + image heuristic
    try:
        system = build_system_prompt(os.getcwd())
    except Exception:
        system = ""
    est = len(system) / 4
    for msg in getattr(session, "messages", []) or []:
        for blk in msg.get("content", []) or []:
            t = blk.get("type")
            if t == "text":
                est += len(blk.get("text", "")) / 4
            elif t == "thinking":
                est += len(blk.get("thinking", "")) / 4
            elif t == "tool_use":
                try:
                    import json as _json

                    j = _json.dumps(blk.get("input", {}), ensure_ascii=False)
                    est += len(j) / 4
                except Exception:
                    est += 10
            elif t == "tool_result":
                c = blk.get("content")
                if isinstance(c, str):
                    est += len(c) / 4
                elif isinstance(c, list):
                    for sub in c:
                        if sub.get("type") == "text":
                            est += len(sub.get("text", "")) / 4
                        elif sub.get("type") == "image":
                            est += IMAGE_EST_TOKENS
                        else:
                            est += 10
                elif c is not None:
                    est += len(str(c)) / 4
            elif t == "image":
                est += IMAGE_EST_TOKENS
            elif t == "document":
                est += IMAGE_EST_TOKENS
    return est


def run_turn(
    session: Any,
    user_text: str,
    *,
    tool_names: set[str] | None = None,
    is_sub: bool = False,
    quiet: bool = False,
) -> str:
    """Execute one user turn through the agent loop.

    WHY: mirrors Engine steps 1-7 in PLAN.md exactly, including the
    content_seen retry gate and block_stop max_tokens discrimination.
    """
    # Effective quiet: is_sub implies quiet unless caller overrides explicitly.
    effective_quiet = quiet or is_sub

    # Step 1: add user text with alternation merge
    session.last_interrupted = False
    provider = session.provider
    if provider is None:
        # WHY before consuming attachments: a failed setup must not swallow
        # staged /attach images or pollute history.
        print("[error] no provider configured")
        return ""

    # Step 1: add user text with alternation merge; /attach images ride first
    attachments = list(session.pending_images)
    session.pending_images.clear()
    new_blocks: list[Any] = attachments + [{"type": "text", "text": user_text}]
    if session.messages and session.messages[-1].get("role") == "user":
        session.messages[-1]["content"].extend(new_blocks)
    else:
        session.messages.append({"role": "user", "content": new_blocks})

    iterations = 0
    pause_rounds = 0
    retried = False  # one automatic retry per run_turn, shared across error families

    specs = _tool_specs(tool_names, is_sub=is_sub)

    warned = False

    while True:
        # WHY pre-send budget warning: advise /compact before the vendor rejects
        # Skip for subagents (quiet).
        if not is_sub and not warned:
            try:
                win = context_window_for(
                    session.model, provider=session.provider, pin=getattr(session, "pin", None)
                )
            except Exception:
                win = None
            if win:
                est = _estimate_tokens(session, user_text)
                if est / win >= BUDGET_WARN_FRACTION:
                    print(
                        f"[context ~{int(100 * est / win)}% of {win} — /compact or /clear advised]"
                    )
                    warned = True
        content_seen = False
        block_stop_indices: set[int] = set()
        response: dict[str, Any] | None = None

        # Step 2: call + render wrapped in try for eager+lazy errors
        try:
            with contextlib.closing(
                provider.stream_chat(
                    session.model,
                    session.messages,
                    system=build_system_prompt(os.getcwd()),
                    tools=specs,
                    provider=session.pin,
                )
            ) as events:
                try:
                    for event in events:
                        et = event.get("type")
                        if et in CONTENT_EVENT_TYPES:
                            content_seen = True
                        if et == "text_delta":
                            if not effective_quiet:
                                print(event.get("text", ""), end="", flush=True)
                        elif et == "thinking_delta":
                            if not effective_quiet:
                                print(dim(event.get("thinking", "")), end="", flush=True)
                        elif et == "block_stop":
                            try:
                                block_stop_indices.add(int(event.get("index", -1)))
                            except (ValueError, TypeError):
                                pass
                        elif et == "message_stop":
                            response = event.get("response")  # type: ignore[assignment]
                            break
                        # tool_use_start / tool_use_delta already flip content_seen, no render
                except KeyboardInterrupt:
                    # provider streams expose cancel() to abort the HTTP request
                    # now, not just release it at the next yield — gap #3 fixed.
                    cancel = getattr(events, "cancel", None)
                    if callable(cancel):
                        try:
                            cancel()
                        except Exception:
                            pass
                    raise
                # ensure newline after streaming if we saw content
                if content_seen and not effective_quiet:
                    print()
        except KeyboardInterrupt:
            session.last_interrupted = True
            print("\n[interrupted]")
            return ""
        except (RateLimitError, ServerError) as e:
            if content_seen:
                print(f"\n[error] {e}")
                return ""
            if retried:
                print(f"[error] {e}")
                return ""
            retried = True
            retry_after = getattr(e, "retry_after", None)
            delay = min(retry_after, 60) if retry_after is not None else 5
            try:
                time.sleep(delay)  # type: ignore[arg-type]
            except KeyboardInterrupt:
                session.last_interrupted = True
                print("\n[interrupted]")
                return ""
            continue
        except RequestTimeoutError as e:
            if content_seen:
                print(f"\n[error] {e}")
                return ""
            if retried:
                print(f"[error] {e}")
                return ""
            retried = True
            continue
        except ContextOverflowError as e:
            print(f"{e} — try /clear or /model with a larger context window")
            return ""
        except AuthError as e:
            print(f"[error] {e} — check API key env var")
            return ""
        except MalformedResponseError as e:
            print(f"[error] {e}")
            return ""
        except APIError as e:
            # 413 arrives as ContextOverflowError since provider b0ad0bb (gap #6)
            print(f"[error] {e}")
            return ""
        except UnsupportedBlockError as e:
            print(f"{e} — try /clear")
            return ""
        except (KeyError, ValueError, TypeError) as e:
            print(f"[error] {e}")
            return ""

        if response is None:
            print("[error] no response")
            return ""

        # Step 3: append assistant turn with pause-merge and empty placeholder
        raw_blocks = response.get("content") or []
        was_empty_placeholder = not raw_blocks
        blocks = raw_blocks if raw_blocks else [{"type": "text", "text": "[no text response]"}]
        # Merge into trailing assistant if present (pause continuation)
        if session.messages and session.messages[-1].get("role") == "assistant":
            session.messages[-1]["content"].extend(list(blocks))
        else:
            session.messages.append({"role": "assistant", "content": list(blocks)})

        # Step 4: usage — use .get (NotRequired) and store latency via .get
        usage = response.get("usage")
        lat = response.get("latency_ms")
        session.last_latency = float(lat) if isinstance(lat, (int, float)) else None  # type: ignore[arg-type]
        if usage is not None:
            session.last_usage = usage
            session.totals.input_tokens += int(usage.get("input_tokens", 0) or 0)
            session.totals.output_tokens += int(usage.get("output_tokens", 0) or 0)
            # cache tokens if present
            if "cache_read_tokens" in usage and usage["cache_read_tokens"] is not None:
                session.totals.cache_read_tokens += int(usage["cache_read_tokens"])
            if "cache_write_tokens" in usage and usage["cache_write_tokens"] is not None:
                session.totals.cache_write_tokens += int(usage["cache_write_tokens"])
            session.totals.turns_with_usage += 1
        session.totals.turns += 1

        stop_reason = response.get("stop_reason")

        # Step 5 routing
        if stop_reason == "tool_use":
            tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
            # If no tool_use blocks but stop_reason is tool_use, treat as final
            if not tool_blocks:
                text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if not text:
                    print("[no text response]")
                return text or "[no text response]"

            results: list[ToolResultBlock] = []
            interrupted = False

            for idx, tb in enumerate(tool_blocks):
                tid = tb.get("id", f"tool_{idx}")
                tname = tb.get("name", "")
                tinput = tb.get("input")
                if not isinstance(tinput, dict):
                    tinput = {}
                try:
                    results.append(_gate_and_run(session, tid, tname, tinput, tool_names, is_sub))
                except (KeyboardInterrupt, EOFError):
                    # this block interrupted, remainder batch-interrupted
                    results.append(_result(tid, tname, "interrupted", is_error=True))
                    for rem in tool_blocks[idx + 1 :]:
                        results.append(
                            _result(
                                rem.get("id", ""),
                                rem.get("name", ""),
                                "denied (batch interrupted)",
                                is_error=True,
                            )
                        )
                    interrupted = True
                    break
                except Exception as e:
                    # tool exception (incl. ToolError) -> is_error with message
                    results.append(_result(tid, tname, _truncate_output(str(e)), is_error=True))
            # Append single user message with all results — pairing invariant
            # Wrap append in KI/EOF guard
            try:
                session.messages.append({"role": "user", "content": list(results)})  # type: ignore[typeddict-item]
            except (KeyboardInterrupt, EOFError):
                # Synthesize full stub set if append interrupted — ensure no dangling tool_use
                # If we already appended? This except is for the append itself; per spec synthesize full set
                # If results already built, ensure they're appended even on interrupt
                # Try to append synthesized results if not already
                try:
                    # If messages length didn't grow, append stub
                    if not (session.messages and session.messages[-1].get("role") == "user"):
                        stub: list[ToolResultBlock] = []
                        for tb in tool_blocks:
                            stub.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tb.get("id", ""),
                                    "name": tb.get("name", ""),
                                    "content": "interrupted",
                                    "is_error": True,
                                }
                            )
                        session.messages.append({"role": "user", "content": stub})  # type: ignore[typeddict-item]
                except Exception:
                    pass
                print("\n[interrupted]")
                session.last_interrupted = True
                return ""

            if interrupted:
                session.last_interrupted = True
                print("[interrupted]")
                return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")

            iterations += 1
            cap = SUB_MAX_ITERATIONS if is_sub else MAX_TOOL_ITERATIONS
            if iterations >= cap:
                print("[stopped: tool loop limit]")
                return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")

            # loop to step 2 for next model turn
            continue

        elif stop_reason == "pause_turn":
            pause_rounds += 1
            if pause_rounds > PAUSE_CAP:
                return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            continue

        elif stop_reason == "max_tokens":
            # WHY: pair blocks with their stream index by POSITION (enumerate), not
            # blocks.index(tb) — equality would map two identical tool_use blocks to
            # the same index and let a truncated duplicate execute as "completed".
            tool_pos = [(i, b) for i, b in enumerate(blocks) if b.get("type") == "tool_use"]
            if tool_pos:
                results2: list[ToolResultBlock] = []
                mt_interrupted = False
                for pos, (orig_idx, tb) in enumerate(tool_pos):
                    tid = tb.get("id", "")
                    tname = tb.get("name", "")
                    tinput = tb.get("input")
                    if not isinstance(tinput, dict):
                        tinput = {}
                    # provider marks unparseable args with truncated= since
                    # b0ad0bb (gap #5); block_stop stays the conservative
                    # fallback for third-party adapters that omit the flag.
                    if orig_idx not in block_stop_indices or tb.get("truncated"):
                        # incomplete args → never execute
                        results2.append(
                            _result(tid, tname, "truncated by max_tokens", is_error=True)
                        )
                        continue
                    try:
                        results2.append(
                            _gate_and_run(session, tid, tname, tinput, tool_names, is_sub)
                        )
                    except (KeyboardInterrupt, EOFError):
                        results2.append(_result(tid, tname, "interrupted", is_error=True))
                        for rem_idx, rem in tool_pos[pos + 1 :]:
                            reason = (
                                "denied (batch interrupted)"
                                if rem_idx in block_stop_indices
                                else "truncated by max_tokens"
                            )
                            results2.append(
                                _result(
                                    rem.get("id", ""),
                                    rem.get("name", ""),
                                    reason,
                                    is_error=True,
                                )
                            )
                        mt_interrupted = True
                        break
                    except Exception as e:
                        results2.append(
                            _result(tid, tname, _truncate_output(str(e)), is_error=True)
                        )
                # Append results and END turn (no re-loop)
                try:
                    session.messages.append({"role": "user", "content": list(results2)})  # type: ignore[typeddict-item]
                except (KeyboardInterrupt, EOFError):
                    # Synthesize stub if append fails
                    try:
                        if not (session.messages and session.messages[-1].get("role") == "user"):
                            stub2 = [
                                _result(
                                    tb.get("id", ""),
                                    tb.get("name", ""),
                                    "interrupted",
                                    is_error=True,
                                )
                                for _i, tb in tool_pos
                            ]
                            session.messages.append({"role": "user", "content": stub2})  # type: ignore[typeddict-item]
                    except Exception:
                        pass
                    print("\n[interrupted]")
                    session.last_interrupted = True
                    return ""
                if mt_interrupted:
                    print("[interrupted]")
                print("[stopped: output limit (truncated tool calls)]")
                return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")  # type: ignore[union-attr]
            else:
                print("[stopped: output limit]")
                return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")  # type: ignore[union-attr]

        else:
            # end_turn and others → final text
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if was_empty_placeholder:
                print("[no text response]")
                return "[no text response]"
            if not text:
                print("[no text response]")
            return text or "[no text response]"
