# Provider gaps report

Defects and capability gaps found in [`ducktape-provider`](https://github.com/adrian729/ducktape-provider) (`src/ducktape_provider/`) while building the harness. All six were reported upstream and fixed in provider commit `b0ad0bb` (2026-09-18) — statuses below; each entry notes what the harness now does with the fix.

## 1. ~~No model metadata~~ — RESOLVED (`Provider.model_info`)

- **Fix:** `ModelInfo {context_window, max_output_tokens}` + `Provider.model_info/async_model_info` + optional `Adapter.model_info` hook — live for Claude and Ollama, `None` for OpenAI.
- **Harness:** the footer/`/usage` window now comes from `model_info` first (cached per model+pin), falling back to the `CONTEXT_WINDOWS` table (still needed for OpenAI and third-party adapters). Residual: OpenAI exposes nothing, so table entries remain the only source there.

## 2. ~~`ToolResultBlock.content` is `str` only~~ — RESOLVED

- **Fix:** `content: str | list[TextBlock | ImageBlock]`.
- **Harness:** tools may return a list of content blocks and the engine passes it through unstringified; `read_file` gained `image: true` → returns a base64 `ImageBlock` (≤5 MB, png/jpg/jpeg/gif/webp), so the model can *see* screenshots/diagrams, not just read text.

## 3. ~~No way to abort an in-flight request~~ — RESOLVED

- **Fix:** stream iterators expose `cancel()` (thread-safe, socket shutdown; also honored before the request starts), async cancellation aborts the HTTP request, and the provider wrapper forwards `close`/`cancel`.
- **Harness:** `Ctrl+C` during a stream now calls `events.cancel()` before unwinding — the request dies immediately instead of lingering to the adapter timeout.

## 4. `usage` is `NotRequired` — RESOLVED (contract now `Usage | None`, required key)

- **Fix:** `Response.usage` is required with explicit `None` for "no data".
- **Harness:** guards kept (`.get` + `N of M turns` line) because third-party adapters may still omit the key; built-ins always answer.

## 5. ~~Truncated tool-call arguments arrived silently as `{}`~~ — RESOLVED

- **Fix:** `ToolUseBlock.truncated: NotRequired[bool]` marks args that couldn't be parsed.
- **Harness:** the `max_tokens` branch trusts the flag first; the `block_stop` position heuristic is kept as a conservative fallback for adapters predating the marker.

## 6. ~~Claude 413 escaped `ContextOverflowError`~~ — RESOLVED

- **Fix:** `errors._classify` maps `status == 413` to `ContextOverflowError`.
- **Harness:** the `APIError.status == 413` workaround was removed; overflow hints ride the canonical exception. (A bare `APIError(413)` from a nonconforming adapter now just prints generically.)

## Also noted, out of provider scope (not gaps)

- Permission gating — was and remains fully supported with nothing needed: tool calls arrive in the `Response` *before* execution, so the harness intercepts, gates, then runs.
- Pricing/costs, tokenization, conversation storage, retries, context-window *policy* beyond metadata — harness concerns; provider stays "the abstraction to interact with each API/model, no more, no less".

## 7. No way to set a cache breakpoint — OPEN (new, after b0ad0bb)

- **Where:** `system: str | None` across `Provider.chat/stream_chat` and `Adapter` (`claude.py:315-316` does `payload["system"] = system` verbatim). Anthropic prompt caching requires `system` as *blocks* with `cache_control: {"type": "ephemeral"}`; the merged `config` can't express it — any `system` key in config is overwritten by the plain string, and reserved-key validation rejects the collision anyway.
- **Impact:** long sessions re-send (and re-bill) the system + history without caching; `usage.cache_read_tokens` stays at 0. The harness reports cache fields but has no lever to trigger them.
- **Proposal (vendor-neutral):** widen `system` to `str | list[TextBlock]` with a cache flag on blocks (adapters map to their vendor's mechanism or ignore), or a `Provider(cache_system=True)` style knob. Passing a raw list off-label was considered and rejected as a hack.
- **Harness interim:** none — cost visibility only (cache totals in `/usage`).
