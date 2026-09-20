# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-19

### Added

- REPL (external loop) with streaming replies, dimmed thinking, readline history, and interrupt handling.
- Agent loop (internal loop): tool round-trips with a hard pairing invariant (`tool_use` ↔ `tool_result` on every path), iteration cap, `pause_turn` continuations, `max_tokens` truncation handling via the provider's `truncated` marker with a `block_stop` fallback.
- Permission gate: per-call `y/N/always`, unified-diff preview for `write_file`, full-command preview for `bash`, per-command `always` keys for bash, bounded `read_file` auto-allow (workspace-scoped, secret-named files always prompt), workspace write sandbox with post-approval recheck, `--no-sandbox` opt-out.
- Tools: `read_file` (offset/limit windows, size caps, `image: true` → image content block), `write_file`, `bash`, `task` (subagents: isolated context, tool allowlist, nested-task guard, shared gate/sandbox, token roll-up).
- Slash commands: `/model` (list/switch, provider-prefixed form, pin-aware validation), `/clear`, `/help`, `/usage`, `/compact` (pair-safe history summarization), `/export` (markdown transcript, no base64, atomic no-clobber), `/quit`.
- Usage & context tracking: cumulative `↑/↓` totals, latest-turn `ctx` with context window from provider `model_info` (cached) and a user-extensible fallback table; pre-send budget advisory at ≥85%.
- Image input: `/attach` / `/detach` staging base64 `ImageBlock`s into the next user message.
- `AGENTS.md` project context appended to the system prompt from the git-root (or launch dir), re-read every turn, bounded search and size.
- Provider gap reporting: six defects reported and since fixed upstream; one (cache breakpoints) remains open — see `PROVIDER_GAPS.md`.
- CI: lint/format (`ruff`), typing (`ty`, hash-pinned) over `src` and `tests`, `unittest` matrix on Python 3.12/3.13/3.14 with `ResourceWarning` as error.

### Process

- The implementation plan and code each passed iterative audit loops (reviewer + independent security confirmation); invariants, interrupt matrix, and docs parity are covered by 145 offline tests requiring no API keys.
