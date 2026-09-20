# ducktape-harness

A small agent harness on top of [ducktape-provider](https://github.com/adrian729/ducktape-provider): chat with any supported model in a REPL, let it call your files and shell through a permission gate, and keep usage/context in view.

```
› change f.txt to say goodbye
Allow tool write_file --- /home/me/proj/f.txt (current)
-hello
+goodbye? [y/N/always]: y
Changed f.txt to goodbye.
↑1412 ↓86 · ctx 1498tok / 200000 (0.7%) · 2210ms
```

## Quickstart

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```bash
# run straight from GitHub (adds the console script `ducktape`)
uv tool install git+https://github.com/adrian729/ducktape-harness
ducktape --model qwen3:8b --provider ollama-local

# or from a clone
git clone https://github.com/adrian729/ducktape-harness && cd ducktape-harness
uv sync --locked
uv run ducktape                          # defaults to claude-sonnet-4
```

Cloud providers need an API key in the environment; Ollama works as-is:

| Provider | How it becomes available |
|---|---|
| `claude` | `ANTHROPIC_API_KEY` set |
| `openai` | `OPENAI_API_KEY` set |
| `ollama-local` | Ollama running at `OLLAMA_HOST` (default `http://127.0.0.1:11434`) |

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `claude-sonnet-4` | Model id — use the bare id (`qwen3:8b`, not `ollama-local:qwen3:8b`) |
| `--provider` | auto | Pin one provider; without it the provider is matched from the model name |
| `--tool-timeout` | 30 | Seconds a `bash` command may run |
| `--http-timeout` | provider default | HTTP seconds per request, `none` to disable |
| `--no-sandbox` | off | Allow `write_file` to paths outside the launch directory (sandboxed by default) |

## In a session

Type a message and press Enter; the answer streams as it arrives (model thinking shows dimmed). Blank lines are ignored. `Ctrl+C` interrupts a running turn — the in-flight request is cancelled immediately (or cancels the current input line); `Ctrl+D` or `/quit` exits.

To show the model a screenshot or diagram: `/attach ./shot.png`, then type your question — the image is sent as base64 inside that message and stays in history like any other content (it counts toward context on later turns).

### Commands

| Command | Does |
|---|---|
| `/model` | List models per provider, plus the current model and pin |
| `/model <name>` | Switch model by bare id — a miss keeps the current model; an id served by several providers asks you to pick one; while pinned (`--provider`), only that provider's ids are accepted |
| `/model <provider> <name>` | Switch **and pin** that provider — e.g. `/model ollama-local qwen3:8b`. Provider and name are separated by a space, never a colon, because model ids contain `:` themselves; an unknown first word reports `unknown provider` |
| `/model <provider>` | List just that provider's models |
| `/clear` | Start a fresh conversation. Token totals and "always"-allowed tools stay; the system prompt is unaffected |
| `/compact [note]` | Summarize older turns, keep last ~3 exchanges + totals; keeps tool pairing |
| `/export [path]` | Save transcript as markdown, images as metadata only |
| `/attach <path>` | Send an image (png/jpg/jpeg/gif/webp, ≤5 MB, extension case-insensitive, `~` expands) with your **next** message; repeatable — staged images show as pending |
| `/detach` | Drop staged images |
| `/usage` | Session totals, last turn, context size, how many turns reported usage |
| `/help` | Commands, gate behavior, and limits |
| `/quit` | Exit (aliases: `/exit`, `/q`, `Ctrl+D`) |

### Permission gate

Before each write or shell call you see exactly what will happen — a unified diff for `write_file` (kept multi-line, capped at 40 diff lines), the full command for `bash` (collapsed to one line, display-capped) — and answer:

| You answer | Effect |
|---|---|
| `y` | Run this one call |
| anything else / Enter | Deny — the model is told "user denied" and continues |
| `always` | `bash`: never ask again for **that exact command**; other tools: for that tool name. Session-only, survives `/clear` |

`read_file` is auto-allowed **when the workspace sandbox is active (the default) and the target is inside it**, and never for secret-shaped names (`.env*`, `*.pem`, `*.key`, `id_*`, `*credentials*`, …) — those always prompt (`config.py: POLICY_AUTO_ALLOW`, `SECRET_PATTERNS`). When the sandbox is off (`--no-sandbox`, or library use without `sandbox_root`) reads are auto-allowed anywhere except secret-named ones. Note `--no-sandbox` disables the workspace boundary for *both* writes and read auto-allow. `write_file` to a path outside the launch directory is denied outright. Approval is re-checked immediately before the write, so a symlink swapped into an escape during the prompt is still denied (a theoretical microsecond window between that re-check and `open()` remains — accepted; run in workspaces you trust). Interrupting the prompt (`Ctrl+C`/`Ctrl+D`) denies that call and every remaining call in the batch, then ends the turn.

### Project context (AGENTS.md)

The `AGENTS.md` **at the project root** is appended to the system prompt on every call. The project root is the nearest ancestor containing `.git` (directory *or* worktree file, searched at most 5 levels up), or the launch directory when not inside a repo. Nothing else counts, so a stray `AGENTS.md` in a parent or your home dir is never picked up. Re-read each turn, so edits apply without a restart.

### Tools the model can use

| Tool | Behavior |
|---|---|
| `read_file` | `path`, optional `offset`/`limit` (lines; windows bounded at 2000). `image: true` returns png/jpg/jpeg/gif/webp (≤5 MB) as an image block the model can see. Auto-allowed (no prompt, within bounds). Files over 1 MB are refused with a pagination hint; text output caps at 100 KB + `[truncated]` |
| `write_file` | `path`, `content` — creates directories, overwrites. Approval prompt shows a unified diff against the current file; paths outside the workspace are denied (see `--no-sandbox`) |
| `bash` | Runs with bash, returns stdout+stderr (capped at 100 KB); a timeout kills it and reports what was printed; non-zero exits show `[exit N]`. The model may request a **shorter** per-call `timeout`, never longer than `--tool-timeout` |
| `task` | Delegate a self-contained subtask to a fresh subagent with isolated context (shares gate + workspace sandbox — an `always` answered inside a subagent applies session-wide); returns final text, all totals roll up. Use for exploration/research |

Relative paths resolve against the harness's current directory; a `cd` inside one `bash` call affects only that call.

Subagents run `run_turn` with a reduced tool set in isolated history; the parent sees only the distilled result at 100 KB truncation, with token totals merged and a 12-iteration cap.

### The footer

```
↑12304 ↓456 · ctx 4100tok / 200000 (2.1%) · 1830ms
```

| Part | Meaning |
|---|---|
| `↑ / ↓` | Session token totals — input (context sent) and output, across all turns; survives `/clear` |
| `ctx` | Context size of the **latest** turn only (not summed; `input_tokens` already includes cache reads). Shows `ctx —` after `/clear` until the next reply |
| `%` | `ctx` against the model's context window, read from provider metadata (`model_info`, live for Claude/Ollama; cached) with a built-in table fallback for providers that expose nothing — extended by `~/.config/ducktape/windows.toml` (`[windows] claude-sonnet-4 = 200000`). Unknown model → no percentage |
| `ms` | Request latency; omitted when a provider doesn't report it |

## Good to know

- Repeated tool use within one turn is capped at 24 iterations (12 in subagents) (`[stopped: tool loop limit]`); a reply that hits the model's output limit ends with `[stopped: output limit]`, or `[stopped: output limit (truncated tool calls)]` if it was cut mid-tool-call.
- At ≥85% of the known context window the harness prints `[context ~N% of W — /compact or /clear advised]` once per turn — run `/compact` (keeps last ~3 exchanges and totals) or `/clear` when you see it.
- The conversation is kept in memory only — nothing is persisted; `/clear` is the undo.
- Errors (bad key, rate limits, model name not served anywhere) print and return to the prompt; transient ones retry once automatically.

## Development

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run python -m unittest discover -s tests -v     # offline, scripted fake adapter — no API keys
```

CI (GitHub Actions) mirrors provider CI, additionally type-checking `src` and `tests` with a hash-pinned `ty`. The provider dependency resolves from its GitHub repo via `uv.lock`; for provider co-development swap the `[tool.uv.sources]` line to a local editable path.

Known upstream defects found while building (reported, not worked around with hacks): see [PROVIDER_GAPS.md](PROVIDER_GAPS.md). Changes: [CHANGELOG.md](CHANGELOG.md).
