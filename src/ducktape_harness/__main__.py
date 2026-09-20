"""REPL entry point — argparse, Provider setup, external loop.

WHY: argparse builds Provider with correct timeout semantics (only pass
timeout if flag given; None default keeps adapter defaults). REPL handles
blank/EOF/KI per spec and delegates to engine + footer.
"""

from __future__ import annotations

import argparse
import os
import readline  # noqa: F401  # side-effect: enables history

from ducktape_provider import Provider

import ducktape_harness.tools  # noqa: F401  # populate REGISTRY
from ducktape_harness.commands import (
    cmd_attach,
    cmd_clear,
    cmd_compact,
    cmd_detach,
    cmd_export,
    cmd_help,
    cmd_model,
    cmd_usage,
)
from ducktape_harness.config import DEFAULT_MODEL, TOOL_TIMEOUT
from ducktape_harness.engine import run_turn
from ducktape_harness.render import format_footer
from ducktape_harness.session import Session
from ducktape_harness.system_prompt import SYSTEM_PROMPT  # noqa: F401  # ensure import


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ducktape", description="Ducktape harness")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model id")
    p.add_argument("--provider", default=None, help="Provider name (pin)")
    p.add_argument("--tool-timeout", type=float, default=TOOL_TIMEOUT, help="Tool timeout seconds")
    p.add_argument("--http-timeout", default=None, help="HTTP timeout seconds or 'none' to disable")
    p.add_argument(
        "--no-sandbox",
        action="store_true",
        help="allow write_file to paths outside the launch directory",
    )
    return p


def _parse_http_timeout(value: str | None) -> tuple[bool, float | None]:
    """Return (given, timeout). Only pass Provider(timeout=...) if flag given."""
    if value is None:
        return False, None
    if isinstance(value, str) and value.lower() == "none":
        return True, None
    try:
        return True, float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        raise SystemExit(f"invalid --http-timeout {value!r}")


def create_provider(http_timeout_given: bool, http_timeout_val: float | None) -> Provider:
    if http_timeout_given:
        return Provider(timeout=http_timeout_val)
    return Provider()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply tool timeout override if given (update config module value for tools)
    if args.tool_timeout is not None:
        import ducktape_harness.config as cfg

        cfg.TOOL_TIMEOUT = float(args.tool_timeout)

    http_given, http_val = _parse_http_timeout(args.http_timeout)

    provider = create_provider(http_given, http_val)

    session = Session(model=args.model, pin=args.provider, provider=provider)
    # WHY default-on: the gate shows previews but a blind yes outside the
    # project is unrecoverable; the launch dir is the agreed blast radius.
    if not args.no_sandbox:
        session.sandbox_root = os.getcwd()

    print(
        f"ducktape harness — model {session.model} · provider"
        f" {session.pin if session.pin else 'auto (matched from model name)'}"
    )
    print("Type /help for commands.")

    while True:
        try:
            line = input("› ")
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            # slash dispatch
            parts = stripped.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/clear":
                print(cmd_clear(session))
                continue
            elif cmd == "/help":
                print(cmd_help())
                continue
            elif cmd == "/usage":
                print(cmd_usage(session))
                continue
            elif cmd == "/attach":
                print(cmd_attach(session, arg))
                continue
            elif cmd == "/detach":
                print(cmd_detach(session, arg))
                continue
            elif cmd == "/model":
                print(cmd_model(session, arg))
                continue
            elif cmd == "/compact":
                print(cmd_compact(session, arg))
                continue
            elif cmd == "/export":
                print(cmd_export(session, arg))
                continue
            else:
                print(f"unknown command {cmd} — try /help")
                continue

        # Defensive KI wrapper around run_turn (spec)
        try:
            result = run_turn(session, stripped)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except SystemExit:
            raise
        except Exception as e:
            print(f"[error] {e}")
            continue

        # Footer: ↑↓ cumulative totals, ctx latest-usage-only, latency independent of usage (render drops None segments)
        footer = format_footer(
            session.totals.input_tokens,
            session.totals.output_tokens,
            session.last_usage,
            session.model,
            session.last_latency,
            provider=session.provider,
            pin=session.pin,
        )
        print(footer)

        # Optionally print result if not already streamed? Engine already printed deltas; no need.
        _ = result


if __name__ == "__main__":
    main()
