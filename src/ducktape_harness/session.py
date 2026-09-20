"""Session state — messages, model, totals.

WHY: Session owns conversation history and usage accounting so engine and
REPL share one mutable truth; reset() keeps totals+gate because /clear is
scoped to messages only per spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ducktape_provider.types import ImageBlock, Message, Usage

from ducktape_harness.config import DEFAULT_MODEL
from ducktape_harness.gate import PermissionGate


@dataclass
class Totals:
    """Cumulative counters across the process lifetime."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turns: int = 0
    turns_with_usage: int = 0


@dataclass
class Session:
    """Mutable harness session.

    WHY: provider is stored here so engine.run_turn can accept (session,
    user_text) per spec without an extra provider argument; gate lives here
    so /clear's reset does not drop always-allowed entries.
    """

    model: str = DEFAULT_MODEL
    pin: str | None = None
    provider: Any | None = None
    messages: list[Message] = field(default_factory=list)
    totals: Totals = field(default_factory=Totals)
    last_usage: Usage | None = None
    last_latency: float | None = None
    gate: PermissionGate = field(default_factory=PermissionGate)
    # workspace root for the write-path policy; None disables enforcement
    # (tests), __main__ sets it to the launch cwd unless --no-sandbox.
    sandbox_root: str | None = None
    # ImageBlocks staged by /attach, consumed (and cleared) by the next run_turn
    pending_images: list[ImageBlock] = field(default_factory=list)
    # set by run_turn only on real user interrupts (not vendor errors) so
    # subagents can tell "aborted" from "failed silently"
    last_interrupted: bool = False

    def reset(self) -> None:
        """Clear messages and staged attachments; totals and gate persist."""
        self.messages.clear()
        self.pending_images.clear()
        self.last_usage = None
        self.last_latency = None
