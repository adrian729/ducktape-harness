"""Scripted fake Adapter for offline tests.

WHY: lets harness tests inject deterministic Provider responses without network,
covering every PLAN.md interrupt/pairing/error matrix while keeping
provider contract faithful (Iterator[StreamEvent] ending message_stop).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any, cast

from ducktape_provider.types import Adapter, Message, ModelInfo, Response, StreamEvent, ToolDef


def _resp(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    r: dict[str, Any] = {"content": content, "stop_reason": stop_reason}
    if usage is not None:
        r["usage"] = usage
    if latency_ms is not None:
        r["latency_ms"] = latency_ms
    return r


def text_events(
    text: str,
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    index: int = 0,
) -> list[dict[str, Any]]:
    """One text block plus optional thinking, ending with message_stop."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    evs: list[dict[str, Any]] = []
    if text:
        evs.append({"type": "text_delta", "index": index, "text": text})
        evs.append({"type": "block_stop", "index": index})
    evs.append({"type": "message_stop", "response": _resp(content, stop_reason, usage, latency_ms)})
    return evs


def thinking_events(
    thinking: str,
    text: str = "done",
    usage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": thinking},
        {"type": "text", "text": text},
    ]
    return [
        {"type": "thinking_delta", "index": 0, "thinking": thinking},
        {"type": "block_stop", "index": 0},
        {"type": "text_delta", "index": 1, "text": text},
        {"type": "block_stop", "index": 1},
        {"type": "message_stop", "response": _resp(content, "end_turn", usage)},
    ]


def tool_events(
    tools: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    latency_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Tools: list of {id, name, input} dicts. Emits tool_use_start+block_stop per tool."""
    events: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []
    for i, t in enumerate(tools):
        events.append({"type": "tool_use_start", "index": i, "id": t["id"], "name": t["name"]})
        events.append({"type": "block_stop", "index": i})
        content.append(
            {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t.get("input", {})}
        )
    events.append(
        {"type": "message_stop", "response": _resp(content, "tool_use", usage, latency_ms)}
    )
    return events


def max_tokens_tool_events(
    tools: list[dict[str, Any]],
    completed_indices: set[int],
    usage: dict[str, Any] | None = None,
    truncated: set[int] | None = None,
) -> list[dict[str, Any]]:
    """max_tokens discriminant: only completed indices get block_stop.

    `truncated` mirrors the provider's post-b0ad0bb marker: block gets
    truncated=True regardless of block_stop.
    """
    events: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []
    for i, t in enumerate(tools):
        events.append({"type": "tool_use_start", "index": i, "id": t["id"], "name": t["name"]})
        if i in completed_indices:
            events.append({"type": "block_stop", "index": i})
        blk: dict[str, Any] = {
            "type": "tool_use",
            "id": t["id"],
            "name": t["name"],
            "input": t.get("input", {}),
        }
        if truncated and i in truncated:
            blk["truncated"] = True
        content.append(blk)
    events.append({"type": "message_stop", "response": _resp(content, "max_tokens", usage)})
    return events


def pause_events(text: str = "paused") -> list[dict[str, Any]]:
    return [
        {"type": "text_delta", "index": 0, "text": text},
        {"type": "block_stop", "index": 0},
        {"type": "message_stop", "response": _resp([{"type": "text", "text": text}], "pause_turn")},
    ]


def empty_events() -> list[dict[str, Any]]:
    return [{"type": "message_stop", "response": _resp([], "end_turn")}]


class FakeAdapter(Adapter):
    """Scripted Adapter queueing per-call event lists or exceptions."""

    def __init__(
        self,
        scripts: list[Any] | None = None,
        models_set: set[str] | None = None,
        available: bool = True,
        infos: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._queue: deque[Any] = deque(scripts or [])
        self._models = models_set if models_set is not None else {"fake-model", "other-model"}
        self._available = available
        self._infos = infos or {}
        self.calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    def set_scripts(self, scripts: list[Any]) -> None:
        self._queue = deque(scripts)

    def push(self, item: Any) -> None:
        self._queue.append(item)

    def is_available(self) -> bool:
        return self._available

    def models(self) -> set[str]:
        return set(self._models)

    def model_info(self, model: str) -> ModelInfo | None:
        return cast("ModelInfo | None", self._infos.get(model))

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        self.chat_calls.append(
            {"model": model, "messages": list(messages), "system": system, "tools": tools}
        )
        if not self._queue:
            return cast(Response, _resp([{"type": "text", "text": "chat-ok"}]))
        item = self._queue.popleft()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, list):
            # find message_stop response
            for ev in item:
                if ev and ev.get("type") == "message_stop":
                    return cast(Response, ev["response"])
            return cast(Response, _resp([{"type": "text", "text": "no-stop"}]))
        if isinstance(item, dict) and "response" in item:
            return cast(Response, item["response"])
        return cast(Response, item)

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "system": system,
                "tools": tools,
                "config": config,
            }
        )
        if not self._queue:
            # default text turn
            default = _resp([{"type": "text", "text": "hello"}])
            events = [
                {"type": "text_delta", "index": 0, "text": "hello"},
                {"type": "block_stop", "index": 0},
                {"type": "message_stop", "response": default},
            ]
            yield from cast("list[StreamEvent]", events)
            return
        item = self._queue.popleft()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, list):
            # filter Nones (from helpers)
            for ev in item:
                if ev is None:
                    continue
                yield cast(StreamEvent, ev)
            return
        # if callable, call to get iterator that may raise lazily
        if callable(item):
            gen = item()  # type: ignore[operator]
            yield from gen  # type: ignore[misc]
            return
        raise TypeError(f"bad script entry {item!r}")


def lazy_raise(exc: BaseException) -> Any:
    """Return a callable that yields one delta then raises — tests lazy ValueError."""

    def gen() -> Iterator[dict[str, Any]]:
        yield {"type": "text_delta", "index": 0, "text": "partial"}
        raise exc
        yield  # unreachable

    return gen
