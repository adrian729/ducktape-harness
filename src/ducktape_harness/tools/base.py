"""Tool protocol and registry.

WHY: single REGISTRY dict lets engine look up tool by name without
import cycles; Tool protocol keeps run(input, cwd) signature uniform.
"""

from __future__ import annotations

from typing import Any, Protocol


class ToolError(Exception):
    """Harness tool failure — caught by engine and turned into is_error block."""


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def run(self, input: dict[str, Any], cwd: str) -> Any:
        """Return str (truncated into the result) or a list of content blocks
        (e.g. images — ToolResultBlock.content accepts str | list[Text|Image])."""
        ...


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.name] = tool
    return tool


# Static spec for the parent-only `task` subagent tool (engine special-cases
# it; no generic run() is registered, but it must appear to the model).
TASK_SPEC: dict[str, Any] = {
    "name": "task",
    "description": (
        "Delegate a self-contained subtask to a fresh agent with its own context. "
        "Returns the subagent's final text. Use for exploration/research to keep "
        "this conversation clean."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task description"},
            "prompt": {"type": "string", "description": "Full prompt for the subagent"},
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of tools the subagent may use",
            },
        },
        "required": ["description", "prompt"],
    },
}
