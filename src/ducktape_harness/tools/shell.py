"""Bash tool.

WHY: subprocess.run with capture+timeout so tool exceptions/timeouts map
to is_error blocks and never crash the harness; coarse 'always' is
documented because the gate cannot preview post-expansion shell effects.
"""

from __future__ import annotations

import subprocess
from typing import Any

from ducktape_harness import config
from ducktape_harness.config import OUTPUT_CAP
from ducktape_harness.tools.base import ToolError, register


class BashTool:
    name = "bash"
    description = (
        "Run a shell command (bash -lc). cwd is the harness cwd; `cd` affects only that command."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
            "timeout": {
                "type": "number",
                "description": "Timeout seconds, must be shorter than the harness limit",
            },
        },
        "required": ["command"],
    }

    def preview(self, input: dict[str, Any], cwd: str) -> str:
        # WHY: the gate must show the full command that will run.
        cmd = input.get("command")
        return cmd if isinstance(cmd, str) else str(input)

    def run(self, input: dict[str, Any], cwd: str) -> str:
        command = input.get("command", "")
        if not isinstance(command, str):
            raise ToolError("command must be a string")
        # Use TOOL_TIMEOUT unless overridden (but per spec tool timeout is fixed; we honor input timeout if given)
        # WHY min(): --tool-timeout is the operator's ceiling; the model may
        # request a SHORTER per-call timeout but never a longer one.
        try:
            requested = float(t) if (t := input.get("timeout")) is not None else None
        except (ValueError, TypeError):
            requested = None
        if requested is None or requested <= 0:
            timeout_val = config.TOOL_TIMEOUT
        else:
            timeout_val = min(requested, config.TOOL_TIMEOUT)
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_val,
            )
        except subprocess.TimeoutExpired as e:
            out = ""
            if e.stdout:
                out += e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace")
            if e.stderr:
                out += e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace")
            if len(out.encode("utf-8")) > OUTPUT_CAP:
                out = (
                    out.encode("utf-8")[:OUTPUT_CAP].decode("utf-8", errors="ignore")
                    + "\n[truncated]"
                )
            msg = f"timed out after {timeout_val}s"
            if out:
                msg += f"\n{out}"
            raise ToolError(msg) from e
        except OSError as e:
            raise ToolError(str(e)) from e
        output = (result.stdout or "") + (result.stderr or "")
        # Truncate to OUTPUT_CAP
        if len(output.encode("utf-8")) > OUTPUT_CAP:
            output = (
                output.encode("utf-8")[:OUTPUT_CAP].decode("utf-8", errors="ignore")
                + "\n[truncated]"
            )
        if result.returncode != 0:
            if output:
                return f"{output}\n[exit {result.returncode}]"
            return f"[exit {result.returncode}]"
        return output if output else "(no output)"


register(BashTool())

__all__ = ["BashTool"]
