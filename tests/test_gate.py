"""Gate tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ducktape_provider import Provider

import ducktape_harness.tools  # noqa: F401
from ducktape_harness.engine import run_turn
from ducktape_harness.gate import PermissionGate
from ducktape_harness.session import Session
from tests.fake_adapter import FakeAdapter, text_events, tool_events


def make_session(scripts):
    adapter = FakeAdapter(scripts=scripts)
    provider = Provider(adapters={"fake": adapter}, autodiscover=False)
    session = Session(model="fake-model", pin="fake", provider=provider)
    session.gate = PermissionGate()
    return session


class TestGateUnit(unittest.TestCase):
    def test_allow(self):
        gate = PermissionGate()
        with patch("builtins.input", return_value="y"):
            self.assertEqual(gate.ask("read_file", "preview"), "allow")

    def test_deny(self):
        gate = PermissionGate()
        with patch("builtins.input", return_value="n"):
            self.assertEqual(gate.ask("bash", "ls"), "deny")

    def test_always_persists(self):
        gate = PermissionGate()
        with patch("builtins.input", return_value="always"):
            gate.ask("read_file", "x")
        self.assertIn("read_file", gate.always_allowed)
        # second call bypasses input
        with patch("builtins.input", side_effect=Exception("should not prompt")):
            self.assertEqual(gate.ask("read_file", "x"), "allow")

    def test_always_bypass_in_engine(self):
        session = make_session(
            [
                tool_events([{"id": "t1", "name": "read_file", "input": {"path": "/tmp/x"}}]),
                text_events("done"),
            ]
        )
        session.gate.always_allowed.add("read_file")
        # should not call input
        with (
            patch("builtins.input", side_effect=Exception("unexpected input")),
            patch("builtins.print"),
        ):
            import os
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "x")
                with open(p, "w") as f:
                    f.write("hi")
                # need to map /tmp/x to real file via cwd trick: tool resolves relative; but we use absolute /tmp/x won't match
                # Instead push a tool that reads a relative path and set cwd
                # Recreate session with relative path
                adapter = FakeAdapter(
                    scripts=[
                        tool_events([{"id": "t1", "name": "read_file", "input": {"path": "x"}}]),
                        text_events("done"),
                    ]
                )
                provider = Provider(adapters={"fake": adapter}, autodiscover=False)
                session.provider = provider  # type: ignore[attr-defined]
                session.gate = PermissionGate({"read_file"})
                with patch("os.getcwd", return_value=td):
                    result = run_turn(session, "hi")
        self.assertEqual(result, "done")


class TestGateInterruptMatrix(unittest.TestCase):
    def test_interrupt_at_prompt_batch(self):
        # Already covered in engine test, but repeat for gate isolation
        session = make_session(
            [
                tool_events(
                    [
                        {"id": "t1", "name": "bash", "input": {"command": "echo hi"}},
                        {"id": "t2", "name": "bash", "input": {"command": "echo bye"}},
                    ]
                )
            ]
        )
        with patch("builtins.input", side_effect=KeyboardInterrupt), patch("builtins.print"):
            run_turn(session, "hi")
        user_msg = session.messages[2]
        self.assertEqual(len(user_msg["content"]), 2)  # type: ignore[arg-type]
        self.assertIn("interrupted", user_msg["content"][0]["content"])  # type: ignore[attr-defined]
        self.assertEqual(user_msg["content"][0]["content"], "interrupted")  # type: ignore[attr-defined]
        self.assertIn("denied (batch interrupted)", user_msg["content"][1]["content"])  # type: ignore[attr-defined]
        self.assertEqual(user_msg["content"][1]["content"], "denied (batch interrupted)")  # type: ignore[attr-defined]

    def test_always_survives_clear(self):
        session = make_session([text_events("ok")])
        session.gate.always_allowed.add("bash")
        from ducktape_harness.commands import cmd_clear

        cmd_clear(session)
        self.assertIn("bash", session.gate.always_allowed)
        self.assertIn("bash", session.gate.always_allowed)
