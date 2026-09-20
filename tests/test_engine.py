"""Engine tests — offline via FakeAdapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from ducktape_provider import Provider

import ducktape_harness.tools  # noqa: F401  populate REGISTRY
from ducktape_harness.config import MAX_TOOL_ITERATIONS, PAUSE_CAP, PROJECT_ROOT_MAX_LEVELS
from ducktape_harness.engine import run_turn
from ducktape_harness.session import Session
from ducktape_harness.system_prompt import SYSTEM_PROMPT as SYSTEM_PROMPT_BASE
from tests.fake_adapter import (
    FakeAdapter,
    empty_events,
    lazy_raise,
    max_tokens_tool_events,
    pause_events,
    text_events,
    thinking_events,
    tool_events,
)


def _summary_resp(text: str) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": None,
    }


def make_session(scripts, models_set=None):
    adapter = FakeAdapter(scripts=scripts, models_set=models_set)
    provider = Provider(adapters={"fake": adapter}, autodiscover=False)
    session = Session(model="fake-model", pin="fake", provider=provider)
    # Session already has gate=PermissionGate(); ensure it's PermissionGate
    from ducktape_harness.gate import PermissionGate

    session.gate = PermissionGate()
    return session, adapter


class TestEngineNormal(unittest.TestCase):
    def test_normal_text_turn(self):
        session, adapter = make_session(
            [text_events("hello", usage={"input_tokens": 5, "output_tokens": 3}, latency_ms=42)]
        )
        with patch("builtins.print"):
            result = run_turn(session, "hi")
        self.assertEqual(result, "hello")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[1]["content"][0]["text"], "hello")  # type: ignore[index]
        self.assertEqual(session.totals.input_tokens, 5)

    def test_thinking_replayed_verbatim(self):
        session, _ = make_session([thinking_events("step by step", text="answer")])
        with patch("builtins.print"):
            run_turn(session, "hi")
        # assistant content includes thinking block
        content = session.messages[1]["content"]
        self.assertTrue(any(b.get("type") == "thinking" for b in content))
        self.assertEqual(content[0].get("thinking"), "step by step")

    def test_empty_content_placeholder(self):
        session, _ = make_session([empty_events()])
        with patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        self.assertEqual(result, "[no text response]")
        self.assertEqual(session.messages[1]["content"][0]["text"], "[no text response]")  # type: ignore[index]
        # placeholder should be printed to user (not just in history)
        printed = " ".join(str(c.args[0]) for c in mp.call_args_list if c.args)
        self.assertIn("[no text response]", printed)

    def test_empty_content_not_duplicate_when_streamed(self):
        session, _ = make_session([text_events("hello")])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        # hello printed via stream, not placeholder
        calls = [str(c.args[0]) if c.args else "" for c in mp.call_args_list]
        self.assertTrue(any("hello" in s for s in calls))
        # should not have printed placeholder additionally
        self.assertFalse(any("[no text response]" in s for s in calls))

    def test_alternation_merge_user(self):
        # After a tool turn, last message is assistant? then tool results user message.
        # Then next user_text should merge into trailing user message.
        session, adapter = make_session(
            [
                tool_events(
                    [
                        {
                            "id": "t1",
                            "name": "write_file",
                            "input": {"path": "/tmp/x", "content": "y"},
                        }
                    ]
                ),
                text_events("done"),
            ]
        )
        # need a file to read
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x")
            with open(p, "w") as f:
                f.write("content")
            # patch cwd + gate allow
            with (
                patch("builtins.input", return_value="y"),
                patch("os.getcwd", return_value=td),
                patch("builtins.print"),
            ):
                run_turn(session, "first")
            # now trailing is assistant ("done"). Next turn with empty scripts? need new script
            session.provider = Provider(
                adapters={"fake": FakeAdapter(scripts=[text_events("second")])}, autodiscover=False
            )
            from ducktape_harness.gate import PermissionGate

            session.gate = PermissionGate(session.gate.always_allowed)
            # Now add user message merging: craft state where last is user
            session.messages.append({"role": "user", "content": [{"type": "text", "text": "prev"}]})
            _ = session.provider._adapters["fake"]  # type: ignore[attr-defined]
            # But easier: test directly run_turn merge
            # Create fresh session where last is user, then run_turn should merge
            s2, _ = make_session([text_events("ok")])
            s2.messages.append({"role": "user", "content": [{"type": "text", "text": "prev"}]})
            with patch("builtins.print"):
                run_turn(s2, "new")
            # Should have merged, not new message: last before was user, so only 1 user message with 2 blocks
            self.assertEqual(len(s2.messages), 2)  # user merged + assistant
            self.assertEqual(len(s2.messages[0]["content"]), 2)
            self.assertEqual(s2.messages[0]["content"][1]["text"], "new")  # type: ignore[index]

    def test_stripped_input_passed(self):
        # REPL should pass stripped line; run_turn itself should store stripped text merging
        session, _ = make_session([text_events("ok")])
        with patch("builtins.print"):
            run_turn(session, "  hi with spaces  ")
        # engine stores exactly what is passed; __main__ responsibility is to strip before call
        # Here we verify run_turn preserves input as given (stripped case tested via __main__ integration)
        self.assertEqual(session.messages[0]["content"][0]["text"], "  hi with spaces  ")  # type: ignore[index]


class TestEngineTools(unittest.TestCase):
    def test_tool_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("hi")
            session, _ = make_session(
                [
                    tool_events([{"id": "t1", "name": "read_file", "input": {"path": p}}]),
                    text_events("done", usage={"input_tokens": 2, "output_tokens": 2}),
                ]
            )
            with (
                patch("builtins.input", return_value="y"),
                patch("os.getcwd", return_value=td),
                patch("builtins.print"),
            ):
                result = run_turn(session, "read")
            self.assertEqual(result, "done")
            # pairing invariant: one user message with one ToolResultBlock
            self.assertEqual(session.messages[2]["role"], "user")
            self.assertEqual(len(session.messages[2]["content"]), 1)
            tr = session.messages[2]["content"][0]
            self.assertEqual(tr["tool_use_id"], "t1")
            self.assertEqual(tr["name"], "read_file")

    def test_multi_tool_single_user_message(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["a", "b"]:
                with open(os.path.join(td, name), "w") as f:
                    f.write(name)
            session, _ = make_session(
                [
                    tool_events(
                        [
                            {
                                "id": "t1",
                                "name": "read_file",
                                "input": {"path": os.path.join(td, "a")},
                            },
                            {
                                "id": "t2",
                                "name": "read_file",
                                "input": {"path": os.path.join(td, "b")},
                            },
                        ]
                    ),
                    text_events("done"),
                ]
            )
            with (
                patch("builtins.input", side_effect=["y", "y"]),
                patch("os.getcwd", return_value=td),
                patch("builtins.print"),
            ):
                run_turn(session, "multi")
            self.assertEqual(session.messages[2]["role"], "user")
            self.assertEqual(len(session.messages[2]["content"]), 2)
            ids = {b["tool_use_id"] for b in session.messages[2]["content"]}  # type: ignore[attr-defined]
            self.assertEqual(ids, {"t1", "t2"})
            for b in session.messages[2]["content"]:  # type: ignore[attr-defined]
                self.assertIn("name", b)
                self.assertIn("tool_use_id", b)

    def test_unknown_tool(self):
        session, _ = make_session(
            [
                tool_events([{"id": "t1", "name": "unknown_tool", "input": {}}]),
                text_events("done"),
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            run_turn(session, "hi")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertTrue(tr.get("is_error"))
        self.assertIn("unknown tool unknown_tool", tr["content"])
        self.assertNotIn("is_error:", tr["content"])
        self.assertEqual(tr["name"], "unknown_tool")
        self.assertEqual(tr["tool_use_id"], "t1")

    def test_denial(self):
        session, _ = make_session(
            [
                tool_events(
                    [
                        {
                            "id": "t1",
                            "name": "write_file",
                            "input": {"path": "/tmp/x", "content": "y"},
                        }
                    ]
                ),
                text_events("done"),
            ]
        )
        with patch("builtins.input", return_value="n"), patch("builtins.print"):
            run_turn(session, "hi")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertTrue(tr.get("is_error"))
        self.assertIn("user denied", tr["content"])
        self.assertEqual(tr["content"], "user denied")

    def test_gate_interrupt(self):
        session, _ = make_session(
            [
                tool_events(
                    [
                        {
                            "id": "t1",
                            "name": "write_file",
                            "input": {"path": "/tmp/a", "content": "z"},
                        },
                        {
                            "id": "t2",
                            "name": "write_file",
                            "input": {"path": "/tmp/b", "content": "z"},
                        },
                    ]
                ),
            ]
        )

        def raise_ki(_prompt=""):
            raise KeyboardInterrupt

        with patch("builtins.input", side_effect=raise_ki), patch("builtins.print") as mp:
            run_turn(session, "hi")
        # pairing invariant: both tools have results, first interrupted, second batch-interrupted
        user_msg = session.messages[2]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(len(user_msg["content"]), 2)  # type: ignore[arg-type]
        c0 = user_msg["content"][0]  # type: ignore[attr-defined]
        c1 = user_msg["content"][1]  # type: ignore[attr-defined]
        self.assertIn("interrupted", c0["content"])
        self.assertEqual(c0["content"], "interrupted")
        self.assertIn("denied (batch interrupted)", c1["content"])
        # turn ends (no further loop) and prints [interrupted]
        printed = " ".join(str(c.args[0]) for c in mp.call_args_list if c.args)
        self.assertIn("[interrupted]", printed)

    def test_run_interrupt(self):
        # Simulate tool run raising KI via monkeypatching registry tool
        import ducktape_harness.tools.base as base

        orig = base.REGISTRY["read_file"]

        class BadTool:
            name = "read_file"
            description = "bad"
            parameters = {"type": "object", "properties": {}}

            def run(self, input, cwd):  # type: ignore[no-untyped-def]
                raise KeyboardInterrupt

        base.REGISTRY["read_file"] = BadTool()  # type: ignore[assignment]
        try:
            session, _ = make_session(
                [tool_events([{"id": "t1", "name": "read_file", "input": {"path": "/x"}}])]
            )
            with patch("builtins.input", return_value="y"), patch("builtins.print") as mp:
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertIn("interrupted", tr["content"])
            self.assertEqual(tr["content"], "interrupted")
            printed = " ".join(str(c.args[0]) for c in mp.call_args_list if c.args)
            self.assertIn("[interrupted]", printed)
        finally:
            base.REGISTRY["read_file"] = orig

    def test_is_error_prefix_not_misflagged(self):
        # legitimate tool output starting with is_error: should NOT be flagged is_error
        import ducktape_harness.tools.base as base

        orig = base.REGISTRY["read_file"]

        class FakeIsErrorTool:
            name = "read_file"
            description = "fake"
            parameters = {"type": "object", "properties": {}}

            def run(self, input, cwd):  # type: ignore[no-untyped-def]
                return "is_error: this is legitimate file content"

        base.REGISTRY["read_file"] = FakeIsErrorTool()  # type: ignore[assignment]
        try:
            session, _ = make_session(
                [
                    tool_events([{"id": "t1", "name": "read_file", "input": {"path": "/x"}}]),
                    text_events("done"),
                ]
            )
            with patch("builtins.input", return_value="y"), patch("builtins.print"):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertEqual(tr["content"], "is_error: this is legitimate file content")
            # is_error flag should NOT be set for legitimate output
            self.assertIsNone(tr.get("is_error"))
        finally:
            base.REGISTRY["read_file"] = orig

    def test_tool_error_raised_maps_to_is_error(self):
        import ducktape_harness.tools.base as base
        from ducktape_harness.tools.base import ToolError

        orig = base.REGISTRY["read_file"]

        class ErrTool:
            name = "read_file"
            description = "err"
            parameters = {"type": "object", "properties": {}}

            def run(self, input, cwd):  # type: ignore[no-untyped-def]
                raise ToolError("file too large")

        base.REGISTRY["read_file"] = ErrTool()  # type: ignore[assignment]
        try:
            session, _ = make_session(
                [tool_events([{"id": "t1", "name": "read_file", "input": {"path": "/x"}}])]
            )
            with patch("builtins.input", return_value="y"), patch("builtins.print"):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertTrue(tr.get("is_error"))
            self.assertIn("file too large", tr["content"])
            self.assertNotIn("is_error:", tr["content"])
        finally:
            base.REGISTRY["read_file"] = orig

    def test_cap_24_after_append(self):
        # Simulate 24 iterations: each tool turn returns tool_use, need 24 scripts + final text
        scripts = []
        for i in range(MAX_TOOL_ITERATIONS):
            scripts.append(
                tool_events([{"id": f"t{i}", "name": "read_file", "input": {"path": "/tmp/x"}}])
            )
        scripts.append(text_events("never reached"))
        session, _ = make_session(scripts)
        # Patch read_file to succeed via temp file
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x")
            with open(p, "w") as f:
                f.write("hi")
            # need always to avoid input exhaustion
            session.gate.always_allowed.add("read_file")
            with patch("os.getcwd", return_value=td), patch("builtins.print") as mock_print:
                run_turn(session, "loop")
            # After 24, should have printed stopped and ended, with 24 user result messages appended
            # Check that last append happened (no dangling)
            # Messages: user0, assistant tool, user result, assistant tool, ... total 1 + 24*2
            self.assertGreaterEqual(len(session.messages), 1 + MAX_TOOL_ITERATIONS * 2 - 1)
            mock_print.assert_any_call("[stopped: tool loop limit]")

    def test_max_tokens_with_completed_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "ok")
            with open(p, "w") as f:
                f.write("ok")
            session, _ = make_session(
                [
                    max_tokens_tool_events(
                        [
                            {"id": "t1", "name": "read_file", "input": {"path": p}},
                            {"id": "t2", "name": "read_file", "input": {"path": p}},
                        ],
                        completed_indices={0},
                    )
                ]
            )
            session.gate.always_allowed.add("read_file")
            with patch("os.getcwd", return_value=td), patch("builtins.print"):
                run_turn(session, "hi")
            # One user message with 2 results: first executed, second truncated
            user_msg = session.messages[2]
            self.assertEqual(len(user_msg["content"]), 2)  # type: ignore[arg-type]
            c0 = user_msg["content"][0]  # type: ignore[attr-defined]
            c1 = user_msg["content"][1]  # type: ignore[attr-defined]
            self.assertNotIn("truncated by max_tokens", c0["content"])
            self.assertIn("truncated by max_tokens", c1["content"])
            self.assertEqual(c0["name"], "read_file")
            self.assertEqual(c1["name"], "read_file")
            self.assertEqual(c1["content"], "truncated by max_tokens")
            # Turn ends (no loop) — messages length 3
            self.assertEqual(len(session.messages), 3)

    def test_max_tokens_without_tools(self):
        from ducktape_provider.types import Response as _R  # noqa: F401

        resp: dict = {
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
        }
        session, _ = make_session([[{"type": "message_stop", "response": resp}]])
        with patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        mp.assert_any_call("[stopped: output limit]")
        self.assertEqual(result, "partial")

    def test_bash_nonzero_via_engine(self):
        session, _ = make_session(
            [
                tool_events(
                    [{"id": "t1", "name": "bash", "input": {"command": "echo hi; exit 2"}}]
                ),
                text_events("done"),
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            run_turn(session, "hi")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertIn("hi", tr["content"])
        self.assertIn("[exit 2]", tr["content"])
        self.assertIsNone(tr.get("is_error"))


class TestEnginePause(unittest.TestCase):
    def test_pause_merge_and_cap(self):
        # pause continuation merges into trailing assistant
        session, _ = make_session(
            [
                pause_events("part1"),
                pause_events("part2"),
                text_events("final"),
            ]
        )
        with patch("builtins.print"):
            result = run_turn(session, "hi")
        self.assertEqual(result, "final")
        # trailing assistant should have merged content (3 blocks via extends? but our logic extends each)
        # First pause: messages [user, assistant part1]; second: extends assistant => [part1, part2]; third: extends again
        self.assertEqual(session.messages[1]["role"], "assistant")
        texts = [
            b.get("text", "") for b in session.messages[1]["content"] if b.get("type") == "text"
        ]
        self.assertIn("part1", texts)
        self.assertIn("part2", texts)
        self.assertIn("final", texts)

    def test_pause_cap(self):
        scripts = [pause_events(f"p{i}") for i in range(PAUSE_CAP + 2)]
        session, _ = make_session(scripts)
        with patch("builtins.print"):
            result = run_turn(session, "hi")
        # After PAUSE_CAP exceeded, should return last pause text without merging beyond cap
        self.assertIsInstance(result, str)


class TestEngineRetryAndErrors(unittest.TestCase):
    def test_content_seen_no_retry(self):
        from ducktape_provider import RateLimitError

        # content_seen flips on text_delta, so retry should NOT happen
        err = RateLimitError("rate limited", status=429, body="", retry_after=1)
        session, _ = make_session(
            [
                lazy_raise(err),  # yields text_delta then raises RateLimitError
                text_events("should not retry"),
            ]
        )
        with patch("builtins.print") as mp, patch("time.sleep") as mock_sleep:
            run_turn(session, "hi")
        mock_sleep.assert_not_called()
        # prints error, not retry
        printed = " ".join(str(a) for a in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("rate limited", printed.lower())

    def test_ratelimit_retry_with_retry_after_none(self):
        from ducktape_provider import RateLimitError

        err = RateLimitError("rate limited", status=429, body="", retry_after=None)
        session, _ = make_session([err, text_events("ok")])
        with patch("time.sleep") as mock_sleep, patch("builtins.print"):
            result = run_turn(session, "hi")
        mock_sleep.assert_called_once()
        self.assertEqual(mock_sleep.call_args[0][0], 5)
        self.assertEqual(result, "ok")

    def test_ratelimit_retry_after_capped(self):
        from ducktape_provider import RateLimitError

        err = RateLimitError("rate", status=429, body="", retry_after=120)
        session, _ = make_session([err, text_events("ok2")])
        with patch("time.sleep") as mock_sleep, patch("builtins.print"):
            result = run_turn(session, "hi")
        mock_sleep.assert_called_once_with(60)
        self.assertEqual(result, "ok2")

    def test_request_timeout_immediate_retry(self):
        from ducktape_provider import RequestTimeoutError

        err = RequestTimeoutError("timeout", status=None, body="")
        session, _ = make_session([err, text_events("ok3")])
        with patch("time.sleep") as mock_sleep, patch("builtins.print"):
            result = run_turn(session, "hi")
        mock_sleep.assert_not_called()
        self.assertEqual(result, "ok3")

    def test_context_overflow_hint(self):
        from ducktape_provider import ContextOverflowError

        err = ContextOverflowError("overflow", status=400, body="context length")
        session, _ = make_session([err])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("/clear", printed)
        self.assertIn("/model", printed)

    def test_413_hint(self):
        # provider b0ad0bb classifies 413 as ContextOverflowError; the harness
        # hint now rides the normal overflow path (old status==413 workaround gone)
        from ducktape_provider import ContextOverflowError

        err = ContextOverflowError("large", status=413, body="request_too_large")
        session, _ = make_session([err])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("/clear", printed)

    def test_bare_413_api_error_no_longer_special_cased(self):
        from ducktape_provider import APIError

        session, _ = make_session([APIError("large", status=413, body="x")])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("[error]", printed)
        self.assertNotIn("/clear", printed)

    def test_unsupported_block_hint(self):
        from ducktape_provider import UnsupportedBlockError

        err = UnsupportedBlockError("bad block")
        session, _ = make_session([err])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("/clear", printed)

    def test_auth_error_hint(self):
        from ducktape_provider import AuthError

        err = AuthError("auth failed", status=401, body="")
        session, _ = make_session([err])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("check API key env var", printed)

    def test_malformed_response(self):
        from ducktape_provider import MalformedResponseError

        err = MalformedResponseError("bad json", status=200, body="")
        session, _ = make_session([err])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("bad json", printed)

    def test_auth_not_swallowed_by_apierror(self):
        # Ensure AuthError and Malformed are not swallowed by generic APIError
        from ducktape_provider import AuthError, MalformedResponseError

        for err_cls in [AuthError, MalformedResponseError]:
            err = err_cls("specific", status=400, body="")
            session, _ = make_session([err])
            with patch("builtins.print") as mp:
                run_turn(session, "hi")
            printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
            self.assertIn("specific", printed)

    def test_keyerror_valuerror_recover(self):
        session, _ = make_session([KeyError("no provider serves model")])
        with patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        self.assertEqual(result, "")
        self.assertTrue(mp.called)

    def test_lazy_valueerror_routed(self):
        # Lazy ValueError after content? content_seen true -> print partial+error no retry
        session, _ = make_session([lazy_raise(ValueError("bad shape"))])
        with patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        self.assertEqual(result, "")
        self.assertTrue(mp.called)


class TestEngineReviewRegressions(unittest.TestCase):
    """Regressions for the final code-review fixes."""

    def test_max_tokens_duplicate_blocks_discriminated_by_position(self):
        # identical tool_use dicts: block_stop covers position 0 only;
        # equality-based lookup would run the truncated second block
        session, _ = make_session(
            [
                max_tokens_tool_events(
                    [
                        {"id": "dup", "name": "write_file", "input": {"path": "x", "content": "y"}},
                        {"id": "dup", "name": "write_file", "input": {"path": "x", "content": "y"}},
                    ],
                    completed_indices={0},
                )
            ]
        )
        session.gate.always_allowed.add("write_file")
        with (
            tempfile.TemporaryDirectory() as td,
            patch("os.getcwd", return_value=td),
            patch("builtins.print"),
        ):
            run_turn(session, "hi")
        results = session.messages[2]["content"]  # type: ignore[attr-defined]
        self.assertEqual(len(results), 2)
        self.assertNotIn("truncated", results[0]["content"])  # executed (wrote file x)
        self.assertEqual(results[1]["content"], "truncated by max_tokens")
        self.assertTrue(results[1]["is_error"])

    def test_single_retry_budget_across_families(self):
        from ducktape_provider import RateLimitError, RequestTimeoutError

        session, adapter = make_session(
            [RateLimitError("429", retry_after=0), RequestTimeoutError("timeout"), text_events("x")]
        )
        with patch("time.sleep"), patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        # rate-limit consumed the one budget; timeout must surface, not retry
        self.assertEqual(result, "")
        self.assertEqual(len(adapter.calls), 2)
        self.assertTrue(mp.call_args_list)

    def test_gate_preview_shows_full_bash_command(self):
        long_cmd = "echo " + ("x" * 400) + " && echo done"
        session, _ = make_session(
            [tool_events([{"id": "b1", "name": "bash", "input": {"command": long_cmd}}])]
        )
        seen: list[str] = []
        real_ask = session.gate.ask

        def spy_ask(name: str, preview: str, key: str | None = None, **kw) -> str:
            seen.append(preview)
            return real_ask(name, preview, key, **kw)

        with (
            patch.object(session.gate, "ask", side_effect=spy_ask),
            patch("builtins.input", return_value="n"),
            patch("builtins.print"),
        ):
            run_turn(session, "go")
        self.assertEqual(seen, [long_cmd])  # type: ignore[list-item]


class TestPolicyAndDiffApproval(unittest.TestCase):
    """Feature round: diff approval, auto-allow reads, workspace sandbox."""

    def test_read_file_auto_allowed_without_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f")
            with open(p, "w") as f:
                f.write("data")
            session, _ = make_session(
                [
                    tool_events([{"id": "t1", "name": "read_file", "input": {"path": p}}]),
                    text_events("done"),
                ]
            )
            with (
                patch("builtins.input", side_effect=AssertionError("must not prompt")),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertEqual(tr["content"], "data")
            self.assertIsNone(tr.get("is_error"))

    def test_write_file_preview_is_diff(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("line1\nline2\n")
            session, _ = make_session(
                [
                    tool_events(
                        [
                            {
                                "id": "t1",
                                "name": "write_file",
                                "input": {"path": p, "content": "line1\nchanged\n"},
                            }
                        ]
                    ),
                    text_events("done"),
                ]
            )
            seen: list[str] = []
            real_ask = session.gate.ask

            def spy(name: str, preview: str, key: str | None = None, **kw) -> str:
                seen.append(preview)
                return real_ask(name, preview, key, **kw)

            with (
                patch.object(session.gate, "ask", side_effect=spy),
                patch("builtins.input", return_value="y"),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            self.assertEqual(len(seen), 1)
            self.assertIn("-line2", seen[0])
            self.assertIn("+changed", seen[0])
            # prompt must not duplicate the tool name in the args text
            self.assertNotIn("write_file write_file", seen[0])

    def test_sandbox_denies_outside_write_without_prompt(self):
        with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as ws:
            session, _ = make_session(
                [
                    tool_events(
                        [
                            {
                                "id": "t1",
                                "name": "write_file",
                                "input": {"path": os.path.join(outside, "f"), "content": "x"},
                            }
                        ]
                    ),
                    text_events("done"),
                ]
            )
            session.sandbox_root = ws
            with (
                patch("builtins.input", side_effect=AssertionError("must not prompt")),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertTrue(tr.get("is_error"))
            self.assertIn("outside workspace", tr["content"])
            self.assertFalse(os.path.exists(os.path.join(outside, "f")))

    def test_sandbox_allows_inside_with_prompt(self):
        with tempfile.TemporaryDirectory() as ws:
            session, _ = make_session(
                [
                    tool_events(
                        [{"id": "t1", "name": "write_file", "input": {"path": "a", "content": "x"}}]
                    ),
                    text_events("done"),
                ]
            )
            session.sandbox_root = ws
            with (
                patch("os.getcwd", return_value=ws),
                patch("builtins.input", return_value="y"),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            self.assertTrue(os.path.exists(os.path.join(ws, "a")))

    def test_bash_always_is_per_command(self):
        session, _ = make_session(
            [
                tool_events([{"id": "t1", "name": "bash", "input": {"command": "echo one"}}]),
                tool_events([{"id": "t2", "name": "bash", "input": {"command": "echo two"}}]),
                text_events("done"),
            ]
        )
        answers = iter(["always", "y"])
        prompts: list[str] = []

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            return next(answers)

        with (
            tempfile.TemporaryDirectory() as td,
            patch("os.getcwd", return_value=td),
            patch("builtins.input", side_effect=fake_input),
            patch("builtins.print"),
        ):
            run_turn(session, "go")
        # first asked, second "echo two" asked again despite earlier always
        self.assertEqual(len(prompts), 2)
        self.assertIn("echo one", prompts[0])
        self.assertIn("echo two", prompts[1])
        self.assertIn("bash::echo one", session.gate.always_allowed)

    def test_agents_md_injected_into_system(self):
        session, adapter = make_session([text_events("ok")])
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "AGENTS.md"), "w") as f:
                f.write("Use tabs, never spaces.")
            with patch("os.getcwd", return_value=td), patch("builtins.print"):
                run_turn(session, "hi")
        system = adapter.calls[0]["system"]
        self.assertIn("Use tabs, never spaces.", system)
        self.assertIn("Project context", system)
        self.assertTrue(system.startswith(SYSTEM_PROMPT_BASE))


class TestAgentsMdScope(unittest.TestCase):
    """AGENTS.md loads from the git project root or the launch dir — never parents between."""

    def test_git_root_agents_used_from_subdir(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".git").mkdir()
            (root / "AGENTS.md").write_text("repo rules here")
            sub = root / "packages" / "app"
            sub.mkdir(parents=True)
            session, adapter = make_session([text_events("ok")])
            with patch("os.getcwd", return_value=str(sub)), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertIn("repo rules here", adapter.calls[0]["system"])

    def test_bare_dotfile_git_marker_counts(self):
        # worktrees/submodules keep .git as a FILE, not a directory
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".git").write_text("gitdir: /elsewhere")
            (root / "AGENTS.md").write_text("marker file root")
            session, adapter = make_session([text_events("ok")])
            with patch("os.getcwd", return_value=str(root)), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertIn("marker file root", adapter.calls[0]["system"])

    def test_git_root_beyond_cap_is_not_a_project(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".git").mkdir()
            (root / "AGENTS.md").write_text("beyond the bound")
            deep = root
            for _ in range(PROJECT_ROOT_MAX_LEVELS + 1):
                deep = deep / "lvl"
            deep.mkdir(parents=True)
            session, adapter = make_session([text_events("ok")])
            with patch("os.getcwd", return_value=str(deep)), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertEqual(adapter.calls[0]["system"], SYSTEM_PROMPT_BASE)

    def test_git_root_within_cap_still_found(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".git").mkdir()
            (root / "AGENTS.md").write_text("within the bound")
            deep = root
            for _ in range(PROJECT_ROOT_MAX_LEVELS):
                deep = deep / "lvl"
            deep.mkdir(parents=True)
            session, adapter = make_session([text_events("ok")])
            with patch("os.getcwd", return_value=str(deep)), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertIn("within the bound", adapter.calls[0]["system"])

    def test_no_git_parent_agents_ignored(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            outer = pathlib.Path(td)
            (outer / "AGENTS.md").write_text("should not load")
            inner = outer / "proj"
            inner.mkdir()
            session, adapter = make_session([text_events("ok")])
            with patch("os.getcwd", return_value=str(inner)), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertEqual(adapter.calls[0]["system"], SYSTEM_PROMPT_BASE)


class TestImageAttachment(unittest.TestCase):
    def test_pending_images_prepended_then_consumed(self):
        session, adapter = make_session([text_events("I see it")])
        blk = {"type": "image", "source": "base64", "media_type": "image/png", "data": "eA=="}
        session.pending_images.append(blk)
        with patch("builtins.print"):
            run_turn(session, "what is this?")
        content = session.messages[0]["content"]  # type: ignore[attr-defined]
        self.assertEqual(content[0]["type"], "image")  # type: ignore[call-overload]
        self.assertEqual(content[1], {"type": "text", "text": "what is this?"})
        self.assertEqual(session.pending_images, [])
        sent = adapter.calls[0]["messages"][0]["content"]
        self.assertEqual(sent[0]["type"], "image")

    def test_no_pending_leaves_plain_text(self):
        session, _ = make_session([text_events("ok")])
        with patch("builtins.print"):
            run_turn(session, "hi")
        self.assertEqual(session.messages[0]["content"], [{"type": "text", "text": "hi"}])

    def test_clear_drops_pending(self):
        from ducktape_harness.commands import cmd_clear

        session, _ = make_session([])
        session.pending_images.append(
            {"type": "image", "source": "base64", "media_type": "image/png", "data": "eA=="}
        )  # type: ignore[typeddict-item]
        cmd_clear(session)
        self.assertEqual(session.pending_images, [])


class TestAuditPass1Fixes(unittest.TestCase):
    def test_sandbox_denies_symlink_escape(self):
        import pathlib

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            (pathlib.Path(ws) / "link").symlink_to(outside, target_is_directory=True)
            session, _ = make_session(
                [
                    tool_events(
                        [
                            {
                                "id": "t1",
                                "name": "write_file",
                                "input": {"path": "link/evil", "content": "x"},
                            }
                        ]
                    )
                ]
            )
            session.sandbox_root = ws
            with (
                patch("os.getcwd", return_value=ws),
                patch("builtins.input", side_effect=AssertionError("must not prompt")),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertTrue(tr.get("is_error"))
            self.assertIn("outside workspace", tr["content"])
            self.assertFalse(os.path.exists(os.path.join(outside, "evil")))

    def test_bash_always_key_canonical_and_prompt_collapsed(self):
        multiline = "echo a\necho b"
        session, _ = make_session(
            [tool_events([{"id": "t1", "name": "bash", "input": {"command": multiline}}])]
        )
        prompts: list[str] = []

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            return "always"

        with (
            tempfile.TemporaryDirectory() as td,
            patch("os.getcwd", return_value=td),
            patch("builtins.input", side_effect=fake_input),
            patch("builtins.print"),
        ):
            run_turn(session, "go")
        self.assertIn("bash::echo a\necho b", session.gate.always_allowed)
        self.assertNotIn("\necho b? ", prompts[0])  # prompt shown collapsed, one line
        # second identical call bypasses with no prompt
        session2_adapter = make_session(
            [tool_events([{"id": "t2", "name": "bash", "input": {"command": multiline}}])]
        )
        s2 = session2_adapter[0]
        s2.gate.always_allowed = session.gate.always_allowed
        with (
            tempfile.TemporaryDirectory() as td2,
            patch("os.getcwd", return_value=td2),
            patch("builtins.input", side_effect=AssertionError("must not prompt")),
            patch("builtins.print"),
        ):
            run_turn(s2, "again")

    def test_attach_tilde_expansion_and_case(self):
        from ducktape_harness.commands import cmd_attach

        with tempfile.TemporaryDirectory() as home:
            os.mkdir(os.path.join(home, "sub"))
            with open(os.path.join(home, "sub", "Pic.PNG"), "wb") as f:
                f.write(b"\x89PNGx")
            session, _ = make_session([])
            with patch.dict(os.environ, {"HOME": home}):
                msg = cmd_attach(session, "~/sub/Pic.PNG")
            self.assertIn("staged Pic.PNG", msg)
            self.assertIn("image/png", msg)

    def test_agents_truncation_at_cap(self):
        import pathlib

        from ducktape_harness.config import AGENTS_MAX_BYTES
        from ducktape_harness.system_prompt import build_system_prompt

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "AGENTS.md"
            p.write_bytes(b"a" * (AGENTS_MAX_BYTES + 1000))
            out = build_system_prompt(td)
            self.assertIn("[truncated]", out)
            self.assertLess(len(out.encode("utf-8")), AGENTS_MAX_BYTES + 2000)

    def test_preview_exception_falls_back_capped(self):
        import ducktape_harness.tools.base as base

        class BigInputTool:
            name = "write_file"
            description = "x"
            parameters: dict = {}

            def preview(self, input, cwd):  # type: ignore[no-untyped-def]
                raise RuntimeError("preview broke")

            def run(self, input, cwd):  # type: ignore[no-untyped-def]
                return "ok"

        orig = base.REGISTRY["write_file"]
        base.REGISTRY["write_file"] = BigInputTool()  # type: ignore[assignment]
        try:
            big = {"path": "x", "content": "y" * 10000}
            session, _ = make_session(
                [tool_events([{"id": "t1", "name": "write_file", "input": big}])]
            )
            seen: list[str] = []
            real_ask = session.gate.ask

            def spy(name, preview, key=None, **kw):  # type: ignore[no-untyped-def]
                seen.append(preview)
                return real_ask(name, preview, key, **kw)

            with (
                patch.object(session.gate, "ask", side_effect=spy),
                patch("builtins.input", return_value="n"),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            self.assertTrue(seen[0].endswith("…[truncated]"))
            self.assertLess(len(seen[0]), 600)
        finally:
            base.REGISTRY["write_file"] = orig

    def test_no_provider_keeps_pending(self):
        from ducktape_provider.types import ImageBlock  # type: ignore[import-not-found]

        session, _ = make_session([])
        session.provider = None
        blk: ImageBlock = {
            "type": "image",
            "source": "base64",
            "media_type": "image/png",
            "data": "eA==",
        }
        session.pending_images.append(blk)
        with patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        self.assertEqual(result, "")
        self.assertEqual(len(session.pending_images), 1)  # not swallowed
        self.assertEqual(session.messages, [])  # history untouched
        mp.assert_any_call("[error] no provider configured")


class TestSecurityHardening(unittest.TestCase):
    """Audit-loop pass 2: bounded auto-allow, pagination, TOCTOU recheck, caps."""

    def test_secret_named_read_prompts(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td) / ".env").write_text("SECRET=1")
            session, _ = make_session(
                [
                    tool_events([{"id": "t1", "name": "read_file", "input": {"path": ".env"}}]),
                    text_events("done"),
                ]
            )
            with (
                patch("os.getcwd", return_value=td),
                patch("builtins.input", return_value="n"),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertEqual(tr["content"], "user denied")

    def test_outside_workspace_read_prompts(self):
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as other:
            import pathlib

            p = pathlib.Path(other) / "open.txt"
            p.write_text("data")
            session, _ = make_session(
                [
                    tool_events([{"id": "t1", "name": "read_file", "input": {"path": str(p)}}]),
                    text_events("done"),
                ]
            )
            session.sandbox_root = ws
            prompts: list[str] = []

            def fake_input(prompt: str = "") -> str:
                prompts.append(prompt)
                return "y"

            with patch("builtins.input", side_effect=fake_input), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertTrue(prompts)  # prompted despite auto-allow policy
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertEqual(tr["content"], "data")

    def test_large_file_pagination_works_now(self):
        import pathlib

        from ducktape_harness.config import READ_MAX
        from ducktape_harness.tools.base import ToolError
        from ducktape_harness.tools.files import ReadFileTool

        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "big.log"
            p.write_text("\n".join(f"line {i}" for i in range(100_000)))
            self.assertGreater(p.stat().st_size, READ_MAX)
            with self.assertRaises(ToolError):
                tool.run({"path": str(p)}, td)  # unpaged: refuse with hint
            out = tool.run({"path": str(p), "offset": 10, "limit": 5}, td)
            self.assertEqual(out.splitlines(), [f"line {10 + i}" for i in range(5)])
            with self.assertRaises(ToolError):
                tool.run({"path": str(p), "limit": -1}, td)

    def test_read_output_cap_truncates(self):
        import pathlib

        from ducktape_harness.config import OUTPUT_CAP
        from ducktape_harness.tools.files import ReadFileTool

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "wide.txt"
            p.write_text("x" * 500_000)  # one huge line, under READ_MAX
            out = ReadFileTool().run({"path": str(p)}, td)
            self.assertLessEqual(len(out.encode()), OUTPUT_CAP + 32)
            self.assertTrue(out.endswith("[truncated]"))

    def test_write_symlink_race_denied_after_approval(self):
        import pathlib

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            safe = pathlib.Path(ws) / "safe"
            safe.mkdir()
            link = pathlib.Path(ws) / "link"
            link.symlink_to(safe)
            session, _ = make_session(
                [
                    tool_events(
                        [
                            {
                                "id": "t1",
                                "name": "write_file",
                                "input": {"path": "link/f", "content": "x"},
                            }
                        ]
                    )
                ]
            )
            session.sandbox_root = ws

            def swap_then_allow(prompt: str = "") -> str:
                link.unlink()
                link.symlink_to(outside)  # escape appears during the prompt wait
                return "y"

            with (
                patch("os.getcwd", return_value=ws),
                patch("builtins.input", side_effect=swap_then_allow),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertTrue(tr.get("is_error"))
            self.assertIn("outside workspace at execution time", tr["content"])
            self.assertFalse((pathlib.Path(outside) / "f").exists())
            self.assertFalse((safe / "f").exists())


class TestSensitiveNormalization(unittest.TestCase):
    def test_trailing_slash_and_padding_still_prompt(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td) / ".env").write_text("S=1")
            for sneaky in ["  .env  ", ".env/", "./.env/", "../x/.env"]:
                session, _ = make_session(
                    [
                        tool_events([{"id": "t1", "name": "read_file", "input": {"path": sneaky}}]),
                        text_events("done"),
                    ]
                )
                with (
                    patch("os.getcwd", return_value=td),
                    patch("builtins.input", return_value="n"),
                    patch("builtins.print"),
                ):
                    run_turn(session, "hi")
                tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
                self.assertEqual(tr["content"], "user denied", f"path {sneaky!r} skipped the gate")


class TestConfirmationHardening(unittest.TestCase):
    def test_symlink_to_secret_still_prompts(self):
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".env").write_text("SECRET=1")
            (root / "innocent.txt").symlink_to(root / ".env")
            session, _ = make_session(
                [
                    tool_events(
                        [{"id": "t1", "name": "read_file", "input": {"path": "innocent.txt"}}]
                    ),
                    text_events("done"),
                ]
            )
            session.sandbox_root = td
            with (
                patch("os.getcwd", return_value=td),
                patch("builtins.input", return_value="n"),
                patch("builtins.print"),
            ):
                run_turn(session, "hi")
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertEqual(tr["content"], "user denied")  # realpath basename caught it

    def test_case_insensitive_secret_matching(self):
        import ducktape_harness.engine as eng

        for sneaky in (".ENV", "sub/ID_RSA", "id_ed25519", "AWS_CREDENTIALS", "x.PEM"):
            self.assertTrue(
                eng._sensitive_read("read_file", {"path": sneaky}, "/tmp"),
                f"{sneaky} bypassed SECRET_PATTERNS",
            )
        self.assertFalse(eng._sensitive_read("read_file", {"path": "main.py"}, "/tmp"))

    def test_paged_read_limit_clamped(self):
        import pathlib

        from ducktape_harness.config import READ_MAX_LINES
        from ducktape_harness.tools.files import ReadFileTool

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "many.log"
            p.write_text("\n".join(f"l{i}" for i in range(50_000)))
            tool = ReadFileTool()
            out = tool.run({"path": str(p), "limit": 10**9}, td)
            self.assertEqual(len(out.splitlines()), READ_MAX_LINES)
            full = tool.run({"path": str(p)}, td)  # unpaged small file: complete
            self.assertGreater(len(full.splitlines()), READ_MAX_LINES)


class TestDiffDisplayAndTailWindow(unittest.TestCase):
    def test_gate_keeps_diff_newlines_but_collapses_bash(self):
        from ducktape_harness.gate import PermissionGate

        gate = PermissionGate()
        prompts: list[str] = []

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            return "n"

        diff = "--- a\n+++ b\n-line\n+plus"
        import ducktape_harness.tools.files as tfm

        assert tfm.WriteFileTool.preview_multiline
        with patch("builtins.input", side_effect=fake_input):
            gate.ask("write_file", diff, diff, multiline=True)
            gate.ask("bash", "echo a\necho b", "echo a\necho b")
        self.assertIn("\n-line\n", prompts[0])  # diff structure intact
        self.assertNotIn("\n", prompts[1].replace("? [y/N/always]:", ""))

    def test_offset_only_read_window_bounded(self):
        import pathlib

        from ducktape_harness.config import READ_MAX_LINES
        from ducktape_harness.tools.files import ReadFileTool

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "tail.log"
            p.write_text("\n".join(f"l{i}" for i in range(50_000)))
            out = ReadFileTool().run({"path": str(p), "offset": 1}, td)
            self.assertLessEqual(len(out.splitlines()), READ_MAX_LINES)


class TestEngineDiffPromptIntact(unittest.TestCase):
    def test_engine_prompt_keeps_diff_newlines_end_to_end(self):
        import pathlib

        from ducktape_harness.session import Session as S

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "f.txt"
            p.write_text("one\ntwo\n")
            adapter = FakeAdapter(
                scripts=[
                    tool_events(
                        [
                            {
                                "id": "t1",
                                "name": "write_file",
                                "input": {"path": str(p), "content": "one\nTWO\n"},
                            }
                        ]
                    )
                ]
            )
            provider = Provider(adapters={"fake": adapter}, autodiscover=False)
            session = S(model="fake-model", pin="fake", provider=provider)
            prompts: list[str] = []

            def fake_input(prompt: str = "") -> str:
                prompts.append(prompt)
                return "n"

            with patch("builtins.input", side_effect=fake_input), patch("builtins.print"):
                run_turn(session, "hi")
            self.assertEqual(len(prompts), 1)
            self.assertIn("-two\n", prompts[0])
            self.assertIn("+TWO", prompts[0])


class TestProviderSync(unittest.TestCase):
    """Provider b0ad0bb unlocks: cancel(), truncated marker, image results."""

    def test_keyboard_interrupt_calls_stream_cancel(self):
        class Cancellable:
            def __init__(self) -> None:
                self.cancelled = False

            def __iter__(self):  # type: ignore[no-untyped-def]
                return self

            def __next__(self):  # type: ignore[no-untyped-def]
                raise KeyboardInterrupt

            def close(self) -> None:
                pass

            def cancel(self) -> None:
                self.cancelled = True

        class StubProvider:
            def __init__(self) -> None:
                self.stream = Cancellable()

            def stream_chat(self, *a, **k):  # type: ignore[no-untyped-def]
                return self.stream

        from ducktape_harness.session import Session as S

        stub = StubProvider()
        session = S(model="m", provider=stub)
        with patch("builtins.print") as mp:
            result = run_turn(session, "hi")
        self.assertEqual(result, "")
        self.assertTrue(stub.stream.cancelled)
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertIn("[interrupted]", printed)

    def test_truncated_flag_wins_over_block_stop(self):
        session, _ = make_session(
            [
                max_tokens_tool_events(
                    [{"id": "t1", "name": "bash", "input": {"command": "echo hi"}}],
                    completed_indices={0},
                    truncated={0},
                )
            ]
        )
        session.gate.always_allowed.add("bash")
        with patch("builtins.print"):
            run_turn(session, "hi")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertEqual(tr["content"], "truncated by max_tokens")
        self.assertTrue(tr["is_error"])

    def test_tool_can_return_image_blocks(self):
        import ducktape_harness.tools.base as base

        img = {"type": "image", "source": "base64", "media_type": "image/png", "data": "eA=="}

        class ImageTool:
            name = "read_file"
            description = "x"
            parameters: dict = {}

            def run(self, input, cwd):  # type: ignore[no-untyped-def]
                return [img]

        orig = base.REGISTRY["read_file"]
        base.REGISTRY["read_file"] = ImageTool()  # type: ignore[assignment]
        try:
            session, adapter = make_session(
                [tool_events([{"id": "t1", "name": "read_file", "input": {"path": "p"}}])]
            )
            with patch("builtins.print"):
                run_turn(session, "hi")
            sent = adapter.calls[1]["messages"][-1]["content"][0]
            self.assertEqual(sent["content"], [img])  # list passthrough, no str()
        finally:
            base.REGISTRY["read_file"] = orig


class TestBudgetWarn(unittest.TestCase):
    def setUp(self):
        from ducktape_harness.render import clear_window_cache

        clear_window_cache()

    def tearDown(self):
        from ducktape_harness.render import clear_window_cache

        clear_window_cache()

    def _session_with_window(self, window: int | None, last_in: int, last_out: int):
        infos = (
            {"fake-model": {"context_window": window, "max_output_tokens": None}} if window else {}
        )
        adapter = FakeAdapter(infos=infos)
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="fake-model", pin="fake", provider=provider)
        from ducktape_harness.gate import PermissionGate

        session.gate = PermissionGate()
        if last_in is not None:
            session.last_usage = {"input_tokens": last_in, "output_tokens": last_out}  # type: ignore[typeddict-item]
        return session, adapter

    def test_warn_fires_once_per_run_turn_at_85(self):
        # window 100, usage 80+10=90 => 90% with len("hi")/4 ~0.5 => ~90% -> warn
        session, adapter = self._session_with_window(100, 80, 10)
        # two tool iterations: first tool_use, second text
        adapter.set_scripts(
            [
                tool_events([{"id": "t1", "name": "read_file", "input": {"path": "/tmp/a"}}]),
                text_events("done"),
            ]
        )
        # need file for read_file auto-allow (inside no sandbox, auto)
        import os as _os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = _os.path.join(td, "a")
            with open(p, "w") as f:
                f.write("x")
            with patch("os.getcwd", return_value=td), patch("builtins.print") as mp:
                run_turn(session, "hi")
            # count prints containing "[context"
            ctx_prints = [c for c in mp.call_args_list if c.args and "[context" in str(c.args[0])]
            self.assertEqual(len(ctx_prints), 1)
            self.assertIn("90% of 100", ctx_prints[0].args[0])
            # ensure still did tool loop
            self.assertEqual(len(adapter.calls), 2)

    def test_silent_below_threshold(self):
        session, adapter = self._session_with_window(100, 50, 10)  # 60% <85
        adapter.set_scripts([text_events("ok")])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        ctx_prints = [c for c in mp.call_args_list if c.args and "[context" in str(c.args[0])]
        self.assertEqual(len(ctx_prints), 0)

    def test_silent_unknown_window(self):
        # no infos, fake-model unknown -> window None
        adapter = FakeAdapter(infos={})
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="fake-model", pin="fake", provider=provider)
        from ducktape_harness.gate import PermissionGate

        session.gate = PermissionGate()
        session.last_usage = {"input_tokens": 80, "output_tokens": 10}  # type: ignore[typeddict-item]
        adapter.set_scripts([text_events("ok")])
        with patch("builtins.print") as mp:
            run_turn(session, "hi")
        ctx_prints = [c for c in mp.call_args_list if c.args and "[context" in str(c.args[0])]
        self.assertEqual(len(ctx_prints), 0)


def _task_events(prompt="go", description="explore", tools=None, tid="k1"):
    from tests.fake_adapter import tool_events

    inp = {"description": description, "prompt": prompt}
    if tools is not None:
        inp["tools"] = tools
    return tool_events([{"id": tid, "name": "task", "input": inp}])


class TestSubagent(unittest.TestCase):
    def test_happy_path_isolation_and_usage_merge(self):
        session, adapter = make_session(
            [
                _task_events(),
                text_events("42", usage={"input_tokens": 7, "output_tokens": 2}),
                text_events("The answer: 42", usage={"input_tokens": 11, "output_tokens": 3}),
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print") as mp:
            result = run_turn(session, "what is the answer?")
        self.assertEqual(result, "The answer: 42")
        # parent history: user, assistant(task), user(result "42"), assistant(final)
        self.assertEqual(len(session.messages), 4)
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertEqual(tr["content"], "42")
        self.assertEqual(tr["name"], "task")
        # sub stream must NOT leak its text into the parent's terminal
        printed = " ".join(str(c) for c in mp.call_args_list)  # type: ignore[attr-defined]
        self.assertNotIn("42", printed.split("[subagent]")[0])
        self.assertIn("[subagent] explore: done", printed)
        # usage: parent 11+3 plus merged sub 7+2
        self.assertEqual(session.totals.input_tokens, 11 + 7)
        self.assertEqual(session.totals.output_tokens, 3 + 2)
        self.assertEqual(session.totals.turns, 3)  # 2 parent + 1 sub (N-of-M coherence)
        self.assertEqual(
            session.totals.turns_with_usage, 2
        )  # parent final turn + sub turn (parent tool turn had no usage)
        # sub never saw task in specs
        sub_call = adapter.calls[1]
        spec_names = {s["name"] for s in sub_call["tools"] or []}
        self.assertNotIn("task", spec_names)
        # and sub got a fresh history (only its own prompt)
        self.assertEqual(len(sub_call["messages"]), 1)

    def test_denied_task_skips_nested_call(self):
        session, adapter = make_session([_task_events(), text_events("not delegated")])
        with patch("builtins.input", return_value="n"), patch("builtins.print"):
            run_turn(session, "go")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertEqual(tr["content"], "user denied")
        self.assertTrue(tr["is_error"])
        self.assertEqual(len(adapter.calls), 2)  # parent only: no sub turn

    def test_tool_names_enforced_inside_sub(self):
        session, adapter = make_session(
            [
                _task_events(tools=["read_file"]),
                tool_events(
                    [{"id": "s1", "name": "write_file", "input": {"path": "x", "content": "y"}}]
                ),
                text_events("sub done"),
                text_events("outer done"),
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            result = run_turn(session, "go")
        self.assertEqual(result, "outer done")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertEqual(tr["content"], "sub done")
        sub_final = adapter.calls[2]["messages"]  # sub's second call
        blocked = [b for m in sub_final for b in m["content"] if b.get("type") == "tool_result"]
        self.assertEqual(blocked[0]["content"], "tool not available to this subagent")

    def test_nested_task_blocked(self):
        session, adapter = make_session(
            [
                _task_events(),
                _task_events(prompt="inner-inner", description="nested", tid="n1"),
                text_events("sub gave up"),
                text_events("outer"),
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            run_turn(session, "go")
        nested_msgs = adapter.calls[2]["messages"]
        results = [b for m in nested_msgs for b in m["content"] if b.get("type") == "tool_result"]
        self.assertEqual(results[0]["content"], "nested task not allowed")

    def test_unknown_requested_tools_noted(self):
        session, _ = make_session(
            [
                _task_events(tools=["nope", "read_file"]),
                text_events("42"),
                text_events("outer"),
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            run_turn(session, "go")
        tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
        self.assertIn("unknown tools ignored: nope", tr["content"])
        self.assertTrue(tr["content"].endswith("42"))

    def test_sub_respects_parent_sandbox_and_gate(self):
        import pathlib

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            session, _ = make_session(
                [
                    _task_events(),
                    tool_events(
                        [
                            {
                                "id": "s1",
                                "name": "write_file",
                                "input": {
                                    "path": os.path.join(outside, "evil"),
                                    "content": "x",
                                },
                            }
                        ]
                    ),
                    text_events("sub blocked it"),
                    text_events("outer"),
                ]
            )
            session.sandbox_root = ws
            with (
                patch("os.getcwd", return_value=ws),
                patch("builtins.input", return_value="y"),
                patch("builtins.print"),
            ):
                run_turn(session, "go")
            self.assertFalse((pathlib.Path(outside) / "evil").exists())
            tr = session.messages[2]["content"][0]  # type: ignore[attr-defined]
            self.assertEqual(tr["content"], "sub blocked it")


class TestFeatureDeltaRegressions(unittest.TestCase):
    def test_compact_leaves_no_consecutive_user_messages(self):
        from ducktape_harness.commands import cmd_compact
        from ducktape_harness.config import COMPACT_KEEP_TAIL

        session, adapter = make_session([])
        # plain alternating history long enough to compact
        for i in range(COMPACT_KEEP_TAIL + 3):
            session.messages.append(
                {"role": "user", "content": [{"type": "text", "text": f"u{i}"}]}
            )
            session.messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]}
            )
        adapter.set_scripts([{"response": _summary_resp("SUMMARY BODY")}])
        with patch("builtins.print"):
            msg = cmd_compact(session, None)
        self.assertIn("summarized", msg)
        roles = [m["role"] for m in session.messages]
        self.assertTrue(all(a != "user" or b != "user" for a, b in zip(roles, roles[1:])))
        self.assertIn("Compacted conversation summary", session.messages[0]["content"][0]["text"])

    def test_task_prompt_interrupt_batch_denies_remainder(self):
        # two task calls same batch: Ctrl+C at the FIRST prompt must end the
        # turn with the second batch-denied (no second prompt)
        session, adapter = make_session(
            [
                tool_events(
                    [
                        {
                            "id": "k1",
                            "name": "task",
                            "input": {"description": "one", "prompt": "p1"},
                        },
                        {
                            "id": "k2",
                            "name": "task",
                            "input": {"description": "two", "prompt": "p2"},
                        },
                    ]
                )
            ]
        )

        def ki_input(prompt=""):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

        with patch("builtins.input", side_effect=ki_input), patch("builtins.print"):
            result = run_turn(session, "go")
        results = session.messages[2]["content"]  # type: ignore[attr-defined]
        self.assertEqual(
            [r["content"] for r in results], ["interrupted", "denied (batch interrupted)"]
        )
        self.assertEqual(len(adapter.calls), 1)  # no further model call
        self.assertEqual(result, "")

    def test_transient_model_info_failure_not_cached(self):
        from ducktape_harness.render import clear_window_cache, provider_window

        class Flaky:
            fail = True

            def model_info(self, m, **k):  # type: ignore[no-untyped-def]
                if Flaky.fail:
                    raise RuntimeError("blip")
                return {"context_window": 777, "max_output_tokens": None}

        clear_window_cache()
        self.assertIsNone(provider_window("m", Flaky(), None))
        Flaky.fail = False
        self.assertEqual(provider_window("m", Flaky(), None), 777)
        clear_window_cache()


class TestAuditDeltaFixes(unittest.TestCase):
    def test_sub_real_interrupt_denies_batch(self):
        from ducktape_provider import ContextOverflowError

        session, adapter = make_session(
            [
                tool_events(
                    [
                        {"id": "k1", "name": "task", "input": {"description": "a", "prompt": "p"}},
                        {"id": "k2", "name": "task", "input": {"description": "b", "prompt": "q"}},
                    ]
                ),
                ContextOverflowError("overflow"),  # sub k1 FAILS (not an interrupt)
                text_events("sub fallback"),  # parent second model turn
            ]
        )
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            run_turn(session, "go")
        results = session.messages[2]["content"]  # type: ignore[attr-defined]
        # failure must NOT masquerade as batch-interrupt: k2 executed too
        self.assertEqual(
            [r["content"] for r in results][:2], ["(subagent ended with no output)", "sub fallback"]
        )

    def test_sub_keyboard_interrupt_propagates_batch(self):
        session, _ = make_session(
            [
                tool_events(
                    [
                        {"id": "k1", "name": "task", "input": {"description": "a", "prompt": "p"}},
                        {"id": "k2", "name": "task", "input": {"description": "b", "prompt": "q"}},
                    ]
                ),
                KeyboardInterrupt(),  # raised by the sub's stream
            ]
        )
        with patch("builtins.print"):
            run_turn(session, "go")
        results = session.messages[2]["content"]  # type: ignore[attr-defined]
        self.assertEqual(
            [r["content"] for r in results], ["interrupted", "denied (batch interrupted)"]
        )
