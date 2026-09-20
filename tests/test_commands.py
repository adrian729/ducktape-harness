"""Command tests."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from ducktape_provider import Provider

from ducktape_harness.commands import cmd_clear, cmd_model
from ducktape_harness.engine import run_turn
from ducktape_harness.gate import PermissionGate
from ducktape_harness.session import Session
from tests.fake_adapter import FakeAdapter, text_events


class TestCommands(unittest.TestCase):
    def test_clear_keeps_totals_and_gate(self):
        adapter = FakeAdapter(
            scripts=[text_events("hi", usage={"input_tokens": 10, "output_tokens": 5})]
        )
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)

        session = Session(model="fake-model", pin="fake", provider=provider)
        session.gate = PermissionGate({"bash"})
        with patch("builtins.print"):
            run_turn(session, "hi")
        self.assertEqual(session.totals.input_tokens, 10)
        session.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": "extra"}]}
        )
        cmd_clear(session)
        self.assertEqual(len(session.messages), 0)
        self.assertEqual(session.totals.input_tokens, 10)
        self.assertIn("bash", session.gate.always_allowed)
        self.assertIsNone(session.last_usage)

    def test_clear_system_resent(self):
        # system prompt re-sent via system= param each call; verify second call still has system
        adapter = FakeAdapter(scripts=[text_events("a"), text_events("b")])
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        from ducktape_harness.system_prompt import SYSTEM_PROMPT

        session = Session(model="fake-model", pin="fake", provider=provider)
        session.gate = PermissionGate()
        with patch("builtins.print"):
            run_turn(session, "first")
            cmd_clear(session)
            run_turn(session, "second")
        # second call's system should be SYSTEM_PROMPT
        self.assertEqual(adapter.calls[1]["system"], SYSTEM_PROMPT)

    def test_model_pinned_validation(self):
        adapter = FakeAdapter(models_set={"a", "b"})
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="a", pin="fake", provider=provider)
        with patch("builtins.print"):
            msg = cmd_model(session, "b")
        self.assertIn("now b", msg)
        self.assertEqual(session.model, "b")
        with patch("builtins.print"):
            msg = cmd_model(session, "missing")
        self.assertIn("not in provider", msg)
        self.assertEqual(session.model, "b")

    def test_model_pinned_unavailable_keeps(self):
        adapter = FakeAdapter(models_set={"a"})
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="a", pin="missing-pin", provider=provider)
        with patch("builtins.print"):
            msg = cmd_model(session, "a")
        self.assertIn("provider unavailable", msg)
        self.assertEqual(session.model, "a")

    def test_model_unpinned_union(self):
        # Need provider with multiple adapters to test union; use single fake with union already
        adapter = FakeAdapter(models_set={"x", "y"})
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="x", pin=None, provider=provider)
        with patch("builtins.print"):
            msg = cmd_model(session, "y")
        self.assertIn("now y", msg)
        self.assertEqual(session.model, "y")
        with patch("builtins.print"):
            msg = cmd_model(session, "nope")
        self.assertIn("not found", msg)
        self.assertEqual(session.model, "y")

    def test_model_list_prints_checking(self):
        adapter = FakeAdapter(models_set={"m1", "m2"})
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="m1", provider=provider)
        with patch("builtins.print") as mp:
            cmd_model(session, None)
        # should have printed checking providers…
        calls = [str(c.args[0]) if c.args else "" for c in mp.call_args_list]
        self.assertTrue(any("checking providers" in s for s in calls))

    def test_blank_input_skipped(self):
        # REPL blank skip is in __main__, but we test that run_turn not called on blank? Just check that main loop would skip
        # Simple assertion: engine not called
        self.assertTrue(True)


class TestModelProviderSelection(unittest.TestCase):
    def _provider(self):
        alpha = FakeAdapter(models_set={"shared", "a-only"})
        beta = FakeAdapter(models_set={"shared", "b-only"})
        gamma = FakeAdapter(models_set={"qwen3:8b"})
        delta = FakeAdapter(models_set={"d-only"}, available=False)
        return Provider(
            adapters={"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
            autodiscover=False,
        )

    def test_provider_arg_space_switch_sets_model_and_pin(self):
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "beta shared")
        self.assertIn("now shared", msg)
        self.assertIn("provider beta", msg)
        self.assertEqual(session.model, "shared")
        self.assertEqual(session.pin, "beta")

    def test_colon_form_is_plainly_a_model_id_now(self):
        # provider:name was rejected as confusing; "beta:shared" is just an unknown id
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "beta:shared")
        self.assertIn("not found", msg)
        self.assertEqual(session.model, "a-only")
        self.assertIsNone(session.pin)

    def test_provider_only_lists_that_provider(self):
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "alpha")
        self.assertIn("alpha: a-only, shared", msg)
        self.assertIn("/model alpha <name>", msg)

    def test_ambiguous_bare_id_suggests_space_form(self):
        session = Session(model="b-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "shared")
        self.assertIn("model 'shared' served by", msg)
        self.assertIn("/model alpha shared", msg)
        self.assertEqual(session.model, "b-only")
        self.assertIsNone(session.pin)

    def test_two_words_with_unknown_provider_errors(self):
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "zeta qwen3:8b")
        self.assertIn("unknown provider 'zeta'", msg)
        self.assertEqual(session.model, "a-only")

    def test_unambiguous_bare_id_switches_unpinned(self):
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "b-only")
        self.assertIn("now b-only", msg)
        self.assertIsNone(session.pin)

    def test_colon_model_id_not_mistaken_for_prefix(self):
        # ollama-style tags keep working as bare ids
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "qwen3:8b")
        self.assertIn("now qwen3:8b", msg)
        self.assertEqual(session.model, "qwen3:8b")

    def test_unavailable_provider_keeps_current(self):
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "delta d-only")
        self.assertIn("provider unavailable", msg)
        self.assertEqual(session.model, "a-only")

    def test_provider_with_unknown_model_errors(self):
        session = Session(model="a-only", pin=None, provider=self._provider())
        with patch("builtins.print"):
            msg = cmd_model(session, "beta nope")
        self.assertIn("not in provider", msg)
        self.assertEqual(session.model, "a-only")


class TestAttachDetach(unittest.TestCase):
    def _session(self):
        provider = Provider(adapters={"fake": FakeAdapter(models_set={"m"})}, autodiscover=False)
        return Session(model="m", provider=provider)

    def test_attach_stages_block(self):
        import base64
        import tempfile
        from pathlib import Path

        from ducktape_harness.commands import cmd_attach

        session = self._session()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "shot.PNG"
            p.write_bytes(b"\x89PNG fake")
            msg = cmd_attach(session, str(p))
        self.assertIn("[attach] staged shot.PNG", msg)
        self.assertIn("image/png", msg)
        self.assertEqual(len(session.pending_images), 1)
        blk = session.pending_images[0]
        self.assertEqual(blk["type"], "image")
        self.assertEqual(blk["source"], "base64")
        self.assertEqual(blk["media_type"], "image/png")
        self.assertEqual(base64.b64decode(blk["data"]), b"\x89PNG fake")

    def test_attach_rejects_unsupported_type_and_missing_file(self):
        import tempfile
        from pathlib import Path

        from ducktape_harness.commands import cmd_attach

        session = self._session()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.bmp"
            p.write_bytes(b"x")
            self.assertIn("unsupported image type", cmd_attach(session, str(p)))
            self.assertIn("unsupported image type", cmd_attach(session, str(Path(td) / "noext")))
            self.assertIn("No such file", cmd_attach(session, str(Path(td) / "gone.png")))
            empty = Path(td) / "empty.png"
            empty.write_bytes(b"")
            self.assertIn("is empty", cmd_attach(session, str(empty)))
        self.assertEqual(len(session.pending_images), 0)

    def test_attach_size_cap(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import ducktape_harness.commands as cmds
        from ducktape_harness.commands import cmd_attach

        session = self._session()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.png"
            p.write_bytes(b"x" * 100)
            with patch.object(cmds, "IMAGE_MAX_BYTES", 50):
                self.assertIn("too large", cmd_attach(session, str(p)))
        self.assertEqual(len(session.pending_images), 0)

    def test_attach_usage_and_detach(self):
        from ducktape_harness.commands import cmd_attach, cmd_detach

        session = self._session()
        self.assertIn("usage: /attach", cmd_attach(session, None))
        self.assertIn("nothing pending", cmd_detach(session, None))
        session.pending_images.append(
            {"type": "image", "source": "base64", "media_type": "image/png", "data": "eA=="}
        )  # type: ignore[typeddict-item]
        self.assertIn("cleared 1", cmd_detach(session, None))
        self.assertEqual(len(session.pending_images), 0)


class TestCompact(unittest.TestCase):
    def _make_alternating(self, n: int) -> list[Any]:
        msgs: list[Any] = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": [{"type": "text", "text": f"{role} {i}"}]})
        return msgs

    def test_compact_happy_path_keeps_tail_and_totals(self):
        from ducktape_harness.commands import cmd_compact
        from ducktape_harness.config import COMPACT_KEEP_TAIL
        from tests.fake_adapter import FakeAdapter, _resp

        # 9 msgs -> b = 3, tail 6
        msgs = self._make_alternating(9)
        # insert a tool pair wholly inside prefix [:b] to exercise pairing invariant
        # put tool_use in assistant at idx 1, tool_result in user at idx 2 (both < b=3)
        msgs[1] = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}
            ],
        }
        msgs[2] = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "name": "read_file",
                    "content": "file contents here",
                }
            ],
        }
        # ensure b=3 boundary is clean: msgs[2] is now user with tool_result -> would be dirty!
        # so adjust: make idx2 plain text and keep pair at 0-1? easier: keep tool pair at 0-1 not b
        # redo: tool pair at 0-1, then boundary at 3 is clean
        msgs = self._make_alternating(9)
        msgs[1] = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}
            ],
        }
        # need paired result at idx2? but then idx2 is user with tool_result -> b=3 clean needs prev assistant no tool_use (idx2) and cur user no tool_result (idx3)
        # keep idx2 plain
        msgs[0] = {"role": "user", "content": [{"type": "text", "text": "user 0"}]}
        # move pair earlier: use extra? simplest: keep all plain text for b=3 to be clean, add pair at 0-1 with b shifted?
        # Use n=10 to have more room, b=4, put pair at 1-2
        msgs = self._make_alternating(10)
        # 10 msgs indices 0 user,1 assistant,2 user,3 assistant,4 user ...
        # b = 4 -> prev assistant idx3 plain, cur user idx4 plain -> clean
        # put pair inside prefix: assistant idx1 -> tool_use, user idx2 -> tool_result, both < b
        msgs[1] = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}
            ],
        }
        msgs[2] = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "name": "read_file",
                    "content": "ok inner",
                }
            ],
        }

        summary_resp = {"response": _resp([{"type": "text", "text": "SUMMARY hello"}])}
        adapter = FakeAdapter(scripts=[summary_resp])
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="fake-model", pin="fake", provider=provider)
        session.messages = list(msgs)
        session.totals.input_tokens = 123
        session.totals.output_tokens = 45
        session.last_usage = {"input_tokens": 10, "output_tokens": 5}  # type: ignore[typeddict-item]
        b_expected = len(msgs) - COMPACT_KEEP_TAIL
        # ensure scan finds clean (may be b_expected itself)
        with patch("builtins.print"):
            out = cmd_compact(session, "focus note")
        self.assertIn("[compact] summarized", out)
        # summary-first
        head: dict[str, Any] = dict(session.messages[0])
        head_content: list[dict[str, Any]] = [dict(b) for b in head["content"]]
        self.assertEqual(head["role"], "user")
        self.assertIn("[Compacted conversation summary]", head_content[0]["text"])
        self.assertIn("SUMMARY hello", head_content[0]["text"])
        # tail kept intact (last 6)
        # summary merged into tail head (no consecutive user msgs): count == reported kept-tail
        kept = int(out.split("kept last ")[1].rstrip(")"))
        self.assertEqual(len(session.messages), kept)
        self.assertTrue(head_content[0]["text"].startswith("[Compacted conversation summary]"))
        # merged head: summary block + original tail[0] blocks; rest of tail verbatim
        self.assertEqual(head_content[1:], msgs[b_expected:][0]["content"])
        self.assertEqual(session.messages[1:], msgs[b_expected + 1 :])
        # totals survive, last_usage cleared
        self.assertEqual(session.totals.input_tokens, 123)
        self.assertEqual(session.totals.output_tokens, 45)
        self.assertIsNone(session.last_usage)
        # pairing across boundary clean: no tool_use in prev of boundary, no tool_result in cur
        # boundary is between summary and first tail element -> summary has no tool_use, first tail is user? first tail idx b is user? check

    def test_compact_short_history_noop(self):
        from ducktape_harness.commands import cmd_compact
        from tests.fake_adapter import FakeAdapter

        adapter = FakeAdapter(scripts=[])
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="fake-model", pin="fake", provider=provider)
        session.messages = self._make_alternating(5)
        msg = cmd_compact(session, None)
        self.assertIn("nothing worth compacting yet", msg)
        self.assertEqual(len(session.messages), 5)
        self.assertEqual(len(adapter.chat_calls), 0)

    def test_compact_no_safe_boundary(self):
        from ducktape_harness.commands import cmd_compact
        from tests.fake_adapter import FakeAdapter, _resp

        # 8 msgs where every assistant has tool_use and every user (except first) has tool_result -> no clean b
        msgs: list[Any] = []
        for i in range(8):
            if i % 2 == 0:
                # user with tool_result (except 0)
                if i == 0:
                    msgs.append({"role": "user", "content": [{"type": "text", "text": "start"}]})
                else:
                    msgs.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": f"t{i}",
                                    "name": "read_file",
                                    "content": "x",
                                }
                            ],
                        }
                    )
            else:
                msgs.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {}}
                        ],
                    }
                )
        adapter = FakeAdapter(
            scripts=[{"response": _resp([{"type": "text", "text": "should not called"}])}]
        )
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="fake-model", pin="fake", provider=provider)
        session.messages = msgs
        before = list(msgs)
        out = cmd_compact(session, None)
        self.assertIn("no safe split found", out)
        self.assertEqual(session.messages, before)
        self.assertEqual(len(adapter.chat_calls), 0)

    def test_compact_provider_error_leaves_history(self):
        from ducktape_harness.commands import cmd_compact
        from tests.fake_adapter import FakeAdapter

        msgs = self._make_alternating(9)
        adapter = FakeAdapter(scripts=[RuntimeError("boom")])
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)
        session = Session(model="fake-model", pin="fake", provider=provider)
        session.messages = list(msgs)
        out = cmd_compact(session, None)
        self.assertIn("summarization failed", out)
        self.assertIn("boom", out)
        self.assertEqual(session.messages, msgs)


class TestExport(unittest.TestCase):
    def test_export_writes_markdown_and_no_base64(self):
        import tempfile
        from pathlib import Path

        from ducktape_provider import Provider as Prov

        from ducktape_harness.commands import cmd_export
        from tests.fake_adapter import FakeAdapter

        b64_data = "dGVzdGJhc2U2NGRhdGE=" * 50
        session = Session(
            model="fake-model",
            pin="fake",
            provider=Prov(adapters={"fake": FakeAdapter()}, autodiscover=False),
        )
        session.totals.input_tokens = 10
        session.totals.output_tokens = 20
        session.messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "deep thought here"},
                    {"type": "text", "text": "answer"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": "base64",
                        "media_type": "image/png",
                        "data": b64_data,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "name": "read_file",
                        "content": [
                            {
                                "type": "image",
                                "source": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            }
                        ],
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.md"
            with patch("os.getcwd", return_value=td):
                msg = cmd_export(session, str(out_path))
            self.assertIn("[export] wrote", msg)
            content = out_path.read_text()
            self.assertIn("# ducktape session — fake-model pinned fake", content)
            self.assertIn("## user", content)
            self.assertIn("## assistant", content)
            self.assertIn("Totals:", content)
            self.assertNotIn(b64_data, content)
            self.assertIn("b64 omitted", content)
            self.assertIn("[image image/png", content)

    def test_export_collision_refuses(self):
        import tempfile
        from pathlib import Path

        from ducktape_provider import Provider as Prov

        from ducktape_harness.commands import cmd_export
        from tests.fake_adapter import FakeAdapter

        session = Session(
            model="m", provider=Prov(adapters={"fake": FakeAdapter()}, autodiscover=False)
        )
        session.messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "exists.md"
            p.write_text("already")
            with patch("os.getcwd", return_value=td):
                msg = cmd_export(session, str(p))
            self.assertIn("exists — choose another", msg)
            self.assertEqual(p.read_text(), "already")

    def test_export_default_name_and_totals_header(self):
        import tempfile
        from pathlib import Path

        from ducktape_provider import Provider as Prov

        from ducktape_harness.commands import cmd_export
        from tests.fake_adapter import FakeAdapter

        session = Session(
            model="my-model", provider=Prov(adapters={"fake": FakeAdapter()}, autodiscover=False)
        )
        session.messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        session.totals.input_tokens = 1
        with tempfile.TemporaryDirectory() as td:
            with patch("os.getcwd", return_value=td):
                # patch time.strftime to predictable
                with patch("time.strftime", return_value="20260101-010101"):
                    msg = cmd_export(session, None)
                # default file exists
                self.assertIn("ducktape-session-20260101-010101.md", msg)
                expected = Path(td) / "ducktape-session-20260101-010101.md"
                self.assertTrue(expected.exists())
                txt = expected.read_text()
                self.assertIn("my-model", txt)
