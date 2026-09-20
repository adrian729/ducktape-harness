"""Session usage and footer tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ducktape_provider import Provider

import ducktape_harness.tools  # noqa: F401
from ducktape_harness.engine import run_turn
from ducktape_harness.gate import PermissionGate
from ducktape_harness.render import context_window_for, format_footer
from ducktape_harness.session import Session
from tests.fake_adapter import FakeAdapter, text_events


class TestUsage(unittest.TestCase):
    def test_totals_accumulation(self):
        adapter = FakeAdapter(
            scripts=[
                text_events(
                    "a", usage={"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2}
                ),
                text_events("b", usage={"input_tokens": 3, "output_tokens": 4}),
            ]
        )
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)

        session = Session(model="fake-model", pin="fake", provider=provider)
        session.gate = PermissionGate()
        with patch("builtins.print"):
            run_turn(session, "hi")
            run_turn(session, "hi2")
        self.assertEqual(session.totals.input_tokens, 13)
        self.assertEqual(session.totals.output_tokens, 9)
        self.assertEqual(session.totals.cache_read_tokens, 2)
        self.assertEqual(session.totals.turns, 2)
        self.assertEqual(session.totals.turns_with_usage, 2)

    def test_last_usage_snapshot(self):
        adapter = FakeAdapter(
            scripts=[
                text_events("a", usage={"input_tokens": 10, "output_tokens": 5}),
                text_events("b", usage={"input_tokens": 1, "output_tokens": 1}),
            ]
        )
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)

        session = Session(model="fake-model", pin="fake", provider=provider)
        session.gate = PermissionGate()
        with patch("builtins.print"):
            run_turn(session, "hi")
            self.assertIsNotNone(session.last_usage)
            first = dict(session.last_usage or {})
            run_turn(session, "hi2")
        self.assertNotEqual(first, session.last_usage)
        assert session.last_usage is not None
        self.assertEqual(session.last_usage["input_tokens"], 1)

    def test_footer_segments(self):
        # None usage -> ctx —
        f = format_footer(10, 5, None, "unknown-model-xyz", None)
        self.assertIn("↑10", f)
        self.assertIn("↓5", f)
        self.assertIn("ctx —", f)
        self.assertNotIn("tok /", f)
        # latency None -> no ms
        self.assertNotIn("ms", f)
        # with usage but no window -> ctx tok without / win
        f2 = format_footer(
            10, 5, {"input_tokens": 100, "output_tokens": 50}, "unknown-model-xyz", 123
        )
        self.assertIn("ctx 150tok", f2)
        self.assertNotIn("/", f2.split("ctx")[1].split("·")[0] if "·" in f2 else f2)
        self.assertIn("123ms", f2)
        # with window
        with patch("ducktape_harness.render.context_window_for", return_value=200):
            f3 = format_footer(
                0, 0, {"input_tokens": 100, "output_tokens": 0}, "claude-sonnet-4", 10
            )
            self.assertIn("/ 200", f3)
            self.assertIn("50.0%", f3)

    def test_context_window_longest_prefix(self):
        windows = {"gpt-4": 8192, "gpt-4o": 128000, "claude": 200000}
        self.assertEqual(context_window_for("gpt-4o-mini", windows), 128000)
        self.assertEqual(context_window_for("gpt-4", windows), 8192)
        self.assertEqual(context_window_for("claude-sonnet-4", windows), 200000)
        self.assertIsNone(context_window_for("unknown", windows))

    def test_usage_reported_ratio(self):
        from ducktape_harness.commands import cmd_usage

        adapter = FakeAdapter(
            scripts=[
                text_events("a"),
                text_events("b", usage={"input_tokens": 1, "output_tokens": 1}),
            ]
        )
        provider = Provider(adapters={"fake": adapter}, autodiscover=False)

        session = Session(model="fake-model", pin="fake", provider=provider)
        session.gate = PermissionGate()
        with patch("builtins.print"):
            run_turn(session, "hi")
            run_turn(session, "hi2")
        out = cmd_usage(session)
        self.assertIn("usage reported in 1 of 2 turns", out)


class TestModelInfoWindow(unittest.TestCase):
    def setUp(self):
        from ducktape_harness.render import clear_window_cache

        clear_window_cache()

    tearDown = setUp

    def test_provider_info_preferred_then_table_fallback(self):
        from ducktape_harness.render import context_window_for
        from tests.fake_adapter import FakeAdapter

        info = {"m-x": {"context_window": 777, "max_output_tokens": 64}}
        provider = FakeAdapter(infos=info)
        self.assertEqual(context_window_for("m-x", provider=provider), 777)
        # no info for model -> longest-prefix table
        self.assertEqual(context_window_for("claude-sonnet-4", provider=provider), 200000)

        # provider raising -> table
        class Bad:
            def model_info(self, m, **k):  # type: ignore[no-untyped-def]
                raise RuntimeError("down")

        self.assertEqual(context_window_for("gpt-4o-mini", provider=Bad()), 128000)

    def test_footer_uses_provider_window(self):
        from ducktape_harness.render import format_footer
        from tests.fake_adapter import FakeAdapter

        provider = FakeAdapter(
            infos={"qwen3:8b": {"context_window": 32768, "max_output_tokens": None}}
        )
        foot = format_footer(
            100,
            20,
            {"input_tokens": 500, "output_tokens": 50},
            "qwen3:8b",
            None,
            provider=provider,
            pin=None,
        )
        self.assertIn("32768", foot)
        self.assertIn("1.7%", foot)  # 550/32768
