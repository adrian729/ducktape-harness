"""Tool tests — in tmpdirs."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from ducktape_harness.config import OUTPUT_CAP, READ_MAX
from ducktape_harness.tools.base import ToolError
from ducktape_harness.tools.files import ReadFileTool, WriteFileTool
from ducktape_harness.tools.shell import BashTool


class TestReadFile(unittest.TestCase):
    def test_read_offset_limit(self):
        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("line1\nline2\nline3\n")
            out = tool.run({"path": p, "offset": 1, "limit": 1}, cwd=td)
            self.assertIn("line2", out)
            self.assertNotIn("line1", out)
            self.assertNotIn("line3", out)

    def test_read_size_cap(self):
        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "big")
            with open(p, "wb") as f:
                f.write(b"x" * (READ_MAX + 1))
            with self.assertRaises(ToolError) as ctx:
                tool.run({"path": p}, cwd=td)
            self.assertIn("offset/limit", str(ctx.exception))

    def test_read_relative_cwd(self):
        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            sub = os.path.join(td, "sub")
            os.makedirs(sub)
            p = os.path.join(sub, "a.txt")
            with open(p, "w") as f:
                f.write("hello")
            out = tool.run({"path": "a.txt"}, cwd=sub)
            self.assertIn("hello", out)

    def test_read_is_error_prefix_not_flagged(self):
        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.txt")
            with open(p, "w") as f:
                f.write("is_error: this is legitimate content\nsecond line")
            out = tool.run({"path": p}, cwd=td)
            self.assertIn("is_error:", out)
            # Should not raise — legitimate content is returned as plain string
            self.assertTrue(out.startswith("is_error:"))

    def test_read_bad_args_raises(self):
        tool = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ToolError):
                tool.run({"path": 123}, cwd=td)  # type: ignore[dict-item]
            with self.assertRaises(ToolError):
                tool.run({"path": "nope", "offset": "bad"}, cwd=td)  # type: ignore[dict-item]


class TestWriteFile(unittest.TestCase):
    def test_write_and_read(self):
        w = WriteFileTool()
        r = ReadFileTool()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "out.txt")
            msg = w.run({"path": p, "content": "world"}, cwd=td)
            self.assertIn("Wrote", msg)
            out = r.run({"path": p}, cwd=td)
            self.assertEqual(out, "world")

    def test_write_relative(self):
        w = WriteFileTool()
        with tempfile.TemporaryDirectory() as td:
            w.run({"path": "a/b.txt", "content": "hi"}, cwd=td)
            self.assertTrue(os.path.exists(os.path.join(td, "a/b.txt")))

    def test_write_bad_args_raises(self):
        w = WriteFileTool()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ToolError):
                w.run({"path": 123, "content": "hi"}, cwd=td)  # type: ignore[dict-item]


class TestBash(unittest.TestCase):
    def test_bash_echo(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            out = tool.run({"command": "echo hello"}, cwd=td)
            self.assertIn("hello", out)

    def test_bash_timeout(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ToolError) as ctx:
                tool.run({"command": "sleep 2", "timeout": 0.1}, cwd=td)
            self.assertIn("timed out", str(ctx.exception))

    def test_bash_timeout_partial_output(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            # command that prints partial then sleeps
            with self.assertRaises(ToolError) as ctx:
                tool.run({"command": "echo partial; sleep 2", "timeout": 0.1}, cwd=td)
            self.assertIn("timed out", str(ctx.exception))
            self.assertIn("partial", str(ctx.exception))

    def test_bash_truncation(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            # generate >100KB
            out = tool.run({"command": f"python3 -c \"print('x'*{OUTPUT_CAP + 1000})\""}, cwd=td)
            self.assertIn("[truncated]", out)
            # ensure not way over cap
            self.assertLess(len(out), OUTPUT_CAP + 5000)

    def test_bash_cwd(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            out = tool.run({"command": "pwd"}, cwd=td)
            self.assertIn(td, out)

    def test_bash_nonzero_with_output(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            out = tool.run({"command": "echo hi; exit 2"}, cwd=td)
            self.assertIn("hi", out)
            self.assertIn("[exit 2]", out)
            # plain output, not is_error — ToolError not raised
            self.assertNotIn("is_error", out)

    def test_bash_nonzero_no_output(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            out = tool.run({"command": "exit 3"}, cwd=td)
            self.assertIn("[exit 3]", out)
            self.assertNotIn("is_error", out)

    def test_bash_tool_timeout_config(self):
        import ducktape_harness.config as cfg
        from ducktape_harness.tools.shell import BashTool as BT

        orig = cfg.TOOL_TIMEOUT
        try:
            cfg.TOOL_TIMEOUT = 0.05
            tool = BT()
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(ToolError) as ctx:
                    tool.run({"command": "sleep 1"}, cwd=td)
                self.assertIn("timed out after 0.05", str(ctx.exception))
                # per-call override still works
                out = tool.run({"command": "echo ok", "timeout": 5}, cwd=td)
                self.assertIn("ok", out)
        finally:
            cfg.TOOL_TIMEOUT = orig

    def test_bash_bad_args_raises(self):
        tool = BashTool()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ToolError):
                tool.run({"command": 123}, cwd=td)  # type: ignore[dict-item]


class TestTimeoutCeiling(unittest.TestCase):
    def test_timeout_override_cannot_exceed_ceiling(self):
        from unittest.mock import patch as p_

        import ducktape_harness.tools.shell as sh

        captured = {}

        def fake_run(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        with p_("subprocess.run", side_effect=fake_run):
            sh.BashTool().run({"command": "x", "timeout": 9999}, "/tmp")
            self.assertEqual(captured["timeout"], sh.config.TOOL_TIMEOUT)
            sh.BashTool().run({"command": "x", "timeout": 5}, "/tmp")
            self.assertEqual(captured["timeout"], 5)
            sh.BashTool().run({"command": "x", "timeout": "junk"}, "/tmp")
            self.assertEqual(captured["timeout"], sh.config.TOOL_TIMEOUT)


class TestReadImage(unittest.TestCase):
    def test_read_image_returns_block(self):
        import base64
        import pathlib

        from ducktape_harness.tools.files import ReadFileTool

        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "pic.png"
            p.write_bytes(b"\x89PNGdata")
            out = ReadFileTool().run({"path": str(p), "image": True}, td)
            self.assertIsInstance(out, list)
            blk = out[0]
            self.assertEqual(blk["type"], "image")
            self.assertEqual(blk["media_type"], "image/png")
            self.assertEqual(base64.b64decode(blk["data"]), b"\x89PNGdata")

    def test_read_image_rejects(self):
        import pathlib

        from ducktape_harness.tools.base import ToolError
        from ducktape_harness.tools.files import ReadFileTool

        with tempfile.TemporaryDirectory() as td:
            txt = pathlib.Path(td) / "a.txt"
            txt.write_text("x")
            with self.assertRaises(ToolError):
                ReadFileTool().run({"path": str(txt), "image": True}, td)
            png = pathlib.Path(td) / "b.png"
            png.write_bytes(b"")
            with self.assertRaises(ToolError):
                ReadFileTool().run({"path": str(png), "image": True}, td)
