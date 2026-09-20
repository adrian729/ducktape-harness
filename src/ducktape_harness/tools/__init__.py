"""Tools package — ensures registry is populated."""

from ducktape_harness.tools import files as _files  # noqa: F401
from ducktape_harness.tools import shell as _shell  # noqa: F401
from ducktape_harness.tools.base import REGISTRY

__all__ = ["REGISTRY"]
