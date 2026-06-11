"""Shared pytest fixtures and import stubs for the TrioScope test suite.

The proprietary ``Trio_UnifiedApi`` package is only available on machines
with the Trio SDK installed (and only ships Windows binaries).  Several
modules under ``src/`` import it at module level, so on CI and other
machines without the SDK we inject a stand-in module before any test
imports run.  Tests that exercise controller behaviour already drive the
code through fakes/mocks, so the stub never needs real functionality.
"""

import sys
from unittest.mock import MagicMock

try:
    import Trio_UnifiedApi  # noqa: F401
except ImportError:
    sys.modules["Trio_UnifiedApi"] = MagicMock(name="Trio_UnifiedApi_stub")
