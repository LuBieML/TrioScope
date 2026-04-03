"""
Trio Controller SCOPE capture module.
Provides high-speed, deterministic parameter capture using the controller's built-in SCOPE command.
"""

from .scope_engine import ScopeEngine, ScopeParameterParser

__all__ = ['ScopeEngine', 'ScopeParameterParser']
