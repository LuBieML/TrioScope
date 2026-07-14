"""
Trio Controller SCOPE capture module.
Provides high-speed, deterministic parameter capture using the controller's built-in SCOPE command,
plus drive-based scope capture via CoE SDO for internal servo drive variables.
"""

from .scope_engine import ScopeEngine, ScopeParameterParser
from .drive_scope_engine import DriveScopeEngine, COMMON_DRIVE_VARIABLES, TRIGGER_MODES
from .axis_parameter_writer import AXIS_PARAMETER_SETTERS, write_axis_parameters

__all__ = [
    'ScopeEngine', 'ScopeParameterParser',
    'DriveScopeEngine', 'COMMON_DRIVE_VARIABLES', 'TRIGGER_MODES',
    'AXIS_PARAMETER_SETTERS', 'write_axis_parameters',
]
