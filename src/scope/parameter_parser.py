"""
Scope parameter string parsing.

Converts user-friendly parameter strings into SCOPE-compatible format.
"""

import re
import logging
from typing import List, Tuple

from .parameters import CHANNEL_PARAMETERS_SET as CHANNEL_PARAMETERS, AXIS_PARAMETERS

AXIS_PARAMETERS = set(AXIS_PARAMETERS)

logger = logging.getLogger(__name__)


class ScopeParameterParser:
    """
    Parses user-friendly parameter strings into SCOPE-compatible format.

    Handles various input formats:
    - MPOS(0) or ?mpos(0) → "MPOS AXIS(0)"
    - MPOS or ?mpos → "MPOS AXIS(0)" (default axis 0)
    - VR(5) → "VR(5)"
    - TABLE(100) → "TABLE(100)"
    - Multiple params: "MPOS(0), DPOS(0), FE(0)" → ["MPOS AXIS(0)", "DPOS AXIS(0)", "FE AXIS(0)"]
    """

    @staticmethod
    def parse_parameter_string(param_str: str) -> Tuple[str, str]:
        """
        Parse a single parameter string into SCOPE format.

        Args:
            param_str: User input like "MPOS(0)" or "VR(5)"

        Returns:
            Tuple of (scope_param_string, display_name)
            Example: ("MPOS AXIS(0)", "MPOS(0)")

        Raises:
            ValueError: If parameter format is invalid
        """
        param_str = param_str.strip()
        if not param_str:
            raise ValueError("Parameter cannot be empty")

        # Remove leading '?' if present
        if param_str.startswith('?'):
            param_str = param_str[1:]

        # Pattern 1: VR(index)
        vr_match = re.match(r'^VR\s*\(\s*(\d+)\s*\)$', param_str, re.IGNORECASE)
        if vr_match:
            index = vr_match.group(1)
            return f"VR({index})", f"VR({index})"

        # Pattern 2: TABLE(index)
        table_match = re.match(r'^TABLE\s*\(\s*(\d+)\s*\)$', param_str, re.IGNORECASE)
        if table_match:
            index = table_match.group(1)
            return f"TABLE({index})", f"TABLE({index})"

        # Pattern 3: PARAM(index) - axis or channel parameter with explicit index
        indexed_param_match = re.match(r'^(\w+)\s*\(\s*(\d+)\s*\)$', param_str, re.IGNORECASE)
        if indexed_param_match:
            param_name = indexed_param_match.group(1).upper()
            index_num = indexed_param_match.group(2)

            if param_name in CHANNEL_PARAMETERS:
                return f"{param_name}({index_num})", f"{param_name} Ch({index_num})"
            elif param_name in AXIS_PARAMETERS:
                return f"{param_name} AXIS({index_num})", f"{param_name}({index_num})"
            else:
                # Unknown parameter - might be valid on controller
                logger.warning(f"Unknown parameter: {param_name}")
                return f"{param_name} AXIS({index_num})", f"{param_name}({index_num})"

        # Pattern 4: PARAM - axis/channel parameter without explicit index (default to 0)
        param_only_match = re.match(r'^(\w+)$', param_str, re.IGNORECASE)
        if param_only_match:
            param_name = param_only_match.group(1).upper()

            if param_name in CHANNEL_PARAMETERS:
                return f"{param_name}(0)", f"{param_name} Ch(0)"
            elif param_name in AXIS_PARAMETERS:
                return f"{param_name} AXIS(0)", f"{param_name}(0)"
            else:
                # Might be a system parameter (no axis needed)
                return param_name, param_name

        raise ValueError(f"Invalid parameter format: {param_str}")

    @staticmethod
    def parse_multiple_parameters(params_str: str) -> Tuple[List[str], List[str]]:
        """
        Parse comma-separated parameter list.

        Args:
            params_str: Comma-separated parameters like "MPOS(0), DPOS(0), FE(0)"

        Returns:
            Tuple of (scope_params_list, display_names_list)

        Raises:
            ValueError: If any parameter is invalid
        """
        param_strs = [p.strip() for p in params_str.split(',') if p.strip()]

        scope_params = []
        display_names = []

        for param_str in param_strs:
            scope_param, display_name = ScopeParameterParser.parse_parameter_string(param_str)
            scope_params.append(scope_param)
            display_names.append(display_name)

        return scope_params, display_names
