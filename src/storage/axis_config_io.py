"""JSON import/export for TrioScope axis parameter configurations."""

import json
from pathlib import Path
from typing import Iterable, List

if __package__ and __package__.startswith("src."):
    from ..models.axis_parameter_config import AxisParameterConfig
else:  # App runtime and PyInstaller import storage as a top-level package.
    from models.axis_parameter_config import AxisParameterConfig


AXIS_CONFIG_FORMAT = "TrioScope axis parameters"
AXIS_CONFIG_VERSION = 1


def save_axis_config(path: str, configs: Iterable[AxisParameterConfig]) -> None:
    items = list(configs)
    _validate_unique_axes(items)
    payload = {
        "format": AXIS_CONFIG_FORMAT,
        "version": AXIS_CONFIG_VERSION,
        "axis_parameters": [config.to_dict() for config in items],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def load_axis_config(path: str) -> List[AxisParameterConfig]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("axis_parameters", payload.get("axes"))
    else:
        raw_items = None

    if not isinstance(raw_items, list):
        raise ValueError("The JSON file must contain an 'axis_parameters' list.")

    configs = [AxisParameterConfig.from_dict(item) for item in raw_items]
    _validate_unique_axes(configs)
    return configs


def _validate_unique_axes(configs: Iterable[AxisParameterConfig]) -> None:
    seen = set()
    for config in configs:
        config.validate()
        if config.axis in seen:
            raise ValueError(f"Axis {config.axis} is configured more than once.")
        seen.add(config.axis)
