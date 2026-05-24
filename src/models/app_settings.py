from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from .trace_config import TraceConfig

@dataclass
class ConnectionSettings:
    ip: str = "192.168.0.245"

@dataclass
class CaptureSettings:
    sample_period: str = "1"
    duration: str = "5.0"
    table_start: str = "0"
    use_end_of_table: bool = True
    capture_mode: str = "continuous"  # "single" or "continuous"
    external_trigger: bool = False

@dataclass
class PlotSettings:
    line_width: float = 1.0
    grid_alpha: float = 0.3
    bg_color: str = "#0A0A0A"

@dataclass
class DisplaySettings:
    plot_mode: str = "time"  # "time", "xy", "xyz"
    window_duration: float = 5.0
    lock_x_axis: bool = True

@dataclass
class AppSettings:
    connection: ConnectionSettings = field(default_factory=ConnectionSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    plot: PlotSettings = field(default_factory=PlotSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    traces: List[TraceConfig] = field(default_factory=list)
    drive_profiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to a dictionary."""
        return {
            "connection": asdict(self.connection),
            "capture": asdict(self.capture),
            "plot": asdict(self.plot),
            "display": asdict(self.display),
            "traces": [t.to_dict() for t in self.traces],
            "drive_profiles": self.drive_profiles
        }
