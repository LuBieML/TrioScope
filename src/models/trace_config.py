from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class TraceConfig:
    """Configuration for a single trace channel."""
    param: str = "MPOS"
    axis: int = 0
    enabled: bool = True
    fft: bool = False
    drive_mode: bool = False
    drive_var_address: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceConfig":
        """Reconstruct trace configuration from a dictionary."""
        return cls(
            param=str(data.get("param", "MPOS") or "MPOS").strip() or "MPOS",
            axis=int(data.get("axis", 0)),
            enabled=data.get("enabled", True) in (True, "true", "True", 1),
            fft=data.get("fft", False) in (True, "true", "True", 1),
            drive_mode=data.get("drive_mode", False) in (True, "true", "True", 1),
            drive_var_address=data.get("drive_var_address")
        )
