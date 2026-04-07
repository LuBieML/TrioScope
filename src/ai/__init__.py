"""
AI-powered scope analysis module.
Provides NanoGPT integration for intelligent interpretation of captured scope data.
"""

from .nanogpt_client import NanoGPTClient
from .signal_metrics import SignalMetrics
from .analysis_panel import AIAnalysisPanel
from .coe_io import (
    coe_read_axis,
    coe_read_slot,
    coe_write_axis,
    read_drive_profile,
    write_drive_profile,
    read_single_pn,
    write_single_pn,
)
from .ethercat_scan import scan_network, EthercatNetwork, EthercatSlot, EthercatSlave
from .ethercat_map_window import EthercatMapWindow

__all__ = [
    'NanoGPTClient', 'SignalMetrics', 'AIAnalysisPanel',
    'coe_read_axis', 'coe_read_slot', 'coe_write_axis',
    'read_drive_profile', 'write_drive_profile',
    'read_single_pn', 'write_single_pn',
    'scan_network', 'EthercatNetwork', 'EthercatSlot', 'EthercatSlave',
    'EthercatMapWindow',
]
