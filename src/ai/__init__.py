"""
AI-powered scope analysis module.
Provides NanoGPT integration for intelligent interpretation of captured scope data.
"""

from .nanogpt_client import NanoGPTClient
from .signal_metrics import SignalMetrics
from .analysis_panel import AIAnalysisPanel

__all__ = ['NanoGPTClient', 'SignalMetrics', 'AIAnalysisPanel']
