"""
Preview video configuration for PyTrickle frame processors.

This module provides configuration for preview view behavior in frame processors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PreviewVideoMode(Enum):
    """Mode for handling video frames during preview/loading."""
    ProgressBar = "ProgressBar"      # Show ProgressBar animation overlay
    Passthrough = "passthrough"  # Pass through original frames


@dataclass
class PreviewVideoConfig:
    """Configuration for preview view behavior."""
    mode: PreviewVideoMode = PreviewVideoMode.ProgressBar  # How to handle video frames during preview
    message: str = "Loading..."              # Message to display in preview view
    progress: Optional[float] = None         # Progress value 0.0-1.0 (None = animated)
    enabled: bool = True                     # Whether preview gating is active
    auto_timeout_seconds: Optional[float] = 1.5  # Seconds without output before preview view auto-enables
    
    def is_ProgressBar_mode(self) -> bool:
        """Return True if this config is set to ProgressBar mode."""
        return self.mode == PreviewVideoMode.ProgressBar

    def is_passthrough_mode(self) -> bool:
        """Return True if this config is set to passthrough mode."""
        return self.mode == PreviewVideoMode.Passthrough
