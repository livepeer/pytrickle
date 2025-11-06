"""
Warmup configuration for PyTrickle frame processors.

This module provides configuration for warmup/loading behavior in frame processors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class WarmupMode(Enum):
    """Mode for handling video frames during warmup/loading."""
    OVERLAY = "overlay"      # Show loading animation overlay
    PASSTHROUGH = "passthrough"  # Pass through original frames


@dataclass
class WarmupConfig:
    """Configuration for warmup/loading behavior."""
    mode: WarmupMode = WarmupMode.OVERLAY  # How to handle video frames during warmup
    message: str = "Loading..."             # Loading message to display in overlay
    progress: Optional[float] = None        # Progress value 0.0-1.0 (None = animated)
    enabled: bool = True                    # Whether warmup gating is active

