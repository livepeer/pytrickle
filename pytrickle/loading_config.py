"""
Loading configuration for PyTrickle frame processors.

This module provides configuration for loading behavior in frame processors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LoadingMode(Enum):
    """Mode for handling video frames during loading."""
    OVERLAY = "overlay"      # Show loading animation overlay
    PASSTHROUGH = "passthrough"  # Pass through original frames


@dataclass
class LoadingConfig:
    """Configuration for loading behavior."""
    mode: LoadingMode = LoadingMode.OVERLAY  # How to handle video frames during loading
    message: str = "Loading..."              # Loading message to display in overlay
    progress: Optional[float] = None         # Progress value 0.0-1.0 (None = animated)
    enabled: bool = True                     # Whether loading gating is active

