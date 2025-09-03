"""
Simple Monotonic Audio Synchronizer

Handles audio timestamp synchronization after frame skipping to maintain
monotonic progression and prevent DTS errors.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from fractions import Fraction
from .frames import AudioFrame

logger = logging.getLogger(__name__)

@dataclass
class AudioSyncConfig:
    """Configuration for audio synchronization."""
    frame_duration_ms: int = 20  # Expected audio frame duration (20ms = 50fps)

class MonotonicAudioSynchronizer:
    """
    Simple monotonic audio synchronizer that ensures audio timestamps
    always increase monotonically to prevent encoder DTS errors.
    """
    
    def __init__(self, config: Optional[AudioSyncConfig] = None):
        """
        Initialize audio synchronizer.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or AudioSyncConfig()
        
        self.last_audio_timestamp: Optional[int] = None
        self.expected_interval: Optional[int] = None
        self.frame_count = 0
    
    def synchronize_audio_frame(self, frame: AudioFrame) -> AudioFrame:
        """
        Ensure audio frame timestamp is strictly monotonic.
        
        Args:
            frame: Input audio frame
            
        Returns:
            AudioFrame with monotonic timestamp
        """
        if self.last_audio_timestamp is None:
            # Initialize with current timestamp
            self.last_audio_timestamp = frame.timestamp
            self.expected_interval = self._calculate_interval(frame.time_base)
            self.frame_count = 1
            logger.debug(f"Audio timeline initialized at {frame.timestamp}")
            return frame
        
        # Calculate next monotonic timestamp
        next_timestamp = self.last_audio_timestamp + self.expected_interval
        
        # Always use monotonic progression - ignore original timestamp
        frame.timestamp = next_timestamp
        logger.debug(f"Audio timestamp set to monotonic: {next_timestamp}")
        
        # Update state
        self.last_audio_timestamp = next_timestamp
        self.frame_count += 1
        
        return frame
    
    def _calculate_interval(self, time_base: Fraction) -> int:
        """Calculate expected audio frame interval in timestamp units."""
        frame_duration_seconds = self.config.frame_duration_ms / 1000.0
        return int(frame_duration_seconds * time_base.denominator / time_base.numerator)
    
    def reset(self):
        """Reset timeline state."""
        self.last_audio_timestamp = None
        self.expected_interval = None
        self.frame_count = 0
        logger.info("Audio synchronizer reset")
    
    def get_stats(self) -> dict:
        """Get synchronization statistics."""
        return {
            "audio_frames": self.frame_count,
            "last_timestamp": self.last_audio_timestamp,
            "interval": self.expected_interval
        }
