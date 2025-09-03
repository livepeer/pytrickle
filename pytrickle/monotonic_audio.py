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
    pass  # No configuration needed - frame duration calculated dynamically

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
        Ensure audio frame timestamp is strictly monotonic while preserving natural timing.
        
        Args:
            frame: Input audio frame
            
        Returns:
            AudioFrame with monotonic timestamp
        """
        if self.last_audio_timestamp is None:
            # Initialize with current timestamp
            self.last_audio_timestamp = frame.timestamp
            self.expected_interval = self._calculate_interval(frame)
            self.frame_count = 1
            return frame
        
        # Use original timestamp if it's already monotonic, otherwise correct it
        if frame.timestamp > self.last_audio_timestamp:
            # Original timestamp is already monotonic, use it
            corrected_timestamp = frame.timestamp
            # Update expected interval based on actual gap
            self.expected_interval = self._calculate_interval(frame)
        else:
            # Original timestamp would break monotonicity, use calculated next timestamp
            corrected_timestamp = self.last_audio_timestamp + self.expected_interval
        
        # Update timestamp for monotonic progression
        frame.timestamp = corrected_timestamp
        
        # Update state
        self.last_audio_timestamp = corrected_timestamp
        self.frame_count += 1
        
        return frame
    
    def _calculate_interval(self, frame: AudioFrame) -> int:
        """Calculate actual audio frame interval based on frame samples and sample rate."""
        frame_duration_seconds = frame.nb_samples / frame.rate
        return int(frame_duration_seconds * frame.time_base.denominator / frame.time_base.numerator)
    
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
