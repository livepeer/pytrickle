"""
Monotonic Audio Timeline Tracker for A/V Sync

Automatically corrects audio frame timestamps to maintain a monotonic timeline,
preventing encoder sync issues without blocking video frame processing.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from .frames import AudioFrame

logger = logging.getLogger(__name__)

@dataclass
class AudioTrackingConfig:
    """Configuration for audio timeline tracking."""
    frame_duration_ms: int = 20
    drift_threshold_ms: int = 50  # Correct drift > 50ms
    max_drift_ms: int = 200  # Reset timeline if drift > 200ms


class MonotonicAudioTracker:
    """
    Lightweight audio timeline tracker that ensures monotonic audio timestamps.
    
    This prevents audio/video sync issues in the encoder by maintaining a consistent
    audio timeline, even when frame skipping occurs or timestamps are irregular.
    """
    
    def __init__(self, frame_duration_ms: int = 20, config: Optional[AudioTrackingConfig] = None):
        """
        Initialize monotonic audio tracker.
        
        Args:
            frame_duration_ms: Expected duration between audio frames in milliseconds
            config: Optional configuration object (overrides individual params)
        """
        # Use config if provided, otherwise create from individual params
        if config is not None:
            self.config = config
        else:
            self.config = AudioTrackingConfig(frame_duration_ms=frame_duration_ms)
        
        self.last_audio_timestamp: Optional[int] = None
        self.expected_next_timestamp: Optional[int] = None
        
    def process_audio_frame(self, audio_frame: AudioFrame) -> AudioFrame:
        """
        Process audio frame with automatic timestamp correction for monotonic timeline.
        
        Args:
            audio_frame: Input audio frame that may have irregular timestamps
            
        Returns:
            AudioFrame with corrected monotonic timestamp
        """
        current_timestamp = audio_frame.timestamp
        
        # Initialize timeline on first audio frame
        if self.last_audio_timestamp is None:
            self.last_audio_timestamp = current_timestamp
            self.expected_next_timestamp = current_timestamp + self.config.frame_duration_ms
            logger.info(f"Audio timeline initialized at timestamp {current_timestamp}")
            return audio_frame
        
        # Calculate drift from expected timeline
        expected_ts = self.expected_next_timestamp
        drift_ms = abs(current_timestamp - expected_ts)
        
        # Determine if correction is needed
        needs_correction = False
        corrected_timestamp = current_timestamp
        
        if current_timestamp <= self.last_audio_timestamp:
            # Timestamp went backwards or stayed same - always correct
            needs_correction = True
            corrected_timestamp = expected_ts
            logger.debug(f"Audio timestamp went backwards: {current_timestamp} <= {self.last_audio_timestamp}")
            
        elif drift_ms > self.config.max_drift_ms:
            # Large drift - reset timeline to current timestamp
            corrected_timestamp = current_timestamp
            
        elif drift_ms > self.config.drift_threshold_ms:
            # Moderate drift - use expected timestamp for smooth timeline
            needs_correction = True
            corrected_timestamp = expected_ts
            
        # Apply correction if needed
        if needs_correction:
            # Adjust timestamp without creating a new frame
            audio_frame.timestamp = corrected_timestamp
            corrected_frame = audio_frame
            logger.debug(f"Audio timestamp corrected: {current_timestamp} -> {corrected_timestamp} (drift: {drift_ms}ms)")
        else:
            corrected_frame = audio_frame
        
        # Update timeline state
        self.last_audio_timestamp = corrected_timestamp
        self.expected_next_timestamp = corrected_timestamp + self.config.frame_duration_ms
        
        return corrected_frame
    
    def reset(self):
        """Reset the audio timeline tracker completely."""
        self.last_audio_timestamp = None
        self.expected_next_timestamp = None
        logger.info("Audio timeline tracker reset")
