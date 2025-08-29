"""
Monotonic Audio Timeline Tracker for A/V Sync

Automatically corrects audio frame timestamps to maintain a monotonic timeline,
preventing encoder sync issues without blocking video frame processing.
"""

import logging
from typing import Optional
from .frames import AudioFrame

logger = logging.getLogger(__name__)


class MonotonicAudioTracker:
    """
    Lightweight audio timeline tracker that ensures monotonic audio timestamps.
    
    This prevents audio/video sync issues in the encoder by maintaining a consistent
    audio timeline, even when frame skipping occurs or timestamps are irregular.
    """
    
    def __init__(self, frame_duration_ms: int = 20):
        """
        Initialize monotonic audio tracker.
        
        Args:
            frame_duration_ms: Expected duration between audio frames in milliseconds
        """
        self.frame_duration_ms = frame_duration_ms
        self.last_audio_timestamp: Optional[int] = None
        self.expected_next_timestamp: Optional[int] = None
        self.drift_correction_count = 0
        self.total_audio_frames = 0
        
        # Drift detection thresholds
        self.drift_threshold_ms = 50  # Correct drift > 50ms
        self.max_drift_ms = 200  # Reset timeline if drift > 200ms
        
    def process_audio_frame(self, audio_frame: AudioFrame) -> AudioFrame:
        """
        Process audio frame with automatic timestamp correction for monotonic timeline.
        
        Args:
            audio_frame: Input audio frame that may have irregular timestamps
            
        Returns:
            AudioFrame with corrected monotonic timestamp
        """
        current_timestamp = audio_frame.timestamp
        self.total_audio_frames += 1
        
        # Initialize timeline on first audio frame
        if self.last_audio_timestamp is None:
            self.last_audio_timestamp = current_timestamp
            self.expected_next_timestamp = current_timestamp + self.frame_duration_ms
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
            
        elif drift_ms > self.max_drift_ms:
            # Large drift - reset timeline to current timestamp
            corrected_timestamp = current_timestamp
            
        elif drift_ms > self.drift_threshold_ms:
            # Moderate drift - use expected timestamp for smooth timeline
            needs_correction = True
            corrected_timestamp = expected_ts
            
        # Apply correction if needed
        if needs_correction:
            self.drift_correction_count += 1
            
            # Log corrections periodically
            if self.drift_correction_count % 50 == 0:
                logger.info(f"Audio timeline corrections: {self.drift_correction_count}/{self.total_audio_frames} "
                           f"({100.0 * self.drift_correction_count / self.total_audio_frames:.1f}%)")
            
            # Adjust timestamp without creating a new frame
            audio_frame.timestamp = corrected_timestamp
            corrected_frame = audio_frame
            logger.debug(f"Audio timestamp corrected: {current_timestamp} -> {corrected_timestamp} (drift: {drift_ms}ms)")
        else:
            corrected_frame = audio_frame
        
        # Update timeline state
        self.last_audio_timestamp = corrected_timestamp
        self.expected_next_timestamp = corrected_timestamp + self.frame_duration_ms
        
        return corrected_frame
    
    def reset(self):
        """Reset the audio timeline tracker."""
        self.last_audio_timestamp = None
        self.expected_next_timestamp = None
        self.drift_correction_count = 0
        self.total_audio_frames = 0
        logger.info("Audio timeline tracker reset")
