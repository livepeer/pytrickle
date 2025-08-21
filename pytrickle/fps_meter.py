"""
FPS Meter - Production-ready frame rate measurement.

Provides real-time FPS tracking with rolling windows and overflow protection.
"""

import time
from collections import deque
from typing import Dict


class FPSMeter:
    """High-performance FPS meter with rolling time windows and overflow protection."""
    
    def __init__(self, window_seconds: float = 10.0, max_samples: int = 600):
        """Initialize FPS meter.
        
        Args:
            window_seconds: Time window for FPS calculation (default: 10s)
            max_samples: Maximum samples to keep (default: 600 = ~60fps * 10s)
        """
        self.window_seconds = window_seconds
        # Ingress tracking (frames coming in)
        self.ingress_video_timestamps = deque(maxlen=max_samples)
        self.ingress_audio_timestamps = deque(maxlen=max_samples)
        # Egress tracking (frames going out)
        self.egress_video_timestamps = deque(maxlen=max_samples)
        self.egress_audio_timestamps = deque(maxlen=max_samples)
        
        # Cache for expensive calculations
        self._last_calculation_time = 0
        self._cached_ingress_fps = 0.0
        self._cached_egress_fps = 0.0
        self._cache_duration = 0.5
        
    def record_ingress_video_frame(self):
        """Record an ingress video frame timestamp."""
        self.ingress_video_timestamps.append(time.time())
    
    def record_ingress_audio_frame(self):
        """Record an ingress audio frame timestamp."""
        self.ingress_audio_timestamps.append(time.time())
        
    def record_egress_video_frame(self):
        """Record an egress video frame timestamp."""
        self.egress_video_timestamps.append(time.time())
    
    def record_egress_audio_frame(self):
        """Record an egress audio frame timestamp."""
        self.egress_audio_timestamps.append(time.time())
    
    def get_ingress_video_fps(self) -> float:
        """Get cached ingress FPS to avoid expensive recalculation."""
        now = time.time()
        if now - self._last_calculation_time > self._cache_duration:
            self._cached_ingress_fps = self._calculate_fps(self.ingress_video_timestamps)
            self._cached_egress_fps = self._calculate_fps(self.egress_video_timestamps)
            self._last_calculation_time = now
        return self._cached_ingress_fps
    
    def get_ingress_audio_fps(self) -> float:
        """Get current ingress audio FPS over the time window."""
        return self._calculate_fps(self.ingress_audio_timestamps)
        
    def get_egress_video_fps(self) -> float:
        """Get cached egress FPS to avoid expensive recalculation."""
        now = time.time()
        if now - self._last_calculation_time > self._cache_duration:
            self._cached_ingress_fps = self._calculate_fps(self.ingress_video_timestamps)
            self._cached_egress_fps = self._calculate_fps(self.egress_video_timestamps)
            self._last_calculation_time = now
        return self._cached_egress_fps
    
    def get_egress_audio_fps(self) -> float:
        """Get current egress audio FPS over the time window."""
        return self._calculate_fps(self.egress_audio_timestamps)
    
    def _calculate_fps(self, timestamps: deque) -> float:
        """Calculate FPS from timestamps within the time window - OPTIMIZED."""
        if len(timestamps) < 2:
            return 0.0
        
        # Optimize: Use deque iteration instead of list comprehension
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Count recent timestamps without creating new list
        recent_count = 0
        first_recent = None
        last_recent = None
        
        # Iterate from newest to oldest (deque is ordered)
        for ts in reversed(timestamps):
            if ts >= cutoff:
                recent_count += 1
                if first_recent is None:
                    first_recent = ts
                last_recent = ts
            else:
                break  # All older timestamps will also be too old
        
        if recent_count < 2 or first_recent is None or last_recent is None:
            return 0.0
        
        time_span = first_recent - last_recent
        return (recent_count - 1) / time_span if time_span > 0 else 0.0
    
    def get_fps_stats(self) -> Dict[str, float]:
        """Get comprehensive FPS statistics."""
        return {
            "ingress_video_fps": self.get_ingress_video_fps(),
            "ingress_audio_fps": self.get_ingress_audio_fps(),
            "egress_video_fps": self.get_egress_video_fps(),
            "egress_audio_fps": self.get_egress_audio_fps()
        }
    
    def reset(self):
        """Reset all measurements."""
        self.ingress_video_timestamps.clear()
        self.ingress_audio_timestamps.clear()
        self.egress_video_timestamps.clear()
        self.egress_audio_timestamps.clear()
        
        # Reset cache
        self._last_calculation_time = 0
        self._cached_ingress_fps = 0.0
        self._cached_egress_fps = 0.0
