"""
Adaptive Frame Skipper - Self-updating intelligent frame dropping.

Monitors processing throughput and adaptively drops frames to maintain
real-time performance with automatic target FPS detection.
"""

import time
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
from fractions import Fraction
from .frames import VideoFrame, AudioFrame
from .fps_meter import FPSMeter

logger = logging.getLogger(__name__)

class FrameProcessingResult(Enum):
    """Explicit result types for frame processing operations."""
    SKIPPED = "skipped"  # Frame was skipped due to adaptive logic

FrameResult = Union[VideoFrame, AudioFrame, FrameProcessingResult, None]
@dataclass
class FrameSkipConfig:
    """Configuration for adaptive frame skipping behavior.
    
    This allows StreamProcessor users to easily configure intelligent frame skipping
    for optimal real-time performance.
    """
    target_fps: Optional[float] = None  # Target FPS (None = auto-detect from ingress)
    adaptation_window: float = 2.0      # Time window for FPS measurement
    adaptation_cooldown: float = 2.0    # Minimum time between adaptations
    max_queue_size: int = 50           # Start dropping frames when queue exceeds this
    max_cleanup_frames: int = 20       # Maximum frames to drop in one cleanup
    
class AdaptiveFrameSkipper:
    """
    Adaptive frame skipper that automatically detects ingress FPS
    and adjusts frame dropping to maintain real-time performance.
    
    Enhanced with timestamp synchronization to prevent encoder DTS errors.
    """
    
    def __init__(
        self,
        config: FrameSkipConfig,
        fps_meter: 'FPSMeter'
    ):
        """
        Initialize adaptive frame skipper.
        
        Args:
            config: FrameSkipConfig object with all configuration
            fps_meter: FPSMeter from protocol for consistent measurements
        """
        self.config = config
        self.fps_meter = fps_meter
        
        # Frame counting for skip patterns
        self.frame_counter = 0
        self.skip_interval = 1  # Skip every N frames (1 = no skipping)
        
        # Last adaptation time to prevent too frequent changes
        self.last_adaptation_time = time.time()
        
        # Simple frame counting
        self.video_frame_count = 0

    async def process_queue(self, input_queue: asyncio.Queue, timeout: float = 5) -> FrameResult:
        """
        Get frames from input queue with intelligent skipping to maintain real-time performance.
        Only skips video frames - audio frames are always processed.
        
        Args:
            input_queue: Asyncio queue containing frames to process
            timeout: Timeout for queue operations
            
        Returns:
            - VideoFrame or AudioFrame: Frame to process
            - None: Queue received shutdown sentinel
            - FrameProcessingResult.SKIPPED: Frame was skipped, caller should get next frame
            - Raises asyncio.TimeoutError: Timeout occurred, no frame available
        """
        try:
            # Handle queue overflow by skipping excess video frames
            await self._cleanup_queue_overflow(input_queue, timeout)
            
            # Get the next frame to potentially process
            frame = await asyncio.wait_for(input_queue.get(), timeout=timeout)
            if frame is None:
                return None  # Sentinel value for shutdown
            
            # Handle frame based on type
            return self._process_frame(frame)
            
        except asyncio.TimeoutError:
            raise  # Let the caller handle the timeout


    def _process_frame(self, frame: Union[VideoFrame, AudioFrame]) -> FrameResult:
        """Process a frame to determine if it should be skipped."""
        if isinstance(frame, VideoFrame):
            return self._process_video_frame(frame)
        else:
            # Audio frames always pass through unchanged - sync handled after frame skipper
            return frame

    def _process_video_frame(self, frame: VideoFrame) -> FrameResult:
        """Process video frame and apply skipping logic."""
        # Count frame for skip pattern
        self.frame_counter += 1
        self.video_frame_count += 1
        
        # Update skip pattern and check if frame should be skipped
        self._adapt_skip_interval()
        should_skip = self._should_skip_frame()
        
        if should_skip:
            return FrameProcessingResult.SKIPPED  # Frame was skipped
        else:
            return frame  # Frame passes through unchanged
    
    def _adapt_skip_interval(self):
        """Adapt skip interval based on current FPS measurements."""
        current_time = time.time()
        
        # Only adapt if enough time has passed (avoid expensive FPS calculation)
        if current_time - self.last_adaptation_time < self.config.adaptation_cooldown:
            return
        
        ingress_fps = self.fps_meter.get_ingress_video_fps()
        
        # Need sufficient data for reliable measurement
        if ingress_fps < 1.0:
            return
        
        # Use ingress FPS as target if not specified (auto-detect mode)
        target = self.config.target_fps if self.config.target_fps is not None else ingress_fps
        
        # Calculate skip interval: skip_interval = ingress_fps / target_fps
        if ingress_fps <= target:
            new_interval = 1
        else:
            new_interval = max(1, round(ingress_fps / target))
        
        # Apply new skip interval
        if new_interval != self.skip_interval:
            logger.info(f"Skip pattern: {ingress_fps:.1f}fps → {target:.1f}fps (skip every {new_interval} frames)")
            self.skip_interval = new_interval
        
        self.last_adaptation_time = current_time

    def _should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped."""
        if self.skip_interval <= 1:
            return False
        
        # Skip every N frames uniformly
        return (self.frame_counter % self.skip_interval) != 1
    
    def reset(self):
        """Reset frame counter and measurements."""
        self.frame_counter = 0
        self.skip_interval = 1
        self.video_frame_count = 0
    
    def set_target_fps(self, target_fps: Optional[float]):
        """Set target FPS and recalculate skip pattern.
        
        Args:
            target_fps: Target FPS (None = auto-detect from ingress)
        """
        if target_fps is None or target_fps > 0:
            self.config.target_fps = target_fps
            # Force immediate recalculation
            self.last_adaptation_time = 0
            self._adapt_skip_interval()
            
            if target_fps is None:
                logger.info("Auto-detect target FPS enabled")
            else:
                logger.info(f"Target FPS set to {target_fps}")
        else:
            logger.warning(f"Invalid target FPS: {target_fps}, keeping current: {self.config.target_fps}")
