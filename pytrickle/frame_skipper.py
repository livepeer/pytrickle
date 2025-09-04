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
from .frames import VideoFrame, AudioFrame
from .fps_meter import FPSMeter

logger = logging.getLogger(__name__)

class FrameProcessingResult(Enum):
    """Explicit result types for frame processing operations."""
    SKIPPED = "skipped"  # Frame was skipped due to adaptive logic

# Type alias for video frame processing result
FrameResult = Union[VideoFrame, FrameProcessingResult, None]
@dataclass
class FrameSkipConfig:
    """Configuration for adaptive frame skipping behavior."""
    target_fps: Optional[float] = None  # Target FPS (None = auto-detect from ingress)
    adaptation_cooldown: float = 0.5    # Minimum time between adaptations
    max_queue_size: int = 15           # Start dropping frames when queue exceeds this
    max_cleanup_frames: int = 50       # Maximum frames to drop in one cleanup
    
class AdaptiveFrameSkipper:
    """
    Adaptive frame skipper that automatically detects ingress FPS
    and adjusts frame dropping to maintain real-time performance.
    """
    
    def __init__(
        self,
        config: FrameSkipConfig,
        fps_meter: 'FPSMeter'
    ):
        """Initialize adaptive frame skipper."""
        self.config = config
        self.fps_meter = fps_meter
        
        # Frame counting for skip patterns
        self.frame_counter = 0
        self.skip_interval = 1  # Skip every N frames (1 = no skipping)
        
        # Last adaptation time to prevent too frequent changes
        self.last_adaptation_time = time.time()

    async def process_video_queue(self, video_queue: asyncio.Queue, timeout: float = 5) -> FrameResult:
        """
        Get video frames from queue with intelligent skipping to maintain real-time performance.
        
        Args:
            video_queue: Asyncio queue containing only video frames
            timeout: Timeout for queue operations
            
        Returns:
            - VideoFrame: Frame to process
            - None: Queue received shutdown sentinel
            - FrameProcessingResult.SKIPPED: Frame was skipped, caller should get next frame
            - Raises asyncio.TimeoutError: Timeout occurred, no frame available
        """
        try:
            # Handle queue overflow by skipping excess video frames
            await self._cleanup_video_queue_overflow(video_queue, timeout)
            
            # Get the next frame to potentially process
            frame = await asyncio.wait_for(video_queue.get(), timeout=timeout)
            if frame is None:
                return None  # Sentinel value for shutdown
            
            # Process video frame with skipping logic
            return self._process_video_frame(frame)
            
        except asyncio.TimeoutError:
            raise  # Let the caller handle the timeout

    async def _cleanup_video_queue_overflow(self, video_queue: asyncio.Queue, timeout: float):
        """Clean up video queue overflow by dropping excess frames."""
        queue_size = video_queue.qsize()
        
        if queue_size <= self.config.max_queue_size:
            return
        
        frames_to_drop = min(queue_size - self.config.max_queue_size, self.config.max_cleanup_frames)
        
        # Drop video frames (no need to check type since this is video-only queue)
        sentinel_consumed = False
        for _ in range(frames_to_drop):
            try:
                frame = await asyncio.wait_for(video_queue.get(), timeout=timeout)
                if frame is None:
                    sentinel_consumed = True
                    break
                # Simply drop the video frame (no requeuing needed)
            except asyncio.TimeoutError:
                break  # No more frames available
        
        if sentinel_consumed:
            await video_queue.put(None)  # Re-queue sentinel

    def _process_video_frame(self, frame: VideoFrame) -> FrameResult:
        """Process video frame and apply skipping logic."""
        # Count frame for skip pattern
        self.frame_counter += 1
        
        # Update skip pattern and check if frame should be skipped
        self._adapt_skip_interval()
        should_skip = self._should_skip_frame()
        
        if should_skip:
            return FrameProcessingResult.SKIPPED
        else:
            return frame
    
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
