"""
Adaptive Frame Skipper - Self-updating intelligent frame dropping.

Monitors processing throughput and adaptively drops frames to maintain
real-time performance with automatic target FPS detection.
"""

import time
import logging
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Literal
from .fps_meter import FPSMeter
from .frames import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)

# Type alias for cleaner return type annotations
FrameResult = Union[VideoFrame, AudioFrame, Literal[False], None]

@dataclass
class SkippingConfig:
    """Configuration for adaptive frame skipping behavior."""
    target_fps: Optional[float] = None
    adaptation_window: float = 2.0
    skip_pattern: str = "uniform"  # "uniform", "keyframe", or "temporal"
    adaptation_cooldown: float = 2.0
    
    # Simple queue overflow prevention
    max_queue_size: int = 50  # Start dropping frames when queue exceeds this
    max_cleanup_frames: int = 20  # Maximum frames to drop in one cleanup


class AdaptiveFrameSkipper:
    """
    Self-updating adaptive frame skipper that automatically detects ingress FPS
    and adjusts frame dropping to maintain real-time processing.
    """
    
    def __init__(
        self,
        target_fps: Optional[float] = None,
        adaptation_window: float = 2.0,
        skip_pattern: str = "uniform",
        config: Optional[SkippingConfig] = None
    ):
        """
        Initialize self-updating adaptive frame skipper.
        
        Args:
            target_fps: Target FPS (None = auto-detect from ingress)
            adaptation_window: Time window for FPS measurement  
            skip_pattern: Frame skipping pattern strategy
            config: Optional configuration object (overrides individual params)
        """
        # Use config if provided, otherwise create from individual params
        if config is not None:
            self.config = config
        else:
            self.config = SkippingConfig(
                target_fps=target_fps,
                adaptation_window=adaptation_window,
                skip_pattern=skip_pattern
            )
        
        # Performance tracking
        self.fps_meter = FPSMeter(window_seconds=self.config.adaptation_window)
        
        # Frame counting for skip patterns
        self.frame_counter = 0
        self.skip_interval = 1  # Skip every N frames (1 = no skipping)
        
        # Last adaptation time to prevent too frequent changes
        self.last_adaptation_time = time.time()

    async def process_queue_with_skipping(self, input_queue: asyncio.Queue, timeout: float = 5) -> FrameResult:
        """
        Get frames from input queue with intelligent skipping to maintain real-time performance.
        Only skips video frames - audio frames are always processed.
        
        Args:
            input_queue: Asyncio queue containing frames to process
            timeout: Timeout for queue operations
            
        Returns:
            - VideoFrame or AudioFrame: Frame to process
            - None: Queue received shutdown sentinel
            - False: Frame was skipped, caller should get next frame
            - Raises asyncio.TimeoutError: Timeout occurred, no frame available
        """
        try:
            # First handle queue overflow by skipping excess video frames
            overflow_frame = await self._handle_queue_overflow(input_queue, timeout)
            if overflow_frame is not None:
                return overflow_frame  # Return audio frame or sentinel
            
            # Get the next frame to potentially process
            frame = await asyncio.wait_for(input_queue.get(), timeout=timeout)
            if frame is None:
                return None  # Sentinel value for shutdown
            
            # Handle frame based on type
            return self._process_frame_for_skipping(frame)
            
        except asyncio.TimeoutError:
            raise  # Let the caller handle the timeout

    async def _handle_queue_overflow(self, input_queue: asyncio.Queue, timeout: float) -> Optional[Union[AudioFrame, None]]:
        """Handle queue overflow by skipping excess video frames."""
        queue_size = input_queue.qsize()
        
        # Simple check: if queue is too big, drop some frames
        if queue_size > self.config.max_queue_size:
            frames_to_drop = min(queue_size - self.config.max_queue_size, self.config.max_cleanup_frames)
            
            # Drop frames, but preserve any audio frames we encounter
            for _ in range(frames_to_drop):
                try:
                    dropped_frame = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                    if dropped_frame is None:
                        return None  # Hit sentinel, stop processing
                    
                    if isinstance(dropped_frame, VideoFrame):
                        self.fps_meter.record_ingress_video_frame()  # Count it for stats
                    else:
                        # Audio frame encountered - return it immediately (don't drop audio!)
                        return dropped_frame
                        
                except asyncio.TimeoutError:
                    break  # No more frames available quickly
        
        return None  # No audio frame found during cleanup

    def _process_frame_for_skipping(self, frame: Union[VideoFrame, AudioFrame]) -> FrameResult:
        """Process a frame to determine if it should be skipped."""
        if isinstance(frame, VideoFrame):
            return self._handle_video_frame(frame)
        else:
            # Audio frames always pass through
            return frame

    def _handle_video_frame(self, frame: VideoFrame) -> FrameResult:
        """Handle video frame processing and skipping logic."""
        # Count frame and record for FPS measurement
        self.frame_counter += 1
        self.fps_meter.record_ingress_video_frame()
        
        # Check adaptation and skip pattern
        self._adapt_skip_pattern()
        should_skip = self._apply_skip_pattern()
        
        if should_skip:
            return False  # Frame was skipped
        else:
            self.fps_meter.record_egress_video_frame()
            return frame
    
    def _adapt_skip_pattern(self):
        """Self-updating adaptation with performance optimization."""
        current_time = time.time()
        
        # Only adapt skip pattern if enough time has passed
        if current_time - self.last_adaptation_time < self.config.adaptation_cooldown:
            return  # EXIT EARLY - avoid expensive FPS calculation
        
        # Now do the expensive FPS calculation only when needed
        ingress_fps = self.fps_meter.get_ingress_video_fps()
        
        # Need sufficient data points for reliable measurement
        if ingress_fps < 1.0:
            return
        
        # Use ingress FPS as target if target_fps is None (auto-detect mode)
        effective_target_fps = self.config.target_fps if self.config.target_fps is not None else ingress_fps
        
        # Direct calculation: skip_interval = ingress_fps / target_fps
        if ingress_fps <= effective_target_fps:
            new_skip_interval = 1
        else:
            new_skip_interval = max(1, round(ingress_fps / effective_target_fps))
        
        # Apply new skip interval
        if new_skip_interval != self.skip_interval:
            logger.info(f"Skip pattern update: ingress_fps={ingress_fps:.1f}, "
                       f"target_fps={effective_target_fps:.1f}, skip_interval={self.skip_interval}->{new_skip_interval}")
            self.skip_interval = new_skip_interval
        
        self.last_adaptation_time = current_time


    def _apply_skip_pattern(self) -> bool:
        """Apply the current skip pattern to determine if this frame should be skipped."""
        if self.skip_interval <= 1:
            return False  # No skipping
        
        if self.config.skip_pattern == "uniform":
            # Skip every N frames uniformly
            return (self.frame_counter % self.skip_interval) != 1
        
        elif self.config.skip_pattern == "keyframe":
            # Simple keyframe pattern - could be enhanced with actual keyframe detection
            return (self.frame_counter % self.skip_interval) != 1
        
        elif self.config.skip_pattern == "temporal":
            # Skip frames with more temporal distance from last processed
            # This is a simple implementation - could be more sophisticated
            return (self.frame_counter % self.skip_interval) != 1
        
        else:
            # Default to uniform
            return (self.frame_counter % self.skip_interval) != 1
    
    def reset_statistics(self):
        """Reset frame counter and measurements."""
        self.frame_counter = 0
        self.skip_interval = 1
        self.fps_meter.reset()
    
    def update_target_fps(self, new_target_fps: Optional[float]):
        """Update the target FPS dynamically and recalculate skip pattern.
        
        Args:
            new_target_fps: New target FPS value (None = auto-detect from ingress)
        """
        if new_target_fps is None or new_target_fps > 0:
            self.config.target_fps = new_target_fps
            # Force immediate recalculation of skip pattern
            self.last_adaptation_time = 0
            self._adapt_skip_pattern()
            if new_target_fps is None:
                logger.info("Updated to auto-detect target FPS from ingress")
            else:
                logger.info(f"Manually updated target_fps to {new_target_fps}")
        else:
            logger.warning(f"Invalid target_fps value: {new_target_fps}, keeping current value: {self.config.target_fps}")
    
    def enable_auto_target_fps(self):
        """Enable automatic target FPS detection (same as setting target_fps=None)."""
        self.update_target_fps(None)
