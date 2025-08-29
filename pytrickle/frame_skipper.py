"""
Adaptive Frame Skipper - Self-updating intelligent frame dropping.

Monitors processing throughput and adaptively drops frames to maintain
real-time performance with automatic target FPS detection.
"""

import time
import logging
import asyncio
from collections import deque
from typing import Optional, Dict, Any, Union
from .fps_meter import FPSMeter
from .frames import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)


class AdaptiveFrameSkipper:
    """
    Self-updating adaptive frame skipper that automatically detects ingress FPS
    and adjusts frame dropping to maintain real-time processing.
    """
    
    def __init__(
        self,
        target_fps: Optional[float] = None,
        adaptation_window: float = 2.0,  # Seconds to measure performance
        skip_pattern: str = "uniform"  # "uniform", "keyframe", or "temporal"
    ):
        """
        Initialize self-updating adaptive frame skipper.
        
        Args:
            target_fps: Target FPS (None = auto-detect from ingress)
            adaptation_window: Time window for FPS measurement
            skip_pattern: Frame skipping pattern strategy
        """
        self.target_fps = target_fps  # None means auto-detect from ingress
        self.adaptation_window = adaptation_window
        self.skip_pattern = skip_pattern
        
        # Performance tracking
        self.fps_meter = FPSMeter(window_seconds=adaptation_window)
        
        # Frame counting for skip patterns
        self.frame_counter = 0
        self.skip_interval = 1  # Skip every N frames (1 = no skipping)
        
        # Last adaptation time to prevent too frequent changes
        self.last_adaptation_time = time.time()
        self.adaptation_cooldown = 2.0  # Skip pattern adaptation interval

    async def process_queue_with_skipping(self, input_queue: asyncio.Queue, timeout: float = 5) -> Union[VideoFrame, AudioFrame, bool, None]:
        """
        Get frames from input queue with intelligent skipping to maintain real-time performance.
        Only skips video frames - audio frames are always processed.
        
        This method will remove multiple video frames from the queue if needed to catch up with
        real-time processing requirements.
        
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
            # First check if we need to skip video frames based on queue size
            initial_queue_size = input_queue.qsize()
            additional_skips = self._calculate_additional_skips(initial_queue_size)
            
            # Skip video frames from queue first if we're behind
            skipped_count = 0
            for _ in range(additional_skips):
                try:
                    skipped_frame = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                    if skipped_frame is None:
                        return None  # Hit sentinel, stop processing
                    
                    # Only skip if it's a video frame
                    if isinstance(skipped_frame, VideoFrame):
                        # Record ingress frame for FPS measurement
                        self.fps_meter.record_ingress_video_frame()
                        skipped_count += 1
                    else:
                        # Audio frames now have corrected timestamps from ingress loop
                        return skipped_frame
                    
                except asyncio.TimeoutError:
                    break  # No more frames available quickly
            
            # Now get the frame we will process
            frame = await asyncio.wait_for(input_queue.get(), timeout=timeout)
            if frame is None:
                return None  # Sentinel value for shutdown
            
            # Apply skip pattern to video frames only
            if isinstance(frame, VideoFrame):
                # Count frame and record for FPS measurement
                self.frame_counter += 1
                self.fps_meter.record_ingress_video_frame()
                
                # Check adaptation and skip pattern
                self._adapt_skip_pattern()
                should_skip = self._apply_skip_pattern()
                
                if should_skip:
                    # Return False to indicate this frame should be skipped
                    # The caller should handle getting the next frame
                    return False
                else:
                    self.fps_meter.record_egress_video_frame()
            # Audio frames pass through unchanged without counting
            
            return frame
            
        except asyncio.TimeoutError:
            raise  # Let the caller handle the timeout

    def _calculate_additional_skips(self, queue_size: int) -> int:
        """Simple queue overflow prevention only."""
        if queue_size <= 10:
            return 0
        
        # Pure queue management - no performance logic
        # This is just overflow prevention, not FPS matching
        if queue_size >= 110:  # Emergency - queue nearly full
            return min(queue_size - 50, 40)  # Aggressive cleanup
        elif queue_size >= 80:   # Queue getting full
            return min(queue_size - 60, 20)  # Moderate cleanup
        elif queue_size >= 40:   # Queue building up
            return min(queue_size - 30, 10)  # Light cleanup
        else:
            return 0  # Let direct target logic handle normal operation
    
    def _adapt_skip_pattern(self):
        """Self-updating adaptation with performance optimization."""
        current_time = time.time()
        
        # Only adapt skip pattern if enough time has passed
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return  # EXIT EARLY - avoid expensive FPS calculation
        
        # Now do the expensive FPS calculation only when needed
        ingress_fps = self.fps_meter.get_ingress_video_fps()
        
        # Need sufficient data points for reliable measurement
        if ingress_fps < 1.0:
            return
        
        # Use ingress FPS as target if target_fps is None (auto-detect mode)
        effective_target_fps = self.target_fps if self.target_fps is not None else ingress_fps
        
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
        
        if self.skip_pattern == "uniform":
            # Skip every N frames uniformly
            return (self.frame_counter % self.skip_interval) != 1
        
        elif self.skip_pattern == "keyframe":
            # Simple keyframe pattern - could be enhanced with actual keyframe detection
            return (self.frame_counter % self.skip_interval) != 1
        
        elif self.skip_pattern == "temporal":
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
            self.target_fps = new_target_fps
            # Force immediate recalculation of skip pattern
            self.last_adaptation_time = 0
            self._adapt_skip_pattern()
            if new_target_fps is None:
                logger.info("Updated to auto-detect target FPS from ingress")
            else:
                logger.info(f"Manually updated target_fps to {new_target_fps}")
        else:
            logger.warning(f"Invalid target_fps value: {new_target_fps}, keeping current value: {self.target_fps}")
    
    def enable_auto_target_fps(self):
        """Enable automatic target FPS detection (same as setting target_fps=None)."""
        self.update_target_fps(None)
