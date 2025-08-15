"""
Adaptive Frame Skipper - Intelligent frame dropping for real-time processing.

Monitors processing throughput vs input rate and adaptively drops frames to maintain
real-time performance when processing falls behind.
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
    Adaptive frame skipper that monitors processing performance and drops frames
    intelligently to maintain real-time processing.
    """
    
    def __init__(
        self,
        target_fps: float = 24.0,
        min_fps_ratio: float = 0.8,  # Skip frames if processing < 80% of input
        max_skip_ratio: float = 0.5,  # Never skip more than 50% of frames
        adaptation_window: float = 2.0,  # Seconds to measure performance
        skip_pattern: str = "uniform"  # "uniform", "keyframe", or "temporal"
    ):
        """
        Initialize adaptive frame skipper.
        
        Args:
            target_fps: Target output FPS
            min_fps_ratio: Minimum processing/input FPS ratio before skipping
            max_skip_ratio: Maximum fraction of frames to skip
            adaptation_window: Time window for performance measurement
            skip_pattern: Frame skipping pattern strategy
        """
        self.target_fps = target_fps
        self.min_fps_ratio = min_fps_ratio
        self.max_skip_ratio = max_skip_ratio
        self.adaptation_window = adaptation_window
        self.skip_pattern = skip_pattern
        
        # Performance tracking
        self.fps_meter = FPSMeter(window_seconds=adaptation_window)
        
        # Frame counting for skip patterns
        self.frame_counter = 0
        self.skip_interval = 1  # Skip every N frames (1 = no skipping)
        
        # Statistics
        self.total_frames_received = 0
        self.total_frames_processed = 0
        self.total_frames_skipped = 0
        self.queue_frames_skipped = 0  # Additional frames skipped from queue
        
        # Last adaptation time to prevent too frequent changes
        self.last_adaptation_time = time.time()
        self.adaptation_cooldown = 2.0  # Slower adaptation - 2.0 seconds between adaptations to prevent oscillation
        
    def should_skip_frame(self, frame_metadata: Optional[Dict[str, Any]] = None, frame_type: str = "video") -> bool:
        """
        Determine if the current frame should be skipped based on performance metrics.
        Only applies to video frames - audio frames are never skipped.
        
        Args:
            frame_metadata: Optional metadata about the frame (timestamp, type, etc.)
            frame_type: Type of frame ("video" or "audio")
            
        Returns:
            True if frame should be skipped, False if it should be processed
        """
        # Only skip video frames, never audio frames
        if frame_type != "video":
            return False
        
        self.total_frames_received += 1
        self.frame_counter += 1
        
        # Record ingress frame for FPS calculation (only for video)
        self.fps_meter.record_ingress_video_frame()
        logger.debug(f"Recorded ingress video frame {self.frame_counter} for FPS calculation")
        
        # Check if we need to adapt the skip pattern
        self._adapt_skip_pattern()
        
        # Determine if this specific frame should be skipped
        should_skip = self._apply_skip_pattern(frame_metadata)
        
        if should_skip:
            self.total_frames_skipped += 1
            logger.debug(f"Skipping video frame {self.frame_counter} (pattern: {self.skip_pattern}, interval: {self.skip_interval})")
        else:
            self.total_frames_processed += 1
            # Record egress frame for FPS calculation (only for video)
            self.fps_meter.record_egress_video_frame()
            logger.debug(f"Recorded egress video frame {self.frame_counter} for FPS calculation")
        
        return should_skip

    async def process_queue_with_skipping(self, input_queue: asyncio.Queue, timeout: float = 5) -> Optional[Union[VideoFrame, AudioFrame]]:
        """
        Get frames from input queue with intelligent skipping to maintain real-time performance.
        Only skips video frames - audio frames are always processed.
        
        This method will remove multiple video frames from the queue if needed to catch up with
        real-time processing requirements.
        
        Args:
            input_queue: Asyncio queue containing frames to process
            timeout: Timeout for queue operations
            
        Returns:
            Frame to process, or None if queue is empty or timeout occurred
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
                        # Count this as a skipped video frame
                        self.total_frames_received += 1
                        self.total_frames_skipped += 1
                        self.queue_frames_skipped += 1
                        self.fps_meter.record_ingress_video_frame()  # Count as received
                        skipped_count += 1
                        logger.debug(f"Skipped video frame from queue (queue_size: {input_queue.qsize()})")
                    else:
                        # Put audio frame back at front of queue
                        temp_queue = asyncio.Queue()
                        await temp_queue.put(skipped_frame)
                        
                        # Move all remaining items to temp queue
                        while not input_queue.empty():
                            try:
                                item = input_queue.get_nowait()
                                await temp_queue.put(item)
                            except asyncio.QueueEmpty:
                                break
                        
                        # Put everything back
                        while not temp_queue.empty():
                            try:
                                item = temp_queue.get_nowait()
                                await input_queue.put(item)
                            except (asyncio.QueueEmpty, asyncio.QueueFull):
                                break
                        break  # Stop skipping since we hit an audio frame
                    
                except asyncio.TimeoutError:
                    break  # No more frames available quickly
            
            if skipped_count > 0:
                logger.debug(f"Skipped {skipped_count} video frames from queue")
            
            # Now get the frame we will actually process
            frame = await asyncio.wait_for(input_queue.get(), timeout=timeout)
            if frame is None:
                return None  # Sentinel value for shutdown
            
            # Check if we should skip this specific frame based on skip pattern
            # Only apply to video frames
            if isinstance(frame, VideoFrame):
                should_skip = self.should_skip_frame(frame_type="video")
                
                if should_skip:
                    # This video frame was marked for individual skipping
                    # Try to get the next frame to process instead
                    try:
                        frame = await asyncio.wait_for(input_queue.get(), timeout=0.1)
                        if frame is None:
                            return None
                        
                        # If the next frame is also video, mark it as processed
                        if isinstance(frame, VideoFrame):
                            self.total_frames_received += 1
                            self.total_frames_processed += 1
                            self.fps_meter.record_ingress_video_frame()
                            self.fps_meter.record_egress_video_frame()
                        # Audio frames don't count in video frame statistics and don't trigger FPS recording
                        
                    except asyncio.TimeoutError:
                        return None  # No replacement frame available
                else:
                    # Video frame is not being skipped, count it as processed
                    # (should_skip_frame already counted it)
                    pass
            else:
                # Audio frame - never skip, never count in video statistics, never record FPS
                # This is critical: audio frames must not affect video FPS measurements
                pass
            
            return frame
            
        except asyncio.TimeoutError:
            return None

    def _calculate_additional_skips(self, queue_size: int) -> int:
        """
        Calculate how many additional frames to skip based on queue size and performance.
        
        Args:
            queue_size: Current size of the input queue
            
        Returns:
            Number of additional frames to skip from queue
        """
        if queue_size <= 2:  # Changed from 1 to 2 for more stability
            return 0
        
        # Get current performance metrics
        input_fps = self.fps_meter.get_ingress_video_fps()
        output_fps = self.fps_meter.get_egress_video_fps()
        
        # Be less aggressive with queue-based skipping for smaller queues
        # Only do aggressive skipping when queue is really full
        if queue_size > 100:  # Queue is 83% full (100/120)
            return min(queue_size - 30, 40)  # Keep more frames, skip less aggressively
        elif queue_size > 80:  # Queue is more than 2/3 full (80/120)
            return min(queue_size - 40, 25)  # Keep more frames
        elif queue_size > 60:  # Queue is 50% full
            return min(queue_size - 45, 15)  # Keep more frames
        elif queue_size > 40:  # Queue is building up
            return min(queue_size // 3, 10)   # Skip 1/3, max 10 (was //2, 15)
        elif queue_size > 20:  # Queue has some buildup
            return min(queue_size // 4, 5)    # Skip 1/4, max 5 (was //3, 8)
        elif queue_size > 10:  # Small buildup
            return min(queue_size // 5, 3)    # Skip 1/5, max 3 (new condition)
        
        # Use performance metrics for fine-tuning when queue is smaller
        if input_fps >= 1.0 and output_fps >= 1.0:
            fps_ratio = output_fps / input_fps
            
            # Be more conservative about skipping from small queues
            if fps_ratio < 0.8:  # Only skip if significantly behind (was 0.9)
                # Calculate required skip ratio to match output capability
                required_skip_ratio = 1.0 - fps_ratio
                frames_to_skip = int(queue_size * required_skip_ratio * 0.5)  # Be more conservative
                
                if fps_ratio < 0.2:
                    # Severely behind, but don't skip too much from small queue
                    return min(max(frames_to_skip, queue_size // 3), 8)  # Was queue_size - 5, 25
                elif fps_ratio < 0.4:
                    # Very behind
                    return min(max(frames_to_skip, queue_size // 4), 5)  # Was queue_size // 2, 15
                elif fps_ratio < 0.6:
                    # Moderately behind
                    return min(max(frames_to_skip, queue_size // 5), 3)  # Was queue_size // 3, 10
                else:
                    # Slightly behind
                    return min(max(frames_to_skip, queue_size // 6), 2)  # Was queue_size // 4, 5
        
        return 0  # Performance is good, no additional skipping needed
    
    def _adapt_skip_pattern(self):
        """Adapt the skip pattern based on current performance metrics."""
        current_time = time.time()
        
        # Only adapt if enough time has passed since last adaptation
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return
        
        # Get current FPS measurements
        input_fps = self.fps_meter.get_ingress_video_fps()
        output_fps = self.fps_meter.get_egress_video_fps()
        
        # Need sufficient data points for reliable measurement
        if input_fps < 1.0 or output_fps < 1.0:
            return
        
        # Calculate simple performance ratio - don't try to estimate "true" input
        # The measured input_fps already reflects the current state
        fps_ratio = output_fps / input_fps if input_fps > 0 else 1.0
        
        # Very conservative adaptation: only change if there's a significant and sustained performance gap
        # Use a much stricter threshold to prevent oscillation
        if fps_ratio < 0.6:  # Only adapt if we're significantly behind (was 0.7)
            # Processing is falling significantly behind, increase skipping
            self._increase_skipping(fps_ratio)
        elif fps_ratio > 1.2 and self.skip_interval > 1:  # Only decrease if we're significantly ahead (was 1.1)
            # Processing is significantly ahead, decrease skipping
            self._decrease_skipping(fps_ratio)
        # If 0.6 <= fps_ratio <= 1.2, do nothing (stable zone)
        
        self.last_adaptation_time = current_time
        
        logger.info(f"Frame skipper adaptation: input_fps={input_fps:.1f}, "
                   f"output_fps={output_fps:.1f}, ratio={fps_ratio:.2f}, "
                   f"skip_interval={self.skip_interval} (stable_zone: 0.6-1.2)")
    
    def _increase_skipping(self, fps_ratio: float):
        """Increase frame skipping to help processing catch up."""
        # Calculate desired skip ratio based on performance deficit
        performance_deficit = 1.0 - fps_ratio
        
        # Be more conservative with skip ratio calculation to prevent over-skipping
        if fps_ratio < 0.2:
            # Very slow processing, but don't skip too aggressively
            desired_skip_ratio = min(0.6, self.max_skip_ratio)
        elif fps_ratio < 0.4:
            # Slow processing
            desired_skip_ratio = min(0.5, self.max_skip_ratio)
        elif fps_ratio < 0.6:
            # Moderate deficit
            desired_skip_ratio = min(0.4, self.max_skip_ratio)
        else:
            # Small deficit, be very conservative
            desired_skip_ratio = min(performance_deficit * 1.5, self.max_skip_ratio)
        
        # Convert skip ratio to skip interval, but be more conservative
        if desired_skip_ratio > 0:
            # Only increase skip_interval by 1 step at a time for stability
            new_skip_interval = self.skip_interval + 1
            max_skip_interval = int(1 / (1 - self.max_skip_ratio)) if self.max_skip_ratio < 1.0 else 10
            self.skip_interval = min(new_skip_interval, max_skip_interval)
        
        logger.info(f"Increased frame skipping: skip_interval={self.skip_interval} "
                   f"(fps_ratio={fps_ratio:.2f}, desired_skip_ratio={desired_skip_ratio:.2f})")
    
    def _decrease_skipping(self, fps_ratio: float):
        """Decrease frame skipping when processing is keeping up."""
        if self.skip_interval > 1:
            self.skip_interval = max(1, self.skip_interval - 1)
            logger.info(f"Decreased frame skipping: skip_interval={self.skip_interval} "
                       f"(fps_ratio={fps_ratio:.2f})")
    
    def _apply_skip_pattern(self, frame_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Apply the current skip pattern to determine if this frame should be skipped."""
        if self.skip_interval <= 1:
            return False  # No skipping
        
        if self.skip_pattern == "uniform":
            # Skip every N frames uniformly
            return (self.frame_counter % self.skip_interval) != 1
        
        elif self.skip_pattern == "keyframe":
            # Try to preserve keyframes if metadata indicates frame type
            if frame_metadata and frame_metadata.get("is_keyframe", False):
                return False  # Never skip keyframes
            return (self.frame_counter % self.skip_interval) != 1
        
        elif self.skip_pattern == "temporal":
            # Skip frames with more temporal distance from last processed
            # This is a simple implementation - could be more sophisticated
            return (self.frame_counter % self.skip_interval) != 1
        
        else:
            # Default to uniform
            return (self.frame_counter % self.skip_interval) != 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive frame skipping statistics."""
        fps_stats = self.fps_meter.get_fps_stats()
        
        skip_ratio = (self.total_frames_skipped / self.total_frames_received 
                     if self.total_frames_received > 0 else 0.0)
        
        return {
            "total_frames_received": self.total_frames_received,
            "total_frames_processed": self.total_frames_processed,
            "total_frames_skipped": self.total_frames_skipped,
            "skip_ratio": skip_ratio,
            "current_skip_interval": self.skip_interval,
            "target_fps": self.target_fps,
            "min_fps_ratio": self.min_fps_ratio,
            "queue_frames_skipped": getattr(self, 'queue_frames_skipped', 0),  # Additional frames skipped from queue
            **fps_stats
        }
    
    def reset_statistics(self):
        """Reset all statistics and measurements."""
        self.total_frames_received = 0
        self.total_frames_processed = 0
        self.total_frames_skipped = 0
        self.queue_frames_skipped = 0
        self.frame_counter = 0
        self.skip_interval = 1
        self.fps_meter.reset()
