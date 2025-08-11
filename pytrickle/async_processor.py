"""
Async Frame Processor - Async processing utilities for PyTrickle.

This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional, Callable, Any, Dict, List
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput

logger = logging.getLogger(__name__)

class AsyncFrameProcessor(ABC):
    """
    Base class for async frame processors.
    
    This class provides a bridge between sync frame processing (required by trickle)
    and async AI processing (used by most AI models). It handles:
    - Async processing queue management
    - Frame buffering and fallback strategies
    - Error handling and recovery
    
    Subclass this to implement your async AI processing logic.
    """
    
    def __init__(
        self, 
        queue_maxsize: int = 30,
        error_callback: Optional[Callable[[Exception], None]] = None
    ):
        """Initialize the async frame processor."""
        self.queue_maxsize = queue_maxsize
        self.error_callback = error_callback
        
        # Processing state
        self.is_started = False
        self.processor_task: Optional[asyncio.Task] = None
        self.frame_count = 0
        
        # Queues for frame processing
        self.input_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.video_output_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.audio_output_queue = asyncio.Queue(maxsize=queue_maxsize)
        
        # Frame correlation and fallback storage
        self.pending_video_frames: Dict[int, VideoFrame] = {}
        self.pending_audio_frames: Dict[int, AudioFrame] = {}
        self.last_video_frame: Optional[VideoFrame] = None
        self.last_audio_frame: Optional[AudioFrame] = None
        
        # Event coordination
        self.shutdown_event = asyncio.Event()
        
        logger = logging.getLogger(self.__class__.__name__)
    
    async def start(self):
        """Start the async processor."""
        if self.is_started and self.processor_task and not self.processor_task.done():
            return
        
        logger.info(f"Starting {self.__class__.__name__}")
        self.is_started = True
        
        # Cancel existing task if it exists and is done/cancelled
        if self.processor_task and self.processor_task.done():
            self.processor_task = None
        
        # Start async processing task
        if not self.processor_task:
            self.processor_task = asyncio.create_task(self._process_frames_async())
    
    async def stop(self):
        """Stop the async processor."""
        if not self.is_started:
            return
        
        logger.info(f"Stopping {self.__class__.__name__}")
        self.is_started = False
        
        # Cancel processor task
        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Clean up queues
        await self._clear_queues()
        
        logger.info(f"{self.__class__.__name__} stopped")
    
    def process_frame_sync(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        """
        Sync interface for frame processing (called by trickle).
        
        This method bridges sync and async processing by:
        1. Enqueueing frames for async processing
        2. Returning processed frames from the output queue
        3. Providing fallback frames when processing isn't ready
        """
        self.frame_count += 1
        
        try:
            if isinstance(frame, VideoFrame):
                return self._process_video_frame_sync(frame)
            elif isinstance(frame, AudioFrame):
                return self._process_audio_frame_sync(frame)
            else:
                # Handle unknown frame types
                logger.warning(f"Unknown frame type: {type(frame)}")
                return self._get_fallback_output(frame)
        
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            if self.error_callback:
                try:
                    self.error_callback(e)
                except Exception:
                    pass
            return self._get_fallback_output(frame)
    
    def _process_video_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        """Process video frame in sync interface."""
        # Store for fallback
        self.last_video_frame = frame
        
        # Enqueue for async processing (non-blocking)
        if self.is_started:
            try:
                self.input_queue.put_nowait(("video", frame, self.frame_count))
            except asyncio.QueueFull:
                logger.debug(f"Input queue full, skipping video frame {self.frame_count}")
        
        # Try to get processed frame
        try:
            processed_frame = self.video_output_queue.get_nowait()
            # Update last processed frame for better fallback
            self.last_video_frame = processed_frame
            return VideoOutput(processed_frame, f"{self.__class__.__name__}_processed")
        except asyncio.QueueEmpty:
            # Return fallback - this will use last_video_frame if available
            return self._get_video_fallback(frame)
        except Exception as e:
            logger.error(f"Error getting processed video frame: {e}")
            return self._get_video_fallback(frame)
    
    def _process_audio_frame_sync(self, frame: AudioFrame) -> AudioOutput:
        """Process audio frame in sync interface."""
        # For audio frames, we want to pass them through unchanged to avoid encoder issues
        # This maintains proper audio timing and prevents encoder breaks
        
        # Store for potential fallback use
        self.last_audio_frame = frame
        
        # Don't enqueue audio for async processing - just pass through
        # This prevents timing issues and encoder breaks
        logger.debug(f"Audio frame passthrough: {self.frame_count}")
        
        # Return the original frame unchanged for immediate output
        return AudioOutput([frame], f"{self.__class__.__name__}_passthrough")
    
    def _get_fallback_output(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        """Get fallback output for unknown frame types."""
        if isinstance(frame, VideoFrame):
            return self._get_video_fallback(frame)
        elif isinstance(frame, AudioFrame):
            return self._get_audio_fallback(frame)
        else:
            raise ValueError(f"Unknown frame type: {type(frame)}")
    
    def _get_video_fallback(self, frame: VideoFrame) -> VideoOutput:
        """Get fallback video output."""
        # If we have a last processed frame, use it as fallback
        if self.last_video_frame is not None:
            # Return the last processed frame to maintain consistency
            return VideoOutput(self.last_video_frame, f"{self.__class__.__name__}_fallback")
        # Otherwise return the current frame as passthrough
        return VideoOutput(frame, f"{self.__class__.__name__}_passthrough")
    
    def _get_audio_fallback(self, frame: AudioFrame) -> AudioOutput:
        """Get fallback audio output."""
        # If we have a last processed audio frame, use it as fallback
        if self.last_audio_frame is not None:
            # Return the last processed audio frame to maintain consistency
            return AudioOutput([self.last_audio_frame], f"{self.__class__.__name__}_fallback")
        # Otherwise return the current frame as passthrough
        return AudioOutput([frame], f"{self.__class__.__name__}_passthrough")
    
    async def _process_frames_async(self):
        """Main async processing loop."""
        try:
            while self.is_started:
                try:
                    # Get frame from input queue
                    frame_data = await asyncio.wait_for(
                        self.input_queue.get(),
                        timeout=1.0
                    )
                    
                    frame_type, frame, frame_id = frame_data
                    
                    if frame_type == "video":
                        # Store frame for correlation
                        self.pending_video_frames[frame_id] = frame
                        processed_frame = await self.process_video_async(frame)
                        if processed_frame:
                            # Ensure frame gets added to output queue
                            await self._add_to_video_queue(processed_frame, frame_id)
                    
                    elif frame_type == "audio":
                        # Audio frames are now handled with immediate passthrough in sync interface
                        # Skip async processing to avoid timing issues
                        logger.debug(f"Skipping async audio processing for frame {frame_id}")
                        continue
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in async processing: {e}")
                    if self.error_callback:
                        try:
                            self.error_callback(e)
                        except Exception:
                            pass
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Async processor error: {e}")
    
    async def _add_to_video_queue(self, processed_frame: VideoFrame, frame_id: int):
        """Add processed video frame to output queue with proper queue management."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.video_output_queue.put_nowait(processed_frame)
                # Clean up correlation data for processed frame
                self.pending_video_frames.pop(frame_id, None)
                return
            except asyncio.QueueFull:
                if attempt < max_attempts - 1:
                    # Remove oldest frame to make room
                    try:
                        self.video_output_queue.get_nowait()
                        logger.debug("Removed oldest video frame to make room for new frame")
                    except asyncio.QueueEmpty:
                        # Queue was emptied, try again
                        continue
                else:
                    logger.warning(f"Could not add processed video frame {frame_id} to output queue after {max_attempts} attempts")
    
    async def _add_to_audio_queue(self, processed_frames: List[AudioFrame], frame_id: int):
        """Add processed audio frames to output queue with proper queue management."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.audio_output_queue.put_nowait(processed_frames)
                # Clean up correlation data for processed frame
                self.pending_audio_frames.pop(frame_id, None)
                return
            except asyncio.QueueFull:
                if attempt < max_attempts - 1:
                    # Remove oldest frame to make room
                    try:
                        self.audio_output_queue.get_nowait()
                        logger.debug("Removed oldest audio frame to make room for new frame")
                    except asyncio.QueueEmpty:
                        # Queue was emptied, try again
                        continue
                else:
                    logger.warning(f"Could not add processed audio frame {frame_id} to output queue after {max_attempts} attempts")
    
    async def _clear_queues(self):
        """Clear all processing queues."""
        queues = [self.input_queue, self.video_output_queue, self.audio_output_queue]
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
    
    # Abstract methods to implement in subclasses
    
    @abstractmethod
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """
        Process a video frame asynchronously.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed video frame or None if processing failed
        """
        pass
    
    @abstractmethod
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """
        Process an audio frame asynchronously.
        
        Args:
            frame: Input audio frame
            
        Returns:
            List of processed audio frames or None if processing failed
        """
        pass
    
    def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters (optional override).
        
        Args:
            params: Dictionary of parameters to update
        """
        pass
    
    def create_sync_bridge(self) -> Callable:
        """
        Create a sync frame processor function from this AsyncFrameProcessor.
        
        This method creates the sync interface needed by TrickleApp and other
        sync contexts, bridging the async processor with sync frame processing.
        
        Returns:
            Sync frame processor function that can be used with TrickleApp
            
        Example:
            processor = MyAIProcessor()
            await processor.start()
            
            app = TrickleApp(frame_processor=processor.create_sync_bridge())
            await app.run_forever()
        """
        def frame_processor(frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
            return self.process_frame_sync(frame)
        
        return frame_processor
    
    @classmethod
    def create_bridge(cls, async_processor: 'AsyncFrameProcessor') -> Callable:
        """
        Create a sync frame processor function from an AsyncFrameProcessor instance.
        
        This is a class method convenience function that creates the sync interface
        needed by TrickleApp and other sync contexts.
        
        Args:
            async_processor: An AsyncFrameProcessor instance
            
        Returns:
            Sync frame processor function
            
        Example:
            processor = MyAIProcessor()
            await processor.start()
            
            bridge = AsyncFrameProcessor.create_bridge(processor)
            app = TrickleApp(frame_processor=bridge)
            await app.run_forever()
        """
        return async_processor.create_sync_bridge()
