"""
Async Frame Processor - Async processing utilities for PyTrickle.

This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Union, Optional, Callable, Any, Dict
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
        """
        Initialize the async frame processor.
        
        Args:
            queue_maxsize: Maximum size for processing queues
            error_callback: Callback for error handling
        """
        self.queue_maxsize = queue_maxsize
        self.error_callback = error_callback
        
        # Processing queues
        self.input_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.video_output_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.audio_output_queue = asyncio.Queue(maxsize=queue_maxsize)
        
        # State management
        self.is_started = False
        self.frame_count = 0
        
        # Async tasks
        self.processor_task: Optional[asyncio.Task] = None
        
        # Fallback frames for when processing isn't ready
        self.last_video_frame: Optional[VideoFrame] = None
        self.last_audio_frame: Optional[AudioFrame] = None
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    async def start(self):
        """Start the async processor."""
        if self.is_started:
            return
        
        logger.info(f"Starting {self.__class__.__name__}")
        self.is_started = True
        
        # Start async processing task
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
        if self.is_started and not self.input_queue.full():
            try:
                self.input_queue.put_nowait(("video", frame, self.frame_count))
            except asyncio.QueueFull:
                pass  # Skip frame if queue is full
        
        # Try to get processed frame
        try:
            processed_frame = self.video_output_queue.get_nowait()
            return VideoOutput(processed_frame, f"{self.__class__.__name__}_processed")
        except asyncio.QueueEmpty:
            # Return fallback
            return self._get_video_fallback(frame)
    
    def _process_audio_frame_sync(self, frame: AudioFrame) -> AudioOutput:
        """Process audio frame in sync interface."""
        # Store for fallback
        self.last_audio_frame = frame
        
        # Enqueue for async processing (non-blocking)
        if self.is_started and not self.input_queue.full():
            try:
                self.input_queue.put_nowait(("audio", frame, self.frame_count))
            except asyncio.QueueFull:
                pass  # Skip frame if queue is full
        
        # Try to get processed frame
        try:
            processed_frames = self.audio_output_queue.get_nowait()
            return AudioOutput(processed_frames, f"{self.__class__.__name__}_processed")
        except asyncio.QueueEmpty:
            # Return fallback
            return self._get_audio_fallback(frame)
    
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
        return VideoOutput(frame, f"{self.__class__.__name__}_passthrough")
    
    def _get_audio_fallback(self, frame: AudioFrame) -> AudioOutput:
        """Get fallback audio output."""
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
                        processed_frame = await self.process_video_async(frame)
                        if processed_frame:
                            try:
                                self.video_output_queue.put_nowait(processed_frame)
                            except asyncio.QueueFull:
                                # Remove oldest and add new
                                try:
                                    self.video_output_queue.get_nowait()
                                    self.video_output_queue.put_nowait(processed_frame)
                                except asyncio.QueueEmpty:
                                    pass
                    
                    elif frame_type == "audio":
                        processed_frames = await self.process_audio_async(frame)
                        if processed_frames:
                            try:
                                self.audio_output_queue.put_nowait(processed_frames)
                            except asyncio.QueueFull:
                                # Remove oldest and add new
                                try:
                                    self.audio_output_queue.get_nowait()
                                    self.audio_output_queue.put_nowait(processed_frames)
                                except asyncio.QueueEmpty:
                                    pass
                
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
    async def process_audio_async(self, frame: AudioFrame) -> Optional[list[AudioFrame]]:
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
