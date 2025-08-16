"""
Frame Processor - Async processing utilities for PyTrickle.
This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.

Features:
- Frame skipping to reduce latency during longer processing times
- Fallback frame caching for smooth streaming (video only)
- Async processing with proper error handling and callbacks
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from .frames import VideoFrame, AudioFrame
from . import ErrorCallback
from .state import StreamState, PipelineState

logger = logging.getLogger(__name__)


class FrameProcessor(ABC):
    """
    Base class for async frame processors.

    This class provides native async frame processing for PyTrickle. It handles:
    - initialization and warmup
    - async processing video and audio frames

    Lifecycle:
    1. Processing begins automatically when streams start
    2. Processing stops automatically when streams end

    Usage patterns:

    # HTTP server with StreamServer (recommended)
    processor = MyProcessor()
    app = StreamServer(frame_processor=processor, port=8000)
    await app.run_forever()

    # Direct client usage (advanced)
    protocol = TrickleProtocol(subscribe_url="...", publish_url="...")
    client = TrickleClient(protocol=protocol, frame_processor=processor)
    await client.start("request_id")

    Subclass this to implement your async AI processing logic.
    """

    def __init__(
        self,
        error_callback: Optional[ErrorCallback] = None,
        enable_frame_caching: bool = True,
        **init_kwargs
    ):
        """Initialize the frame processor.
        
        Args:
            error_callback: Optional error callback for processing errors
            enable_frame_caching: Enable frame caching for video fallback behavior
            **init_kwargs: Additional kwargs passed to initialize() method
        """
        self.error_callback = error_callback
        self.state: Optional[StreamState] = None
        self.model_loaded: bool = False

            
        # Frame caching for better video fallback behavior
        self.enable_frame_caching = enable_frame_caching
        self._last_processed_video_frame: Optional[VideoFrame] = None

        self.load_model(**init_kwargs)
        self.model_loaded = True

    def attach_state(self, state: StreamState) -> None:
        """Attach a pipeline state manager and set IDLE if model already loaded."""
        self.state = state
        if self.model_loaded:
            self.state.set_state(PipelineState.IDLE)

    # ========= Frame caching for video fallback behavior =========
    def _create_fallback_video_frame(self, current_frame: VideoFrame) -> Optional[VideoFrame]:
        """Create a fallback video frame using the last processed frame with current timing."""
        if not self.enable_frame_caching or self._last_processed_video_frame is None:
            return None
            
        # Update timestamp to current frame for proper timing
        fallback_frame = VideoFrame.from_av_video(
            tensor=self._last_processed_video_frame.tensor,
            timestamp=current_frame.timestamp,
            time_base=current_frame.time_base
        )
        return fallback_frame
    

    def reset_frame_cache(self) -> None:
        """Reset cached frames - useful when starting new streams or workflows."""
        self._last_processed_video_frame = None
        logger.debug("Frame cache reset")

    async def process_video_with_fallback(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process video frame with smart fallback to cached frame."""
        try:
            result = await self.process_video_async(frame)
            
            if isinstance(result, VideoFrame):
                # Cache successful result
                if self.enable_frame_caching:
                    self._last_processed_video_frame = result
                return result
            else:
                # Processing returned None - try fallback
                fallback = self._create_fallback_video_frame(frame)
                if fallback is not None:
                    logger.debug("Using cached video frame as fallback")
                    return fallback
                else:
                    logger.debug("No cached video frame available, returning None")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            if self.error_callback:
                try:
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("video_processing_error", e)
                    else:
                        self.error_callback("video_processing_error", e)
                except Exception:
                    pass
            
            # Try fallback on exception
            fallback = self._create_fallback_video_frame(frame)
            if fallback is not None:
                logger.debug("Using cached video frame as fallback after error")
                return fallback
            else:
                logger.debug("No cached video frame available after error, returning None")
                return None
    
    async def process_audio(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Process audio frame without fallback - returns processing result or None."""
        try:
            result = await self.process_audio_async(frame)
            return result if isinstance(result, list) and len(result) > 0 else None
                    
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            if self.error_callback:
                try:
                    if asyncio.iscoroutinefunction(self.error_callback):
                        await self.error_callback("audio_processing_error", e)
                    else:
                        self.error_callback("audio_processing_error", e)
                except Exception:
                    pass
            return None

    # ===== Optional lifecycle hooks; subclasses may override =====
    async def reset_timing(self) -> None:
        """Optional hook to reset timing state (e.g., frame counters) between streams."""
        return

    @abstractmethod
    def load_model(self, *kwargs):
        """
        Load the model.

        This method should be implemented to load any required models or resources.
        It is called automatically during initialization.
        
        Args:
            *kwargs: Additional parameters for model loading
        """
        pass

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

    @abstractmethod
    def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters (optional override).

        Args:
            params: Dictionary of parameters to update
        """
        pass
