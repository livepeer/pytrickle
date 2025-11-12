"""
Frame Processor - Async processing utilities for PyTrickle.
This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from .frames import VideoFrame, AudioFrame
from .base import ErrorCallback
from .state import StreamState
from .loading_config import LoadingConfig, LoadingMode

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
        loading_config: Optional[LoadingConfig] = None,
        **init_kwargs
    ):
        """Initialize the frame processor.
        
        Args:
            error_callback: Optional error callback for processing errors.
                           If None, errors will be logged but not propagated.
            loading_config: Optional loading configuration for loading overlay/passthrough behavior.
                           If None, no loading gating is applied.
            **init_kwargs: Additional kwargs passed to load_model() method
        """
        self.error_callback = error_callback
        self.state: Optional[StreamState] = None
        self._model_loaded = False
        self._model_load_lock = asyncio.Lock()
        
        # Loading state management
        self.loading_config = loading_config
        self._loading_active: bool = False
        self._frame_counter: int = 0

    def attach_state(self, state: StreamState) -> None:
        """Attach a pipeline state manager."""
        self.state = state

    async def ensure_model_loaded(self, **kwargs):
        """Thread-safe wrapper that ensures model is loaded exactly once."""
        async with self._model_load_lock:
            if not self._model_loaded:
                await self.load_model(**kwargs)
                self._model_loaded = True
                
                # After load_model completes, mark startup complete
                if self.state:
                    self.state.set_startup_complete()
                    logger.debug(f"Model loaded - startup complete for {self.__class__.__name__}")
                else:
                    logger.debug(f"Model loaded for {self.__class__.__name__}")
            else:
                logger.debug(f"Model already loaded for {self.__class__.__name__}")

    @abstractmethod
    async def load_model(self, **kwargs):
        """
        Load the model.

        This method should be implemented to load any required models or resources.
        It is called automatically when needed.
        
        Args:
            **kwargs: Additional parameters for model loading
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
    async def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters (optional override).

        Args:
            params: Dictionary of parameters to update
        """
        pass

    async def on_stream_start(self):
        """
        Called when a stream starts or client connects.
        
        Override this method to perform initialization operations like:
        - Starting background tasks
        - Initializing state
        - Setting up resources
        - Starting timers or loops
        """
        pass

    async def on_stream_stop(self):
        """
        Called when a stream stops or client disconnects.
        
        Override this method to perform cleanup operations like:
        - Cancelling background tasks
        - Resetting internal state
        - Cleaning up resources
        - Stopping timers or loops
        """
        pass
    
    
    def _is_loading_active(self) -> bool:
        """
        Check if loading is currently in progress.
        
        Returns:
            True if loading is active
        """
        if not self.loading_config or not self.loading_config.enabled:
            return False
        return self._loading_active
    
    def set_loading_config(self, config: Optional[LoadingConfig]) -> None:
        """
        Update loading configuration dynamically.
        
        Args:
            config: New loading configuration, or None to disable loading gating
        """
        self.loading_config = config
        logger.debug(f"Loading config updated: {config}")
    
    def _should_show_loading_overlay(self) -> bool:
        """
        Helper to determine if loading overlay should be shown.
        
        Returns:
            True if overlay should be shown, False for passthrough
        """
        if not self._is_loading_active():
            return False
        
        if not self.loading_config:
            return False
        
        return self.loading_config.mode == LoadingMode.OVERLAY
