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
from . import ErrorCallback
from .state import StreamState
from .warmup_config import WarmupConfig, WarmupMode

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
        warmup_config: Optional[WarmupConfig] = None,
        **init_kwargs
    ):
        """Initialize the frame processor.
        
        Args:
            error_callback: Optional error callback for processing errors.
                           If None, errors will be logged but not propagated.
            warmup_config: Optional warmup configuration for loading overlay/passthrough behavior.
                          If None, no warmup gating is applied.
            **init_kwargs: Additional kwargs passed to load_model() method
        """
        self.error_callback = error_callback
        self.state: Optional[StreamState] = None
        self._model_loaded = False
        self._model_load_lock = asyncio.Lock()
        
        # Warmup/loading state management
        self.warmup_config = warmup_config
        self._loading_active: bool = False
        self._warmup_done: asyncio.Event = asyncio.Event()
        self._warmup_done.set()  # Initially not in warmup state
        self._frame_counter: int = 0
        self._warmup_task: Optional[asyncio.Task] = None

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
    
    def _start_warmup_sequence(self, warmup_coro) -> None:
        """
        Start (or restart) the warmup routine with loading overlay management.
        
        Args:
            warmup_coro: Coroutine that performs the actual warmup work
        """
        try:
            # Cancel any existing warmup task
            if self._warmup_task and not self._warmup_task.done():
                self._warmup_task.cancel()
                try:
                    # Don't await here, just cancel and move on
                    pass
                except Exception:
                    logger.debug("Previous warmup task cleanup error", exc_info=True)
        except Exception:
            logger.debug("Error cancelling prior warmup task", exc_info=True)
        
        # Reset state for new warmup
        self._frame_counter = 0
        self._loading_active = True
        self._warmup_done.clear()
        
        logger.info("Starting warmup sequence (loading_active=True, warmup_done=False)")
        
        async def _warmup_and_finish():
            try:
                await warmup_coro
            except Exception:
                logger.debug("Warmup failed while running warmup sequence", exc_info=True)
            finally:
                self._loading_active = False
                self._warmup_done.set()
                logger.info("Warmup sequence finished (loading_active=False, warmup_done=True)")
                
                # Don't manage state transitions here - let the wrapper handle it
                # This allows both FrameProcessor subclasses and plain functions to work
                # For plain functions: _InternalFrameProcessor will handle state
                # For FrameProcessor subclasses: they can manage their own state if needed
        
        # Create and store the warmup task
        self._warmup_task = asyncio.create_task(_warmup_and_finish())
    
    def _is_warmup_active(self) -> bool:
        """
        Check if warmup is currently in progress.
        
        Returns:
            True if warmup is active and not yet complete
        """
        if not self.warmup_config or not self.warmup_config.enabled:
            return False
        return self._loading_active and not self._warmup_done.is_set()
    
    def set_warmup_config(self, config: Optional[WarmupConfig]) -> None:
        """
        Update warmup configuration dynamically.
        
        Args:
            config: New warmup configuration, or None to disable warmup gating
        """
        self.warmup_config = config
        logger.debug(f"Warmup config updated: {config}")
    
    def _should_show_loading_overlay(self) -> bool:
        """
        Helper to determine if loading overlay should be shown.
        
        Returns:
            True if overlay should be shown, False for passthrough
        """
        if not self._is_warmup_active():
            return False
        
        if not self.warmup_config:
            return False
        
        return self.warmup_config.mode == WarmupMode.OVERLAY
