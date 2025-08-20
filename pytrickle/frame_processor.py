"""
Frame Processor - Async processing utilities for PyTrickle.
This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.
"""

import logging
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
        **init_kwargs
    ):
        """Initialize the frame processor.
        
        Args:
            error_callback: Optional error callback for processing errors.
                           If None, errors will be logged but not propagated.
            **init_kwargs: Additional kwargs passed to load_model() method
        """
        self.error_callback = error_callback
        self.state: Optional[StreamState] = None
        self.model_loaded: bool = False
        
        try:
            self.load_model(**init_kwargs)
            self.model_loaded = True
            # If a state manager is already attached, mark ready automatically
            if self.state is not None:
                self.state.set_state(PipelineState.IDLE)
        except Exception:
            # If load fails and we have a state manager, mark ERROR
            if self.state is not None:
                self.state.set_state(PipelineState.ERROR)
            raise

    def attach_state(self, state: StreamState) -> None:
        """Attach a pipeline state manager and set IDLE if model already loaded."""
        self.state = state
        if self.model_loaded:
            self.state.set_state(PipelineState.IDLE)

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
