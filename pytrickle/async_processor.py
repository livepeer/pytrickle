"""
Async Frame Processor - Async processing utilities for PyTrickle.

This module provides base classes and utilities for async frame processing,
making it easy to integrate AI models and async pipelines with PyTrickle.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List
from .frames import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)

class AsyncFrameProcessor(ABC):
    """
    Base class for async frame processors.
    
    This class provides native async frame processing for PyTrickle. It handles:
    - Async processing queue management
    - Error handling and recovery
    - Lazy processing start (only begins when stream starts)
    - Automatic reset to idle state when streams end
    - Direct integration with TrickleClient and TrickleApp
    
    Lifecycle:
    1. Call start() to initialize the processor
    2. Processing begins automatically when streams start
    3. Processing stops automatically when streams end
    4. Processor returns to idle state ready for the next stream
    
    Usage patterns:
    
    # HTTP server with TrickleApp (recommended)
    processor = MyProcessor()
    await processor.start()
    app = TrickleApp(frame_processor=processor, port=8080)
    await app.run_forever()
    
    # Direct client usage (advanced)
    protocol = TrickleProtocol(subscribe_url="...", publish_url="...")
    client = TrickleClient(protocol=protocol, frame_processor=processor)
    await client.start("request_id")
    
    Subclass this to implement your async AI processing logic.
    """
    
    def __init__(
        self, 
        error_callback: Optional[Callable[[Exception], None]] = None
    ):
        """Initialize the async frame processor."""
        self.error_callback = error_callback
        
        # Processing state
        self.is_started = False
        self.is_processing = False  # Tracks if actively processing frames
        self.frame_count = 0
        
        logger = logging.getLogger(self.__class__.__name__)
    
    async def start(self):
        """Initialize the async processor."""
        if self.is_started:
            return
        
        logger.info(f"Initializing {self.__class__.__name__}")
        self.is_started = True
        self.frame_count = 0
        self.is_processing = False
        
        # Call optional subclass initialization
        await self.initialize()
    
    async def start_processing(self):
        """Start active frame processing (called when stream begins)."""
        if not self.is_started:
            await self.start()
        
        if self.is_processing:
            return
        
        logger.info(f"Starting frame processing for {self.__class__.__name__}")
        self.is_processing = True
        self.frame_count = 0
    
    async def stop_processing(self):
        """Stop active frame processing (called when stream ends)."""
        if not self.is_processing:
            return
        
        logger.info(f"Stopping frame processing for {self.__class__.__name__}")
        self.is_processing = False
        self.frame_count = 0
        
        logger.info(f"Frame processing stopped for {self.__class__.__name__}")

    async def stop(self):
        """Stop the async processor completely."""
        if not self.is_started:
            return
        
        logger.info(f"Stopping {self.__class__.__name__}")
        
        # Stop processing first
        await self.stop_processing()
        
        self.is_started = False
        
        logger.info(f"{self.__class__.__name__} stopped")
    
    # Optional hook methods for subclasses
    
    async def initialize(self):
        """
        Optional initialization hook called during start().
        
        Override this method to perform any async setup, model loading,
        warmup, or other initialization tasks.
        """
        pass
    
    # Utility methods and properties
    
    @property
    def status(self) -> str:
        """Get the current processor status."""
        if not self.is_started:
            return "stopped"
        elif self.is_processing:
            return "processing"
        else:
            return "ready"
    
    def get_frame_count(self) -> int:
        """Get the current frame count."""
        return self.frame_count
    
    def increment_frame_count(self) -> int:
        """Increment and return the frame count."""
        self.frame_count += 1
        return self.frame_count
    
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