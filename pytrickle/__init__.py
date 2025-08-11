"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from typing import Union, Callable, Optional, Coroutine, Any

# Type alias for error callback functions
ErrorCallback = Union[
    Callable[[str, Optional[Exception]], None],
    Callable[[str, Optional[Exception]], Coroutine[Any, Any, None]]
]

from .client import TrickleClient
from .server import TrickleApp
from .protocol import TrickleProtocol
from .frames import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer,
)
from .state import StreamState
from .publisher import TricklePublisher
from .subscriber import TrickleSubscriber
from .health import StreamHealthManager
from .manager import BaseStreamManager, TrickleStreamManager, StreamHandler
from .stream_handler import TrickleStreamHandler
from .register import RegisterCapability

# Async processing utilities
from .async_processor import AsyncFrameProcessor
from .simple_async_processor import SimpleAsyncProcessor

from . import api

__version__ = "0.1.1"

__all__ = [
    "TrickleClient",
    "TrickleApp",
    "TrickleProtocol",
    "VideoFrame",
    "AudioFrame", 
    "VideoOutput",
    "AudioOutput",
    "TricklePublisher",
    "TrickleSubscriber",
    "StreamHealthManager",
    "BaseStreamManager", 
    "TrickleStreamManager",
    "StreamHandler",
    "TrickleStreamHandler",
    "FrameBuffer",
    "StreamState",
    "RegisterCapability",
    "api",
    "ErrorCallback",
    
    # Async processing utilities
    "AsyncFrameProcessor",
    "SimpleAsyncProcessor",
] 