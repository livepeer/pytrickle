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
from .server import StreamServer
from .protocol import TrickleProtocol
from .frames import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer,
)
from .state import StreamState
from .base import TrickleComponent, ComponentState
from .publisher import TricklePublisher
from .subscriber import TrickleSubscriber
from .health import StreamHealthManager
from .manager import BaseStreamManager, TrickleStreamManager, StreamHandler
from .stream_handler import TrickleStreamHandler
from .utils.register import RegisterCapability

# Processing utilities
from .frame_processor import FrameProcessor
from .stream_processor import StreamProcessor
from .fps_meter import FPSMeter

from . import api

__version__ = "0.1.1"

__all__ = [
    "TrickleClient",
    "StreamServer",
	"StreamProcessor",
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
    "TrickleComponent",
    "ComponentState",
    "RegisterCapability",
    "api",
    "ErrorCallback",
    "FrameProcessor",
    "FPSMeter"
] 