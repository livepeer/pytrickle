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
from .server import TrickleApp, create_app
from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from .tensors import tensor_to_av_frame
from .publisher import TricklePublisher
from .subscriber import TrickleSubscriber

__version__ = "0.1.1"

__all__ = [
    "TrickleClient",
    "TrickleApp",
    "create_app",
    "TrickleProtocol",
    "VideoFrame",
    "AudioFrame", 
    "VideoOutput",
    "AudioOutput",
    "TricklePublisher",
    "TrickleSubscriber",
    "tensor_to_av_frame",
    "ErrorCallback",
] 