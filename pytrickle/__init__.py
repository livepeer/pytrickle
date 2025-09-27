"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from typing import Union, Callable, Optional, Coroutine, Any

# Type alias for error callback functions (async only)
ErrorCallback = Callable[[str, Optional[Exception]], Coroutine[Any, Any, None]]

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
from .manager import BaseStreamManager, TrickleStreamManager, StreamHandler
from .stream_handler import TrickleStreamHandler
from .utils.register import RegisterCapability

# Processing utilities
from .frame_processor import FrameProcessor
from .stream_processor import StreamProcessor
from .fps_meter import FPSMeter
from .frame_skipper import FrameSkipConfig


from .decorators import (
    trickle_handler,
    video_handler,
    audio_handler,
    model_loader,
    param_updater,
    on_stream_stop,
)

from . import api

from .version import VERSION as __version__

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
    "FPSMeter",
    "FrameSkipConfig",
    "trickle_handler",
    "video_handler",
    "audio_handler",
    "model_loader",
    "param_updater",
    "on_stream_stop",
    "__version__"
] 