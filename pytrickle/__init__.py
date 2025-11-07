"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from typing import Callable, Optional, Coroutine, Any

# Type alias for error callback functions (async only)
ErrorCallback = Callable[[str, Optional[Exception]], Coroutine[Any, Any, None]]

from .base import TrickleComponent, ComponentState  # noqa: E402
from .client import TrickleClient  # noqa: E402
from .server import StreamServer  # noqa: E402
from .protocol import TrickleProtocol  # noqa: E402
from .frames import (  # noqa: E402
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer,
    build_loading_overlay_frame,
)  # noqa: E402
from .state import StreamState  # noqa: E402
from .publisher import TricklePublisher  # noqa: E402
from .subscriber import TrickleSubscriber  # noqa: E402
from .manager import BaseStreamManager, TrickleStreamManager, StreamHandler  # noqa: E402
from .stream_handler import TrickleStreamHandler  # noqa: E402
from .utils.register import RegisterCapability  # noqa: E402

# Processing utilities
from .frame_processor import FrameProcessor  # noqa: E402
from .stream_processor import StreamProcessor  # noqa: E402
from .fps_meter import FPSMeter  # noqa: E402
from .frame_skipper import FrameSkipConfig  # noqa: E402
from .warmup_config import WarmupConfig, WarmupMode  # noqa: E402

from . import api  # noqa: E402

from .version import __version__  # noqa: E402

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
    "WarmupConfig",
    "WarmupMode",
    "build_loading_overlay_frame",
    "__version__"
] 