"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from .client import TrickleClient
from .server import StreamServer
from .protocol import TrickleProtocol
from .frames import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer,
    build_loading_overlay_frame,
)
from .state import StreamState
from .base import TrickleComponent, ComponentState, ErrorCallback
from .publisher import TricklePublisher
from .subscriber import TrickleSubscriber
from .manager import BaseStreamManager, TrickleStreamManager, StreamHandler
from .stream_handler import TrickleStreamHandler
from .utils.register import RegisterCapability

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