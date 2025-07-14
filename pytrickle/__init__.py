"""
Trickle App - A Python package for high-performance video streaming over trickle protocol.

Provides functionality to subscribe to and publish video streams with real-time processing.
"""

from .client import TrickleClient, SimpleTrickleClient
from .server import TrickleApp, create_app
from .protocol import TrickleProtocol
from .frames import VideoFrame, AudioFrame, VideoOutput, AudioOutput
from .tensors import tensor_to_av_frame
from .publisher import TricklePublisher, create_publisher
from .subscriber import TrickleSubscriber, create_subscriber
from .exceptions import (
    TrickleException, 
    TrickleConnectionError, 
    TrickleStreamClosedError, 
    TrickleTimeoutError, 
    TrickleMaxRetriesError,
    ErrorPropagator
)
from .retry_utils import (
    RetryConfig, 
    retry_async, 
    retry_sync,
    NETWORK_RETRY_CONFIG,
    OPTIONAL_CHANNEL_RETRY_CONFIG,
    REQUIRED_CHANNEL_RETRY_CONFIG
)
from .stream_manager import (
    StreamConfig,
    StreamHandler,
    StreamManager,
    create_simple_stream,
    run_stream_until_complete
)

__version__ = "0.1.0"

__all__ = [
    "TrickleClient",
    "SimpleTrickleClient",
    "TrickleApp",
    "create_app",
    "TrickleProtocol",
    "VideoFrame",
    "AudioFrame", 
    "VideoOutput",
    "AudioOutput",
    "TricklePublisher",
    "TrickleSubscriber",
    "create_publisher",
    "create_subscriber",
    "tensor_to_av_frame",
    "TrickleException",
    "TrickleConnectionError",
    "TrickleStreamClosedError",
    "TrickleTimeoutError",
    "TrickleMaxRetriesError",
    "ErrorPropagator",
    "RetryConfig",
    "retry_async",
    "retry_sync",
    "NETWORK_RETRY_CONFIG",
    "OPTIONAL_CHANNEL_RETRY_CONFIG",
    "REQUIRED_CHANNEL_RETRY_CONFIG",
    "StreamConfig",
    "StreamHandler",
    "StreamManager",
    "create_simple_stream",
    "run_stream_until_complete",
] 