"""
Test utilities for Trickle streaming components.

Provides factories and helpers for creating test doubles and mock objects
to improve testability of streaming components.
"""

import asyncio
from typing import Optional, Callable, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from .client import TrickleClient
from .protocol import TrickleProtocol
from .frame_processor import FrameProcessor
from .lifecycle import ProtocolFactory, ClientFactory


class MockProtocolFactory:
    """Test factory that creates mock protocols."""
    
    def __init__(self, mock_protocol=None):
        self.mock_protocol = mock_protocol or self._create_default_mock()
    
    def _create_default_mock(self):
        """Create a default mock protocol."""
        mock = MagicMock()
        mock.stop = AsyncMock()
        mock.start = AsyncMock()
        mock.emit_monitoring_event = AsyncMock()
        mock.fps_meter = MagicMock()
        mock.fps_meter.get_fps_stats.return_value = {"ingress": 30.0, "egress": 29.5}
        return mock
    
    def create_protocol(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: str = "",
        events_url: str = "",
        data_url: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_framerate: Optional[int] = None,
        publisher_timeout: Optional[float] = None,
        subscriber_timeout: Optional[float] = None,
    ) -> TrickleProtocol:
        """Create a mock protocol."""
        # Store creation parameters for verification
        self.last_creation_params = {
            "subscribe_url": subscribe_url,
            "publish_url": publish_url,
            "control_url": control_url,
            "events_url": events_url,
            "data_url": data_url,
            "width": width,
            "height": height,
            "max_framerate": max_framerate,
            "publisher_timeout": publisher_timeout,
            "subscriber_timeout": subscriber_timeout,
        }
        return self.mock_protocol


class MockClientFactory:
    """Test factory that creates mock clients."""
    
    def __init__(self, mock_client=None):
        self.mock_client = mock_client or self._create_default_mock()
    
    def _create_default_mock(self):
        """Create a default mock client."""
        mock = MagicMock()
        mock.running = False
        mock._client_task = None
        mock._start_call_count = 0
        mock._stop_call_count = 0
        
        # Create an AsyncMock for start that returns immediately but has proper mock attributes
        mock_start = AsyncMock()
        
        async def start_side_effect(request_id="default"):
            mock._start_call_count += 1
            mock.running = True
            mock.request_id = request_id
            
            # Check if we're supposed to raise an exception
            if hasattr(mock_start, '_test_exception') and mock_start._test_exception:
                raise mock_start._test_exception
            
            # For successful cases, simulate a brief running state to allow state updates
            # but don't hang. This prevents immediate cleanup that would reset state.
            if not hasattr(mock_start, '_immediate_return') or not mock_start._immediate_return:
                # Brief pause to allow state to be observed in tests
                await asyncio.sleep(0.001)
            
            return
        
        mock_start.side_effect = start_side_effect
        
        # Create a simple stop method that just updates state
        async def mock_stop():
            mock._stop_call_count += 1
            mock.running = False
            # No need for complex task cancellation since start() returns immediately
        
        mock.start = mock_start
        mock.stop = mock_stop
        mock.publish_data = AsyncMock()
        
        # Mock coordination events
        mock.stop_event = MagicMock()
        mock.stop_event.clear = MagicMock()
        mock.error_event = MagicMock() 
        mock.error_event.clear = MagicMock()
        
        # Mock output queue
        mock.output_queue = MagicMock()
        mock.output_queue.empty.return_value = True
        mock.output_queue.get_nowait = MagicMock()
        
        return mock
    
    def create_client(
        self,
        protocol: TrickleProtocol,
        frame_processor: FrameProcessor,
        control_handler: Optional[Callable] = None,
    ) -> TrickleClient:
        """Create a mock client."""
        # Store creation parameters for verification
        self.last_creation_params = {
            "protocol": protocol,
            "frame_processor": frame_processor,
            "control_handler": control_handler,
        }
        self.mock_client.protocol = protocol
        return self.mock_client


class MockFrameProcessor(FrameProcessor):
    """Simple frame processor for testing."""
    
    def __init__(self, **kwargs):
        self.test_params = {}
        super().__init__(**kwargs)
    
    def load_model(self, **kwargs):
        """Test model loader."""
        pass
    
    async def process_video_async(self, frame):
        """Test video processor."""
        return frame
    
    async def process_audio_async(self, frame):
        """Test audio processor."""
        return [frame]
    
    def update_params(self, params: Dict[str, Any]):
        """Test parameter updater."""
        if params:
            self.test_params.update(params)


def create_test_lifecycle_manager(
    frame_processor: Optional[FrameProcessor] = None,
    mock_protocol: Optional[MagicMock] = None,
    mock_client: Optional[MagicMock] = None,
    **kwargs
):
    """Create a lifecycle manager configured for testing."""
    from .state import StreamState
    
    if frame_processor is None:
        frame_processor = MockFrameProcessor()
    
    state = StreamState()
    
    protocol_factory = MockProtocolFactory(mock_protocol)
    client_factory = MockClientFactory(mock_client)
    
    from .lifecycle import StreamLifecycleManager
    return StreamLifecycleManager(
        frame_processor=frame_processor,
        state=state,
        protocol_factory=protocol_factory,
        client_factory=client_factory,
        **kwargs
    )


def create_test_server_for_endpoints(
    capability_name: str = "test-model",
    pipeline: str = "test-pipeline", 
    version: str = "1.0.0",
    enable_default_routes: bool = True,
):
    """
    Create a StreamServer for testing HTTP endpoints without stream lifecycle issues.
    
    Uses the successful pattern from test_system_endpoints.py - real processor,
    no complex mocking, focuses on HTTP API testing.
    """
    from .stream_processor import _InternalFrameProcessor
    from .server import StreamServer
    
    # Import example functions for a real processor
    try:
        from examples.process_video_example import load_model, process_video, update_params
        
        processor = _InternalFrameProcessor(
            video_processor=process_video,
            audio_processor=None,
            model_loader=load_model,
            param_updater=update_params,
            name=capability_name,
        )
    except ImportError:
        # Fallback to mock processor if examples aren't available
        processor = MockFrameProcessor()
    
    server = StreamServer(
        frame_processor=processor,
        port=0,  # ephemeral port for safety
        capability_name=capability_name,
        pipeline=pipeline,
        version=version,
        enable_default_routes=enable_default_routes,
    )
    
    # Attach state and mark ready (following successful pattern)
    processor.attach_state(server.state)
    server.state.set_startup_complete()
    
    return server


def create_simple_async_mock():
    """
    Create a simple AsyncMock that returns immediately.
    
    Use this instead of complex mock behaviors that simulate long-running operations.
    Follows the successful pattern of focusing on API contracts rather than execution.
    """
    return AsyncMock(return_value=None)


def configure_mock_client_for_success(mock_client_factory):
    """Configure mock client to behave successfully for testing state updates."""
    mock_client_factory.mock_client.start._test_exception = None
    mock_client_factory.mock_client.start._immediate_return = False


def configure_mock_client_for_error(mock_client_factory, error_message="Test client error"):
    """Configure mock client to raise an exception for testing error handling."""
    mock_client_factory.mock_client.start._test_exception = Exception(error_message)
    mock_client_factory.mock_client.start._immediate_return = False


def configure_mock_client_for_immediate_return(mock_client_factory):
    """Configure mock client to return immediately without any delay."""
    mock_client_factory.mock_client.start._test_exception = None
    mock_client_factory.mock_client.start._immediate_return = True
