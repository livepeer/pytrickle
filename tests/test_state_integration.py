"""
Tests for StreamState integration with server components.

Tests state management, transitions, and integration with the simplified server architecture.
Validates that state changes work correctly for monitoring and health reporting.
"""

import pytest
import pytest_asyncio
import asyncio
from aiohttp.test_utils import TestServer, TestClient
from unittest.mock import patch, MagicMock

from pytrickle.server import StreamServer
from pytrickle.state import StreamState, PipelineState
from pytrickle.test_utils import MockFrameProcessor, create_mock_client


def get_stream_route(server, endpoint):
    """Get the full route path for a streaming endpoint."""
    prefix = server.route_prefix.rstrip('/')
    return f"{prefix}/stream/{endpoint}"


@pytest_asyncio.fixture
async def server_with_state():
    """Create a server with state for integration testing."""
    processor = MockFrameProcessor()
    
    server = StreamServer(
        frame_processor=processor,
        port=0,
        capability_name="state-test",
        pipeline="state-test-pipeline",
        version="1.0.0",
    )
    
    processor.attach_state(server.state)
    yield server, processor


class TestStreamStateCore:
    """Test core StreamState functionality."""

    def test_initial_state(self):
        """Test StreamState initial values."""
        state = StreamState()
        
        assert state.state == PipelineState.LOADING
        assert not state.running
        assert not state.pipeline_ready
        assert not state.active_client
        assert state.active_streams == 0
        assert not state.startup_complete
        assert not state.is_error()

    def test_state_transitions(self):
        """Test PipelineState transitions."""
        state = StreamState()
        
        # LOADING → IDLE
        state.set_startup_complete()
        state.set_state(PipelineState.IDLE)
        data = state.get_state()
        assert state.state == PipelineState.IDLE
        assert data["status"] == "IDLE"
        
        # IDLE with active streams → status becomes OK (internal state stays IDLE)
        state.update_active_streams(1)
        state.set_active_client(True)
        data = state.get_state()
        assert state.state == PipelineState.IDLE  # Internal state doesn't auto-transition
        assert data["status"] == "OK"  # But status reflects activity
        
        # When streams stop → status becomes IDLE
        state.update_active_streams(0)
        state.set_active_client(False)
        data = state.get_state()
        assert state.state == PipelineState.IDLE  # Internal state stays IDLE
        assert data["status"] == "IDLE"  # Status reflects no activity
        
        # IDLE → ERROR
        state.set_error("Test error")
        assert state.state == PipelineState.ERROR
        assert state.is_error()

    def test_error_state_management(self):
        """Test error state setting and clearing."""
        state = StreamState()
        
        # Set error
        state.set_error("Test error message")
        assert state.is_error()
        assert state.state == PipelineState.ERROR
        assert state.error_event.is_set()
        
        # Clear error
        state.clear_error()
        assert not state.is_error()
        assert state.state == PipelineState.LOADING  # Returns to LOADING (as per implementation)
        assert not state.error_event.is_set()

    def test_active_client_tracking(self):
        """Test active client state tracking."""
        state = StreamState()
        
        assert not state.active_client
        
        state.set_active_client(True)
        assert state.active_client
        
        state.set_active_client(False)
        assert not state.active_client

    def test_active_streams_tracking(self):
        """Test active streams counting."""
        state = StreamState()
        
        assert state.active_streams == 0
        
        state.update_active_streams(3)
        assert state.active_streams == 3
        
        state.update_active_streams(-1)  # Should clamp to 0
        assert state.active_streams == 0

    def test_startup_completion(self):
        """Test startup completion tracking."""
        state = StreamState()
        
        assert not state.startup_complete
        assert not state.pipeline_ready
        
        state.set_startup_complete()
        assert state.startup_complete
        assert state.pipeline_ready

    def test_get_state_method(self):
        """Test the get_state method returns proper format."""
        state = StreamState()
        
        # Initial state
        data = state.get_state()
        assert data["status"] == "LOADING"
        assert "pipeline_ready" in data
        assert not data["pipeline_ready"]
        
        # Ready with active streams (auto-transitions to OK)
        state.set_startup_complete()
        state.set_state(PipelineState.IDLE)
        state.update_active_streams(1)
        state.set_active_client(True)
        data = state.get_state()
        assert data["status"] == "OK"  # Auto-transitioned to OK
        
        # Error state
        state.set_error("Test error")
        data = state.get_state()
        assert data["status"] == "ERROR"

    def test_get_pipeline_state_method(self):
        """Test the get_pipeline_state method."""
        state = StreamState()
        
        # Initial state
        data = state.get_pipeline_state()
        assert data["status"] == "LOADING"
        assert not data["startup_complete"]
        assert data["active_streams"] == 0
        
        # After startup complete
        state.set_startup_complete()
        data = state.get_pipeline_state()
        assert data["status"] == "IDLE"
        assert data["startup_complete"]
        
        # With active streams (auto-transitions to OK)
        state.update_active_streams(2)
        data = state.get_pipeline_state()
        assert data["status"] == "OK"  # Auto-transitioned to OK
        assert data["active_streams"] == 2


class TestFrameProcessorStateIntegration:
    """Test FrameProcessor integration with StreamState."""

    def test_frame_processor_state_attachment(self):
        """Test frame processor state attachment."""
        processor = MockFrameProcessor()
        state = StreamState()
        
        # Initially has no state attached
        assert hasattr(processor, 'state')
        assert processor.state is None
        
        # Attach new state
        processor.attach_state(state)
        assert processor.state is state

    @pytest.mark.asyncio
    async def test_frame_processor_model_loading_success(self):
        """Test frame processor model loading with state updates."""
        processor = MockFrameProcessor()
        state = StreamState()
        processor.attach_state(state)
        
        # Model loading should work without errors
        await processor.load_model()
        # MockFrameProcessor doesn't change state, but real ones would

    @pytest.mark.asyncio
    async def test_frame_processor_model_loading_failure(self):
        """Test frame processor handles model loading failures."""
        processor = MockFrameProcessor()
        state = StreamState()
        processor.attach_state(state)
        
        # Mock a failing load_model
        async def failing_load_model(**kwargs):
            raise Exception("Model loading failed")
        
        processor.load_model = failing_load_model
        
        # Should handle the exception gracefully
        try:
            await processor.load_model()
            assert False, "Expected exception"
        except Exception as e:
            assert "Model loading failed" in str(e)

    @pytest.mark.asyncio
    async def test_internal_frame_processor_integration(self):
        """Test _InternalFrameProcessor integration with state."""
        from pytrickle.stream_processor import _InternalFrameProcessor
        from examples.process_video_example import load_model, process_video, update_params
        
        processor = _InternalFrameProcessor(
            video_processor=process_video,
            model_loader=load_model,
            param_updater=update_params,
            name="test-processor",
        )
        
        state = StreamState()
        processor.attach_state(state)
        
        # Test model loading
        await processor.load_model()
        
        # Test parameter updates
        await processor.update_params({"intensity": 0.8})


class TestServerStateIntegration:
    """Test StreamServer integration with StreamState."""

    @pytest.mark.asyncio
    async def test_server_state_initialization(self, server_with_state):
        """Test server state initialization."""
        server, processor = server_with_state
        
        # Server should have state
        assert server.state is not None
        assert isinstance(server.state, StreamState)
        
        # Processor should have state attached
        assert processor.state is server.state

    @pytest.mark.asyncio
    async def test_server_health_endpoint_state_integration(self, server_with_state):
        """Test health endpoint reflects server state."""
        server, processor = server_with_state
        
        from aiohttp.test_utils import TestServer, TestClient
        
        app = server.get_app()
        test_server_instance = TestServer(app)
        
        async with test_server_instance:
            client = TestClient(test_server_instance)
            async with client:
                # Test initial state
                resp = await client.get("/health")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"status": "LOADING"}
                
                # Complete startup
                server.state.set_startup_complete()
                
                resp = await client.get("/health")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"status": "IDLE"}
                
                # Set error state
                server.state.set_error("Test error")
                
                resp = await client.get("/health")
                assert resp.status == 200  # Always HTTP 200 per specification
                data = await resp.json()
                assert data == {"status": "ERROR"}

    @pytest.mark.asyncio
    async def test_server_stream_state_validation(self, server_with_state):
        """Test that server properly validates and updates state during stream operations."""
        server, processor = server_with_state
        
        from aiohttp.test_utils import TestServer, TestClient
        
        app = server.get_app()
        test_server_instance = TestServer(app)
        
        async with test_server_instance:
            client = TestClient(test_server_instance)
            async with client:
                # Initial state validation
                resp = await client.get(get_stream_route(server, "status"))
                assert resp.status == 200
                data = await resp.json()
                assert data["client_active"] is False
                assert data["active_streams"] == 0
                
                # Test state changes through direct manipulation (simulating successful stream start)
                server.state.set_active_client(True)
                server.state.update_active_streams(1)
                
                # Mock client to simulate active stream
                mock_client = create_mock_client()
                server.current_client = mock_client
                
                # Validate state changes are reflected
                resp = await client.get(get_stream_route(server, "status"))
                assert resp.status == 200
                data = await resp.json()
                assert data["client_active"] is True
                assert data["active_streams"] == 1
                
                # Test state cleanup
                server.current_client = None
                server.state.set_active_client(False)
                server.state.update_active_streams(0)
                
                resp = await client.get(get_stream_route(server, "status"))
                assert resp.status == 200
                data = await resp.json()
                assert data["client_active"] is False
                assert data["active_streams"] == 0


class TestStateTransitionEdgeCases:
    """Test edge cases in state transitions."""

    def test_invalid_state_transition(self):
        """Test handling of invalid state transitions."""
        state = StreamState()
        
        # Test setting invalid state (should work since it's just an enum)
        state.set_state(PipelineState.ERROR)
        assert state.state == PipelineState.ERROR

    def test_concurrent_state_updates(self):
        """Test concurrent state updates don't cause issues."""
        state = StreamState()
        
        # Simulate concurrent updates
        state.update_active_streams(5)
        state.set_active_client(True)
        state.set_startup_complete()
        
        # All should be applied
        assert state.active_streams == 5
        assert state.active_client
        assert state.startup_complete

    def test_state_consistency_after_errors(self):
        """Test state remains consistent after error conditions."""
        state = StreamState()
        
        # Set up normal state
        state.set_startup_complete()
        state.set_state(PipelineState.IDLE)
        state.update_active_streams(2)
        state.set_active_client(True)
        
        # Introduce error
        state.set_error("Test error")
        assert state.is_error()
        
        # Clear error - should return to consistent state
        state.clear_error()
        assert not state.is_error()
        assert state.state == PipelineState.LOADING  # Returns to LOADING (as per implementation)
        # After clearing error, need to set back to IDLE manually
        state.set_state(PipelineState.IDLE)
        assert state.pipeline_ready
        assert state.active_streams == 2  # Stream count preserved
        assert state.active_client  # Client state preserved

    def test_event_coordination(self):
        """Test asyncio event coordination."""
        state = StreamState()
        
        # Test error event
        assert not state.error_event.is_set()
        state.set_error("Test")
        assert state.error_event.is_set()
        state.clear_error()
        assert not state.error_event.is_set()


class TestStateIntegrationWithStreamProcessor:
    """Test state integration with stream processing components."""

    @pytest.mark.asyncio
    async def test_stream_processor_state_integration(self):
        """Test StreamProcessor integration with state."""
        from pytrickle.stream_processor import _InternalFrameProcessor
        from examples.process_video_example import load_model, process_video, update_params
        
        processor = _InternalFrameProcessor(
            video_processor=process_video,
            model_loader=load_model,
            param_updater=update_params,
            name="integration-test",
        )
        
        state = StreamState()
        processor.attach_state(state)
        
        # Test that processor can work with state
        assert processor.state is state
        
        # Test model loading
        await processor.load_model()
        
        # Test parameter updates
        test_params = {"intensity": 0.7, "effect": "test"}
        await processor.update_params(test_params)
        
        # Verify processor handled params (internal behavior may vary)
        # The important thing is no exceptions were raised
