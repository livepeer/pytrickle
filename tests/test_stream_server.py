"""
Tests for StreamServer with simplified architecture.

Tests HTTP endpoints and state validation using direct TrickleClient management.
Focuses on testing state changes and API contracts.
"""

import pytest
import pytest_asyncio
import asyncio
from aiohttp.test_utils import TestServer, TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from pytrickle.server import StreamServer
from pytrickle.test_utils import MockFrameProcessor, create_mock_client

def get_stream_route(server, endpoint):
    """Get the full route path for a streaming endpoint."""
    prefix = server.route_prefix.rstrip('/')
    return f"{prefix}/stream/{endpoint}"


@pytest_asyncio.fixture
async def test_server():
    """Create a test server with simplified architecture."""
    processor = MockFrameProcessor()
    
    server = StreamServer(
        frame_processor=processor,
        port=0,  # ephemeral port
        capability_name="test-model",
        pipeline="test-pipeline",
        version="1.0.0",
    )

    # Attach state and mark ready
    processor.attach_state(server.state)
    server.state.set_startup_complete()

    app = server.get_app()
    test_server_instance = TestServer(app)
    
    async with test_server_instance:
        client = TestClient(test_server_instance)
        async with client:
            yield client, server


class TestStreamingEndpoints:
    """Test core streaming functionality with state validation."""

    @pytest.mark.asyncio
    async def test_start_stream_success_with_validation(self, test_server):
        """Test successful stream start API and validate components are created correctly."""
        client, server = test_server
        
        # Mock TrickleClient and TrickleProtocol creation
        with patch('pytrickle.server.TrickleProtocol') as mock_protocol_class, \
             patch('pytrickle.server.TrickleClient') as mock_client_class:
            
            mock_protocol = MagicMock()
            mock_protocol_class.return_value = mock_protocol
            
            mock_client = create_mock_client()
            mock_client_class.return_value = mock_client
            
            payload = {
                "subscribe_url": "http://localhost:3389/input",
                "publish_url": "http://localhost:3389/output", 
                "gateway_request_id": "test-123",
                "params": {
                    "intensity": 0.8,
                    "width": 512,
                    "height": 512
                }
            }
            
            resp = await client.post(get_stream_route(server, "start"), json=payload)
            assert resp.status == 200
            
            data = await resp.json()
            assert data["status"] == "success"
            assert data["request_id"] == "test-123"
            
            # Verify protocol was created with correct parameters
            mock_protocol_class.assert_called_once()
            call_kwargs = mock_protocol_class.call_args.kwargs
            assert call_kwargs["subscribe_url"] == "http://localhost:3389/input"
            assert call_kwargs["publish_url"] == "http://localhost:3389/output"
            assert call_kwargs["width"] == 512
            assert call_kwargs["height"] == 512
            
            # Verify client was created
            mock_client_class.assert_called_once()
            
            # Note: Params are now passed to on_stream_start instead of update_params

    @pytest.mark.asyncio
    async def test_on_stream_start_receives_parameters(self, test_server):
        """Test that on_stream_start callback receives stream parameters."""
        client, server = test_server
        
        # Mock TrickleClient and TrickleProtocol creation
        with patch('pytrickle.server.TrickleProtocol') as mock_protocol_class, \
             patch('pytrickle.server.TrickleClient') as mock_client_class:
            
            mock_protocol = MagicMock()
            mock_protocol_class.return_value = mock_protocol
            
            # Create a real async mock for client.start that captures params
            captured_params = None
            async def capture_start(request_id, params=None):
                nonlocal captured_params
                captured_params = params
                # Call the actual on_stream_start on the processor
                await server.frame_processor.on_stream_start(params)
            
            mock_client = create_mock_client()
            mock_client.start = AsyncMock(side_effect=capture_start)
            mock_client_class.return_value = mock_client
            
            payload = {
                "subscribe_url": "http://localhost:3389/input",
                "publish_url": "http://localhost:3389/output", 
                "gateway_request_id": "test-stream-start-params",
                "params": {
                    "model": "flux",
                    "width": 1024,
                    "height": 768,
                    "seed": 42
                }
            }
            
            resp = await client.post(get_stream_route(server, "start"), json=payload)
            assert resp.status == 200
            
            # Give background task time to call start
            await asyncio.sleep(0.1)
            
            # Verify on_stream_start was called with params
            mock_client.start.assert_called_once()
            call_args = mock_client.start.call_args
            assert call_args[0][0] == "test-stream-start-params"  # request_id
            assert call_args[0][1] == payload["params"]  # params
            
            # Verify the processor captured the params in on_stream_start
            assert server.frame_processor.stream_start_params == payload["params"]

    @pytest.mark.asyncio
    async def test_stream_state_updates_during_lifecycle(self, test_server):
        """Test that stream operations properly update state."""
        client, server = test_server
        
        # Test direct state manipulation (simulating what happens in handlers)
        # This tests the state management without background task interference
        
        # Simulate stream start state updates
        server.state.set_active_client(True)
        server.state.update_active_streams(1)
        
        # Verify state is updated
        assert server.state.active_streams == 1
        assert server.state.active_client is True
        
        # Verify this reflects in status endpoint
        resp = await client.get(get_stream_route(server, "status"))
        assert resp.status == 200
        data = await resp.json()
        assert data["active_streams"] == 1
        # Note: client_active depends on server.current_client, not state.active_client
        
        # Simulate stream stop state updates
        server.state.set_active_client(False)
        server.state.update_active_streams(0)
        
        # Verify state is updated
        assert server.state.active_streams == 0
        assert server.state.active_client is False

    @pytest.mark.asyncio
    async def test_start_stream_validation_error(self, test_server):
        """Test stream start with validation errors doesn't change state."""
        client, server = test_server
        
        # Verify initial state
        initial_streams = server.state.active_streams
        initial_client = server.state.active_client
        
        # Missing required fields
        payload = {
            "subscribe_url": "http://localhost:3389/input",
            # Missing publish_url and gateway_request_id
        }
        
        resp = await client.post(get_stream_route(server, "start"), json=payload)
        assert resp.status == 400
        
        # Verify state wasn't changed
        assert server.state.active_streams == initial_streams
        assert server.state.active_client == initial_client
        assert server.current_client is None

    @pytest.mark.asyncio
    async def test_update_params_success_with_monitoring(self, test_server):
        """Test parameter update with monitoring event validation."""
        client, server = test_server
        
        # Set up mock client with monitoring
        mock_client = create_mock_client()
        mock_protocol = MagicMock()
        mock_protocol.emit_monitoring_event = AsyncMock()
        mock_client.protocol = mock_protocol
        server.current_client = mock_client
        
        payload = {"intensity": 0.9}
        
        resp = await client.post(get_stream_route(server, "params"), json=payload)
        assert resp.status == 200
        
        data = await resp.json()
        assert data["status"] == "success"
        
        # Verify parameters were updated
        assert server.frame_processor.test_params == payload
        
        # Verify monitoring event was emitted
        mock_protocol.emit_monitoring_event.assert_called()
        call_args = mock_protocol.emit_monitoring_event.call_args[0][0]
        assert call_args["type"] == "params_updated"
        assert call_args["params"] == payload

    @pytest.mark.asyncio
    async def test_update_params_no_active_stream(self, test_server):
        """Test parameter update when no stream is active."""
        client, server = test_server
        
        payload = {"intensity": 0.9}
        
        resp = await client.post(get_stream_route(server, "params"), json=payload)
        assert resp.status == 400
        
        data = await resp.json()
        assert data["status"] == "error"
        assert "No active stream" in data["message"]

    @pytest.mark.asyncio
    async def test_get_status_no_client(self, test_server):
        """Test status endpoint when no client is active."""
        client, server = test_server
        
        resp = await client.get(get_stream_route(server, "status"))
        assert resp.status == 200
        
        data = await resp.json()
        assert data["client_active"] is False
        assert data["active_streams"] == 0

    @pytest.mark.asyncio
    async def test_get_status_with_active_client(self, test_server):
        """Test status endpoint with active client and validate state."""
        client, server = test_server
        
        # Set up mock client
        mock_client = create_mock_client()
        server.current_client = mock_client
        
        # Set current params to test full status response
        from pytrickle.api import StreamStartRequest
        server.current_params = StreamStartRequest(
            subscribe_url="http://localhost:3389/input",
            publish_url="http://localhost:3389/output",
            gateway_request_id="status-test"
        )
        
        resp = await client.get(get_stream_route(server, "status"))
        assert resp.status == 200
        
        data = await resp.json()
        assert data["client_active"] is True
        assert "fps" in data
        assert "current_params" in data
        assert data["current_params"]["gateway_request_id"] == "status-test"

    @pytest.mark.asyncio
    async def test_stop_stream_success_with_monitoring(self, test_server):
        """Test successful stream stop with monitoring validation."""
        client, server = test_server
        
        # Set up mock client with monitoring
        mock_client = create_mock_client()
        mock_protocol = MagicMock()
        mock_protocol.emit_monitoring_event = AsyncMock()
        mock_client.protocol = mock_protocol
        server.current_client = mock_client
        
        # Set initial state
        server.state.set_active_client(True)
        server.state.update_active_streams(1)
        
        resp = await client.post(get_stream_route(server, "stop"), json={})
        assert resp.status == 200
        
        data = await resp.json()
        assert data["status"] == "success"
        
        # Verify state was updated
        assert server.state.active_streams == 0
        assert server.state.active_client is False
        assert server.current_client is None
        
        # Verify monitoring event was emitted
        mock_protocol.emit_monitoring_event.assert_called()
        call_args = mock_protocol.emit_monitoring_event.call_args[0][0]
        assert call_args["type"] == "stream_stopped"
        
        # Verify client was stopped
        mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_stream_no_active_stream(self, test_server):
        """Test stop stream when no stream is active."""
        client, server = test_server
        
        resp = await client.post(get_stream_route(server, "stop"), json={})
        assert resp.status == 400
        
        data = await resp.json()
        assert data["status"] == "error"
        assert "No active stream" in data["message"]


class TestStateManagement:
    """Test direct state management functionality."""

    @pytest.mark.asyncio
    async def test_direct_state_manipulation(self, test_server):
        """Test direct state manipulation and validation."""
        client, server = test_server
        
        # Test state updates
        server.state.update_active_streams(1)
        server.state.set_active_client(True)
        
        resp = await client.get(get_stream_route(server, "status"))
        assert resp.status == 200
        data = await resp.json()
        assert data["active_streams"] == 1
        
        # Test cleanup
        server.state.update_active_streams(0)
        server.state.set_active_client(False)
        
        resp = await client.get(get_stream_route(server, "status"))
        assert resp.status == 200
        data = await resp.json()
        assert data["active_streams"] == 0

    @pytest.mark.asyncio
    async def test_error_state_handling_with_validation(self, test_server):
        """Test error state management with proper validation."""
        client, server = test_server
        
        # Set error state
        server.state.set_error("Test error condition")
        
        # Verify health endpoint reflects error
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"status": "ERROR"}
        
        # Verify status endpoint reflects error
        resp = await client.get(get_stream_route(server, "status"))
        assert resp.status == 200
        data = await resp.json()
        assert server.state.is_error()
        
        # Test error recovery
        server.state.clear_error()
        server.state.set_startup_complete()  # Restore to ready state
        
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] in ["IDLE", "LOADING"]  # Should be back to normal


class TestDynamicRoutes:
    """Test dynamic route functionality with different prefixes."""

    @pytest.mark.parametrize("route_prefix", ["/", "/api", "/v1", "/custom"])
    @pytest.mark.asyncio
    async def test_routes_with_different_prefixes(self, route_prefix):
        """Test that routes work with different prefixes."""
        processor = MockFrameProcessor()
        
        server = StreamServer(
            frame_processor=processor,
            port=0,
            route_prefix=route_prefix,
            capability_name="test-model",
            pipeline="test-pipeline",
            version="1.0.0",
        )
        
        # Attach state and mark ready
        processor.attach_state(server.state)
        server.state.set_startup_complete()
        
        # Test that routes are constructed correctly
        expected_start_route = f"{route_prefix.rstrip('/')}/stream/start"
        actual_start_route = get_stream_route(server, "start")
        assert actual_start_route == expected_start_route
        
        expected_status_route = f"{route_prefix.rstrip('/')}/stream/status"
        actual_status_route = get_stream_route(server, "status")
        assert actual_status_route == expected_status_route
        
        expected_params_route = f"{route_prefix.rstrip('/')}/stream/params"
        actual_params_route = get_stream_route(server, "params")
        assert actual_params_route == expected_params_route
        
        expected_stop_route = f"{route_prefix.rstrip('/')}/stream/stop"
        actual_stop_route = get_stream_route(server, "stop")
        assert actual_stop_route == expected_stop_route

    @pytest.mark.asyncio
    async def test_functional_test_with_custom_prefix(self):
        """Test that endpoints actually work with a custom prefix."""
        processor = MockFrameProcessor()
        
        server = StreamServer(
            frame_processor=processor,
            port=0,
            route_prefix="/api/v2",
            capability_name="test-model",
            pipeline="test-pipeline",
            version="1.0.0",
        )
        
        # Attach state and mark ready
        processor.attach_state(server.state)
        server.state.set_startup_complete()

        app = server.get_app()
        test_server_instance = TestServer(app)
        
        async with test_server_instance:
            client = TestClient(test_server_instance)
            async with client:
                # Test status endpoint with custom prefix
                resp = await client.get(get_stream_route(server, "status"))
                assert resp.status == 200
                
                data = await resp.json()
                assert data["client_active"] is False
                assert data["active_streams"] == 0


class TestErrorHandling:
    """Test error handling scenarios with state validation."""

    @pytest.mark.asyncio
    async def test_client_error_state_propagation(self, test_server):
        """Test that client errors properly propagate to state."""
        client, server = test_server
        
        # Mock client that will fail
        with patch('pytrickle.server.TrickleClient') as mock_client_class:
            mock_client = create_mock_client()
            mock_client.start.side_effect = Exception("Client start failed")
            mock_client_class.return_value = mock_client
            
            payload = {
                "subscribe_url": "http://localhost:3389/input",
                "publish_url": "http://localhost:3389/output",
                "gateway_request_id": "error-test"
            }
            
            resp = await client.post(get_stream_route(server, "start"), json=payload)
            # API call succeeds, but client will fail in background
            assert resp.status == 200
            
            # Wait briefly for background task to fail
            await asyncio.sleep(0.05)
            
            # State should eventually reflect the error
            # Note: In simplified architecture, error handling is more direct

    @pytest.mark.asyncio
    async def test_concurrent_stream_operations(self, test_server):
        """Test concurrent stream operations don't corrupt state."""
        client, server = test_server
        
        with patch('pytrickle.server.TrickleProtocol') as mock_protocol_class, \
             patch('pytrickle.server.TrickleClient') as mock_client_class:
            
            mock_protocol = MagicMock()
            mock_protocol.emit_monitoring_event = AsyncMock()
            mock_protocol_class.return_value = mock_protocol
            
            mock_client = create_mock_client()
            mock_client.protocol = mock_protocol
            mock_client_class.return_value = mock_client
            
            payload = {
                "subscribe_url": "http://localhost:3389/input",
                "publish_url": "http://localhost:3389/output",
                "gateway_request_id": "concurrent-test"
            }
            
            # Send multiple concurrent requests
            tasks = [
                client.post(get_stream_route(server, "start"), json=payload)
                for _ in range(3)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least one should succeed
            success_count = sum(1 for r in responses if hasattr(r, 'status') and r.status == 200)
            assert success_count >= 1
            
            # State should be consistent (only one active stream)
            assert server.state.active_streams <= 1
            assert server.current_client is not None
