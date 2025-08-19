"""
Tests for all system endpoints in StreamServer.

Tests the /health, /version, /hardware/info, and /hardware/stats endpoints
using the example processor to ensure proper integration and response formats.
"""

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer, TestClient
from unittest.mock import patch, MagicMock

from pytrickle.stream_processor import _InternalFrameProcessor
from pytrickle.server import StreamServer
from pytrickle.state import PipelineState

# Import example processor functions
from examples.process_video_example import load_model, process_video, update_params


@pytest_asyncio.fixture
async def test_server():
    """Create a test server with the example processor."""
    processor = _InternalFrameProcessor(
        video_processor=process_video,
        audio_processor=None,
        model_loader=load_model,
        param_updater=update_params,
        name="trickle-stream-example",
    )

    server = StreamServer(
        frame_processor=processor,
        port=0,  # ephemeral port for safety
        capability_name="trickle-stream-example",
        pipeline="byoc",
        version="0.0.1",
        enable_default_routes=True,
    )

    # Attach state to processor and mark as ready
    processor.attach_state(server.state)
    server.state.set_startup_complete()

    app = server.get_app()
    test_server_instance = TestServer(app)
    
    async with test_server_instance:
        client = TestClient(test_server_instance)
        async with client:
            yield client, server


@pytest.mark.asyncio
async def test_version_endpoint(test_server):
    """Test /version endpoint returns expected payload."""
    client, server = test_server
    
    resp = await client.get("/version")
    assert resp.status == 200
    data = await resp.json()
    assert data == {
        "pipeline": "byoc",
        "model_id": "trickle-stream-example",
        "version": "0.0.1",
    }


@pytest.mark.asyncio
async def test_health_endpoint_ready_state(test_server):
    """Test /health endpoint returns proper state when ready."""
    client, server = test_server
    
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    
    # Health endpoint now returns exact specification format
    assert data == {"status": "IDLE"}  # Ready but no active streams
    assert isinstance(data["status"], str)
    assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]


@pytest.mark.asyncio
async def test_health_endpoint_with_active_streams(test_server):
    """Test /health endpoint when there are active streams."""
    client, server = test_server
    
    # Simulate active stream
    server.state.update_active_streams(1)
    
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    
    assert data == {"status": "OK"}  # Ready with active streams
    assert isinstance(data["status"], str)
    assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]


@pytest.mark.asyncio
async def test_health_endpoint_error_state(test_server):
    """Test /health endpoint returns ERROR status."""
    client, server = test_server
    
    # Set error state
    server.state.set_error("Test error")
    
    resp = await client.get("/health")
    assert resp.status == 200  # Always HTTP 200 per specification
    data = await resp.json()
    
    assert data == {"status": "ERROR"}  # Exact format specification
    assert isinstance(data["status"], str)
    assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]


@pytest.mark.asyncio
async def test_health_endpoint_loading_state(test_server):
    """Test /health endpoint during loading state."""
    client, server = test_server
    
    # Reset to loading state
    server.state.set_state(PipelineState.LOADING)
    server.state.startup_complete = False
    
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    
    assert data == {"status": "LOADING"}  # Exact format specification
    assert isinstance(data["status"], str)
    assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]


@pytest.mark.asyncio
async def test_hardware_info_endpoint(test_server):
    """Test /hardware/info endpoint returns expected structure."""
    client, server = test_server
    
    with patch.object(server.hardware_info, 'get_gpu_compute_info') as mock_gpu_info:
        # Mock GPU info response - should return Dict[int, GPUComputeInfo]
        from pytrickle.utils.hardware import GPUComputeInfo
        mock_gpu_info.return_value = {
            0: GPUComputeInfo(
                id="0",
                name="NVIDIA GeForce RTX 3080",
                memory_total=10737418240,
                memory_free=8589934592,
                major=8,
                minor=6
            )
        }
        
        resp = await client.get("/hardware/info")
        assert resp.status == 200
        data = await resp.json()
        
        # Check expected structure
        assert "pipeline" in data
        assert "model_id" in data
        assert "gpu_info" in data
        
        assert data["pipeline"] == "byoc"
        assert data["model_id"] == "trickle-stream-example"
        assert "0" in data["gpu_info"]  # GPU 0 should be present
        assert data["gpu_info"]["0"]["name"] == "NVIDIA GeForce RTX 3080"


@pytest.mark.asyncio
async def test_hardware_stats_endpoint(test_server):
    """Test /hardware/stats endpoint returns expected structure."""
    client, server = test_server
    
    with patch.object(server.hardware_info, 'get_gpu_utilization_stats') as mock_gpu_stats:
        # Mock GPU stats response - should return Dict[int, GPUUtilizationInfo]
        from pytrickle.utils.hardware import GPUUtilizationInfo
        mock_gpu_stats.return_value = {
            0: GPUUtilizationInfo(
                id="0",
                name="NVIDIA GeForce RTX 3080",
                memory_total=10737418240,
                memory_free=8589934592,
                utilization_compute=75,
                utilization_memory=60
            )
        }
        
        resp = await client.get("/hardware/stats")
        assert resp.status == 200
        data = await resp.json()
        
        # Check expected structure
        assert "pipeline" in data
        assert "model_id" in data
        assert "gpu_stats" in data
        
        assert data["pipeline"] == "byoc"
        assert data["model_id"] == "trickle-stream-example"
        assert "0" in data["gpu_stats"]  # GPU 0 should be present
        assert data["gpu_stats"]["0"]["utilization_compute"] == 75
        assert data["gpu_stats"]["0"]["utilization_memory"] == 60


@pytest.mark.asyncio
async def test_hardware_endpoints_handle_exceptions(test_server):
    """Test hardware endpoints handle exceptions gracefully."""
    client, server = test_server
    
    # Test /hardware/info with exception
    with patch.object(server.hardware_info, 'get_gpu_compute_info') as mock_gpu_info:
        mock_gpu_info.side_effect = Exception("GPU info unavailable")
        
        resp = await client.get("/hardware/info")
        assert resp.status == 500
        # The error response is plain text, not JSON
        text = await resp.text()
        assert "GPU info unavailable" in text or "Internal Server Error" in text
    
    # Test /hardware/stats with exception
    with patch.object(server.hardware_info, 'get_gpu_utilization_stats') as mock_gpu_stats:
        mock_gpu_stats.side_effect = Exception("GPU stats unavailable")
        
        resp = await client.get("/hardware/stats")
        assert resp.status == 500
        # The error response is plain text, not JSON
        text = await resp.text()
        assert "GPU stats unavailable" in text or "Internal Server Error" in text


@pytest.mark.asyncio
async def test_all_endpoints_respond(test_server):
    """Smoke test that all system endpoints respond without crashing."""
    client, server = test_server
    
    endpoints = ["/health", "/version", "/hardware/info", "/hardware/stats"]
    
    for endpoint in endpoints:
        resp = await client.get(endpoint)
        # Should not crash - status codes can vary based on mocks/state
        assert resp.status in [200, 500, 503]
        
        # Should return valid JSON
        data = await resp.json()
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_health_endpoint_response_format_consistency(test_server):
    """Test that /health endpoint always returns exact format specification."""
    client, server = test_server
    
    # Test various states
    states_to_test = [
        (PipelineState.LOADING, "LOADING"),
        (PipelineState.IDLE, "IDLE"),
        (PipelineState.ERROR, "ERROR"),
    ]
    
    for state, expected_status in states_to_test:
        server.state.set_state(state)
        
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        
        # Check exact format specification
        assert data == {"status": expected_status}
        assert isinstance(data["status"], str)
        assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]


@pytest.mark.asyncio
async def test_version_endpoint_custom_values(test_server):
    """Test /version endpoint with custom pipeline and version values."""
    # Create server with custom values
    processor = _InternalFrameProcessor(
        video_processor=process_video,
        model_loader=load_model,
        param_updater=update_params,
        name="custom-processor",
    )

    server = StreamServer(
        frame_processor=processor,
        port=0,
        capability_name="custom-model",
        pipeline="custom-pipeline",
        version="1.2.3",
        enable_default_routes=True,
    )

    app = server.get_app()
    test_server_instance = TestServer(app)
    
    async with test_server_instance:
        client = TestClient(test_server_instance)
        async with client:
            resp = await client.get("/version")
            assert resp.status == 200
            data = await resp.json()
            assert data == {
                "pipeline": "custom-pipeline",
                "model_id": "custom-model",
                "version": "1.2.3",
            }
