"""
Tests for all system endpoints in StreamServer.

Tests the /health, /version, /hardware/info, and /hardware/stats endpoints
with comprehensive state validation using the simplified architecture.
"""

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer, TestClient
from unittest.mock import patch, MagicMock

from pytrickle.stream_processor import _InternalFrameProcessor
from pytrickle.server import StreamServer
from pytrickle.state import PipelineState
from pytrickle.test_utils import MockFrameProcessor


@pytest_asyncio.fixture
async def test_server():
    """Create a test server with the example processor."""
    # Try to import example functions, fall back to mock if not available
    try:
        from pytrickle.examples.process_video_example import load_model, process_video, update_params
        processor = _InternalFrameProcessor(
            video_processor=process_video,
            audio_processor=None,
            model_loader=load_model,
            param_updater=update_params,
            name="trickle-stream-example",
        )
    except ImportError:
        processor = MockFrameProcessor()

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


@pytest_asyncio.fixture
async def health_test_server():
    """Create a server specifically for health endpoint testing with state validation."""
    processor = MockFrameProcessor()

    server = StreamServer(
        frame_processor=processor,
        port=0,
        capability_name="health-test",
        pipeline="health-test-pipeline",
        version="1.0.0",
    )

    # Attach state but don't set startup complete yet for testing
    processor.attach_state(server.state)

    app = server.get_app()
    test_server_instance = TestServer(app)
    
    async with test_server_instance:
        client = TestClient(test_server_instance)
        async with client:
            yield client, server


class TestVersionEndpoint:
    """Test version endpoint functionality."""

    @pytest.mark.asyncio
    async def test_version_endpoint(self, test_server):
        """Test /version endpoint returns expected payload."""
        client, server = test_server
        
        resp = await client.get("/version")
        assert resp.status == 200
        data = await resp.json()
        assert data["pipeline"] == "byoc"
        assert data["model_id"] == "trickle-stream-example"
        # Version should start with 0.1. (setuptools-scm generates versions like 0.1.4.dev0+...)
        assert data["version"].startswith("0.1.")

    @pytest.mark.asyncio
    async def test_version_endpoint_custom_values(self):
        """Test /version and /app-version endpoints with custom pipeline and version values."""
        # Try to use example functions, fall back to simple test functions
        try:
            from pytrickle.examples.process_video_example import load_model, process_video, update_params
        except ImportError:
            # Define simple test functions if examples aren't available
            async def process_video(frame):
                return frame
            async def load_model():
                pass
            async def update_params(params):
                pass
        
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
                # Test /version endpoint returns package version
                resp = await client.get("/version")
                assert resp.status == 200
                data = await resp.json()
                assert data["pipeline"] == "custom-pipeline"
                assert data["model_id"] == "custom-model"
                # Version should start with 0.1. (setuptools-scm generates versions like 0.1.4.dev0+...)
                assert data["version"].startswith("0.1.")
                
                # Test /app-version endpoint returns custom app version
                resp = await client.get("/app-version")
                assert resp.status == 200
                data = await resp.json()
                assert data == {
                    "version": "1.2.3",  # Custom app version
                }
    
    @pytest.mark.asyncio
    async def test_app_version_endpoint(self, test_server):
        """Test /app-version endpoint returns expected payload."""
        client, server = test_server
        
        resp = await client.get("/app-version")
        assert resp.status == 200
        data = await resp.json()
        assert data == {
            "version": "0.0.1",  # Default app version from server init
        }


class TestHealthEndpointStateValidation:
    """Test health endpoint with comprehensive state validation."""

    @pytest.mark.asyncio
    async def test_health_status_loading(self, health_test_server):
        """Test health endpoint returns 'LOADING' during startup."""
        client, server = health_test_server
        
        # Before startup complete - should be LOADING
        resp = await client.get("/health")
        assert resp.status == 200
        
        data = await resp.json()
        assert data == {"status": "LOADING"}
        assert isinstance(data["status"], str)
        assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]

    @pytest.mark.asyncio
    async def test_health_status_idle(self, health_test_server):
        """Test health endpoint returns 'IDLE' when ready but no streams."""
        client, server = health_test_server
        
        # Complete startup but no active streams
        server.state.set_startup_complete()
        
        resp = await client.get("/health")
        assert resp.status == 200
        
        data = await resp.json()
        assert data == {"status": "IDLE"}
        assert isinstance(data["status"], str)
        assert data["status"] in ["LOADING", "OK", "ERROR", "IDLE"]

    @pytest.mark.asyncio
    async def test_health_status_ok_with_active_streams(self, health_test_server):
        """Test health endpoint returns 'OK' with active streams."""
        client, server = health_test_server
        
        # Complete startup and add active streams
        server.state.set_startup_complete()
        server.state.update_active_streams(2)
        server.state.set_active_client(True)
        
        resp = await client.get("/health")
        assert resp.status == 200
        
        data = await resp.json()
        assert data == {"status": "OK"}
        
        # Validate state consistency
        assert server.state.active_streams == 2
        assert server.state.active_client is True
        assert server.state.pipeline_ready is True

    @pytest.mark.asyncio
    async def test_health_status_error_with_state_validation(self, health_test_server):
        """Test health endpoint returns 'ERROR' and validates state."""
        client, server = health_test_server
        
        # Set up active streams first
        server.state.set_startup_complete()
        server.state.update_active_streams(3)
        server.state.set_active_client(True)
        
        # Verify it's OK first
        resp = await client.get("/health")
        data = await resp.json()
        assert data == {"status": "OK"}
        
        # Set error state
        server.state.set_error("Test error condition")
        
        # Should now return ERROR
        resp = await client.get("/health")
        assert resp.status == 200  # Always HTTP 200 per specification
        
        data = await resp.json()
        assert data == {"status": "ERROR"}
        
        # Validate error state
        assert server.state.is_error()
        assert server.state.error_event.is_set()

    @pytest.mark.asyncio
    async def test_health_state_transitions_comprehensive(self, health_test_server):
        """Test all health state transitions with validation."""
        client, server = health_test_server
        
        # Test each state independently to avoid state pollution
        
        # 1. Test LOADING state
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "LOADING"
        assert not server.state.startup_complete
        
        # 2. Test IDLE state
        server.state.set_startup_complete()
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "IDLE"
        assert server.state.pipeline_ready
        assert server.state.startup_complete
        
        # 3. Test OK state
        server.state.update_active_streams(1)
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "OK"
        assert server.state.active_streams > 0
        
        # 4. Test ERROR state
        server.state.set_error("Test error")
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ERROR"
        assert server.state.is_error()


class TestHardwareEndpoints:
    """Test hardware information endpoints."""

    @pytest.mark.asyncio
    async def test_hardware_info_endpoint(self, test_server):
        """Test /hardware/info endpoint returns expected structure."""
        client, server = test_server
        
        with patch.object(server.hardware_info, 'get_gpu_compute_info') as mock_gpu_info:
            # Mock GPU info response
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

    @pytest.mark.asyncio
    async def test_hardware_stats_endpoint(self, test_server):
        """Test /hardware/stats endpoint returns expected structure."""
        client, server = test_server
        
        with patch.object(server.hardware_info, 'get_gpu_utilization_stats') as mock_gpu_stats:
            # Mock GPU stats response
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

    @pytest.mark.asyncio
    async def test_all_endpoints_respond(self, test_server):
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
