"""
Tests for StreamLifecycleManager.

Tests the separated lifecycle management functionality for better
testability and maintainability.
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock

from pytrickle.lifecycle import StreamLifecycleManager
from pytrickle.test_utils import create_test_lifecycle_manager, MockProtocolFactory, MockClientFactory
from pytrickle.api import StreamStartRequest
from pytrickle.state import StreamState, PipelineState


@pytest.mark.asyncio
async def test_lifecycle_manager_start_stream():
    """Test stream start with lifecycle manager."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Create start request
    params = StreamStartRequest(
        subscribe_url="http://localhost:3389/input",
        publish_url="http://localhost:3389/output",
        gateway_request_id="test-123",
        params={"intensity": 0.8}
    )
    
    # Start stream
    result = await lifecycle_manager.start_stream(params)
    
    # Verify result
    assert result.success is True
    assert result.message == "Stream started successfully"
    assert result.request_id == "test-123"
    
    # Verify state updates
    assert lifecycle_manager.state.active_client is True
    assert lifecycle_manager.state.active_streams == 1
    assert lifecycle_manager.current_params == params


@pytest.mark.asyncio
async def test_lifecycle_manager_stop_stream():
    """Test stream stop with lifecycle manager."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Start a stream first
    params = StreamStartRequest(
        subscribe_url="http://localhost:3389/input",
        publish_url="http://localhost:3389/output",
        gateway_request_id="test-123"
    )
    await lifecycle_manager.start_stream(params)
    
    # Verify it's started
    assert lifecycle_manager.state.active_streams == 1
    
    # Stop stream
    result = await lifecycle_manager.stop_stream()
    
    # Verify result
    assert result.success is True
    assert result.message == "Stream stopped successfully"
    
    # Verify state updates
    assert lifecycle_manager.state.active_client is False
    assert lifecycle_manager.state.active_streams == 0


@pytest.mark.asyncio
async def test_lifecycle_manager_stop_no_stream():
    """Test stop when no stream is active."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Try to stop without starting
    result = await lifecycle_manager.stop_stream()
    
    # Should fail gracefully
    assert result.success is False
    assert "No active stream to stop" in result.message


@pytest.mark.asyncio
async def test_lifecycle_manager_update_params():
    """Test parameter updates with lifecycle manager."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Start a stream first
    params = StreamStartRequest(
        subscribe_url="http://localhost:3389/input",
        publish_url="http://localhost:3389/output",
        gateway_request_id="test-123"
    )
    await lifecycle_manager.start_stream(params)
    
    # Update parameters
    update_params = {"intensity": 0.9, "delay": 0.1}
    result = await lifecycle_manager.update_params(update_params)
    
    # Verify result
    assert result.success is True
    assert result.message == "Parameters updated successfully"
    
    # Verify parameters were passed to frame processor
    assert lifecycle_manager.frame_processor.test_params == update_params


@pytest.mark.asyncio
async def test_lifecycle_manager_update_params_no_stream():
    """Test parameter update when no stream is active."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Try to update without starting
    result = await lifecycle_manager.update_params({"intensity": 0.5})
    
    # Should fail gracefully
    assert result.success is False
    assert "No active stream to update" in result.message


@pytest.mark.asyncio
async def test_lifecycle_manager_get_status():
    """Test status retrieval with lifecycle manager."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Get status without active stream
    status = lifecycle_manager.get_status()
    assert status["client_active"] is False
    
    # Start a stream
    params = StreamStartRequest(
        subscribe_url="http://localhost:3389/input",
        publish_url="http://localhost:3389/output",
        gateway_request_id="test-123"
    )
    await lifecycle_manager.start_stream(params)
    
    # Get status with active stream
    status = lifecycle_manager.get_status()
    assert status["client_active"] is True
    assert "current_params" in status
    assert status["current_params"]["gateway_request_id"] == "test-123"


@pytest.mark.asyncio
async def test_mock_factories_store_creation_params():
    """Test that mock factories store creation parameters for verification."""
    protocol_factory = MockProtocolFactory()
    client_factory = MockClientFactory()
    
    lifecycle_manager = StreamLifecycleManager(
        frame_processor=None,  # Will be created by test utils
        state=StreamState(),
        protocol_factory=protocol_factory,
        client_factory=client_factory,
    )
    
    # Create test utils version
    from pytrickle.test_utils import MockFrameProcessor
    lifecycle_manager.frame_processor = MockFrameProcessor()
    
    # Start a stream
    params = StreamStartRequest(
        subscribe_url="http://localhost:3389/input",
        publish_url="http://localhost:3389/output",
        gateway_request_id="test-123",
        params={"width": 1920, "height": 1080}
    )
    
    await lifecycle_manager.start_stream(params)
    
    # Verify protocol factory was called with correct parameters
    assert protocol_factory.last_creation_params["subscribe_url"] == "http://localhost:3389/input"
    assert protocol_factory.last_creation_params["publish_url"] == "http://localhost:3389/output"
    assert protocol_factory.last_creation_params["width"] == 1920
    assert protocol_factory.last_creation_params["height"] == 1080
    
    # Verify client factory was called
    assert client_factory.last_creation_params["frame_processor"] == lifecycle_manager.frame_processor


@pytest.mark.asyncio
async def test_state_updates_are_synchronous():
    """Test that state updates happen synchronously and predictably."""
    lifecycle_manager = create_test_lifecycle_manager()
    
    # Initial state
    assert lifecycle_manager.state.active_streams == 0
    assert lifecycle_manager.state.active_client is False
    
    # Start stream
    params = StreamStartRequest(
        subscribe_url="http://localhost:3389/input",
        publish_url="http://localhost:3389/output",
        gateway_request_id="test-123"
    )
    
    result = await lifecycle_manager.start_stream(params)
    assert result.success is True
    
    # State should be updated immediately and synchronously
    assert lifecycle_manager.state.active_streams == 1
    assert lifecycle_manager.state.active_client is True
    
    # Stop stream
    result = await lifecycle_manager.stop_stream()
    assert result.success is True
    
    # State should be updated immediately
    assert lifecycle_manager.state.active_streams == 0
    assert lifecycle_manager.state.active_client is False
