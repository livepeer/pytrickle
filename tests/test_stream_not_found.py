"""
Tests for stream_not_found (404) error handling.

Verifies that when a trickle stream is not found (404 error), the stream
properly terminates and cleanup occurs.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from pytrickle.client import TrickleClient
from pytrickle.protocol import TrickleProtocol
from pytrickle.subscriber import TrickleSubscriber
from pytrickle.test_utils import MockFrameProcessor


class TestStreamNotFoundClient:
    """Test client-level handling of stream_not_found errors."""

    @pytest.mark.asyncio
    async def test_stream_not_found_sets_stop_event(self):
        """Test that stream_not_found error sets stop_event in client."""
        processor = MockFrameProcessor()
        protocol = MagicMock(spec=TrickleProtocol)
        protocol.error_callback = None
        
        client = TrickleClient(
            protocol=protocol,
            frame_processor=processor
        )
        
        # Verify initial state
        assert not client.stop_event.is_set()
        assert not client.error_event.is_set()
        
        # Simulate stream_not_found error
        await client._on_protocol_error("stream_not_found", Exception("404 error"))
        
        # Verify stop_event is set
        assert client.stop_event.is_set()
        # Note: Based on current implementation, error_event is NOT set for stream_not_found
        # This is intentional - stream_not_found is treated as clean shutdown
        
    @pytest.mark.asyncio
    async def test_stream_not_found_vs_other_errors(self):
        """Test that stream_not_found sets stop_event but not error_event, unlike other errors."""
        processor = MockFrameProcessor()
        protocol = MagicMock(spec=TrickleProtocol)
        protocol.error_callback = None
        
        client = TrickleClient(
            protocol=protocol,
            frame_processor=processor
        )
        
        # Test stream_not_found - should set stop_event only
        await client._on_protocol_error("stream_not_found", Exception("404 error"))
        assert client.stop_event.is_set()
        assert not client.error_event.is_set()
        
        # Reset
        client.stop_event.clear()
        client.error_event.clear()
        
        # Test other error - should set error_event
        await client._on_protocol_error("connection_failed", Exception("Connection failed"))
        assert not client.stop_event.is_set()
        assert client.error_event.is_set()
        
    @pytest.mark.asyncio
    async def test_stream_not_found_calls_error_callback(self):
        """Test that stream_not_found error calls the error callback."""
        processor = MockFrameProcessor()
        protocol = MagicMock(spec=TrickleProtocol)
        protocol.error_callback = None
        
        error_callback = AsyncMock()
        client = TrickleClient(
            protocol=protocol,
            frame_processor=processor,
            error_callback=error_callback
        )
        
        exception = Exception("404 error for http://localhost:3389/input/0")
        await client._on_protocol_error("stream_not_found", exception)
        
        # Verify error callback was called
        error_callback.assert_called_once_with("stream_not_found", exception)
        
    @pytest.mark.asyncio
    async def test_stream_not_found_terminates_ingress_loop(self):
        """Test that ingress loop terminates when stop_event is set due to stream_not_found."""
        processor = MockFrameProcessor()
        protocol = MagicMock(spec=TrickleProtocol)
        protocol.error_callback = None
        
        # Mock ingress_loop to check stop_event - must be async generator
        async def mock_ingress_loop(stop_event):
            frame_count = 0
            while not stop_event.is_set():
                frame_count += 1
                if frame_count > 10:  # Safety limit
                    break
                await asyncio.sleep(0.01)
            # Should exit when stop_event is set
            # Empty generator - no frames yielded
            return
            yield  # Makes it an async generator (unreachable but required for type)
        
        protocol.ingress_loop = mock_ingress_loop
        
        client = TrickleClient(
            protocol=protocol,
            frame_processor=processor
        )
        
        # Start ingress loop in background
        ingress_task = asyncio.create_task(client._ingress_loop())
        
        # Wait a bit, then trigger stream_not_found
        await asyncio.sleep(0.05)
        await client._on_protocol_error("stream_not_found", Exception("404 error"))
        
        # Wait for ingress loop to terminate
        try:
            await asyncio.wait_for(ingress_task, timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Ingress loop did not terminate after stream_not_found")
        
        # Verify stop_event is set
        assert client.stop_event.is_set()


class TestStreamNotFoundSubscriber:
    """Test subscriber-level handling of 404 errors."""

    @pytest.mark.asyncio
    async def test_subscriber_preconnect_404_returns_none(self):
        """Test that subscriber preconnect returns None on 404."""
        error_callback = AsyncMock()
        subscriber = TrickleSubscriber(
            "http://localhost:3389/input",
            error_callback=error_callback,
            max_retries=1
        )
        
        # Mock aiohttp session to return 404
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.release = AsyncMock()
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            await subscriber.start()
            subscriber.session = mock_session
            
            result = await subscriber.preconnect()
            
            # Should return None on 404
            assert result is None
            
            # Should call error callback with stream_not_found
            error_callback.assert_called_once()
            call_args = error_callback.call_args
            assert call_args[0][0] == "stream_not_found"
            assert "404" in str(call_args[0][1])
            
    @pytest.mark.asyncio
    async def test_subscriber_404_no_retries(self):
        """Test that subscriber does not retry on 404 error."""
        error_callback = AsyncMock()
        subscriber = TrickleSubscriber(
            "http://localhost:3389/input",
            error_callback=error_callback,
            max_retries=3  # Even with multiple retries, 404 should not retry
        )
        
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.release = AsyncMock()
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            await subscriber.start()
            subscriber.session = mock_session
            
            result = await subscriber.preconnect()
            
            # Should return None immediately (no retries)
            assert result is None
            
            # Should only call error callback once (no retries)
            assert error_callback.call_count == 1
            
            # Verify GET was only called once (no retries for 404)
            assert mock_session.get.call_count == 1

