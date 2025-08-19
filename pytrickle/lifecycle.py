"""
Stream Lifecycle Manager for Trickle Streaming.

Separates stream lifecycle management from HTTP handling to improve
testability and maintainability.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any, Protocol
from dataclasses import dataclass

from .client import TrickleClient
from .protocol import TrickleProtocol
from .api import StreamStartRequest
from .state import StreamState, PipelineState
from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class StreamResult:
    """Result of stream start operation."""
    success: bool
    message: str
    request_id: Optional[str] = None
    error: Optional[Exception] = None


@dataclass
class StopResult:
    """Result of stream stop operation."""
    success: bool
    message: str
    error: Optional[Exception] = None


class ProtocolFactory(Protocol):
    """Protocol for creating TrickleProtocol instances."""
    
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
        """Create a TrickleProtocol instance."""
        ...


class ClientFactory(Protocol):
    """Protocol for creating TrickleClient instances."""
    
    def create_client(
        self,
        protocol: TrickleProtocol,
        frame_processor: FrameProcessor,
        control_handler: Optional[Callable] = None,
    ) -> TrickleClient:
        """Create a TrickleClient instance."""
        ...


class DefaultProtocolFactory:
    """Default factory for creating TrickleProtocol instances."""
    
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
        """Create a TrickleProtocol instance."""
        return TrickleProtocol(
            subscribe_url=subscribe_url,
            publish_url=publish_url,
            control_url=control_url,
            events_url=events_url,
            data_url=data_url,
            width=width,
            height=height,
            max_framerate=max_framerate,
            publisher_timeout=publisher_timeout,
            subscriber_timeout=subscriber_timeout,
        )


class DefaultClientFactory:
    """Default factory for creating TrickleClient instances."""
    
    def create_client(
        self,
        protocol: TrickleProtocol,
        frame_processor: FrameProcessor,
        control_handler: Optional[Callable] = None,
    ) -> TrickleClient:
        """Create a TrickleClient instance."""
        return TrickleClient(
            protocol=protocol,
            frame_processor=frame_processor,
            control_handler=control_handler,
        )


class StreamLifecycleManager:
    """Manages stream lifecycle operations independently of HTTP handling."""
    
    def __init__(
        self,
        frame_processor: FrameProcessor,
        state: StreamState,
        protocol_factory: ProtocolFactory = None,
        client_factory: ClientFactory = None,
        publisher_timeout: Optional[float] = None,
        subscriber_timeout: Optional[float] = None,
    ):
        """Initialize the lifecycle manager.
        
        Args:
            frame_processor: Frame processor instance
            state: Stream state manager
            protocol_factory: Factory for creating protocols (defaults to DefaultProtocolFactory)
            client_factory: Factory for creating clients (defaults to DefaultClientFactory)
            publisher_timeout: Timeout for publishers
            subscriber_timeout: Timeout for subscribers
        """
        self.frame_processor = frame_processor
        self.state = state
        self.protocol_factory = protocol_factory or DefaultProtocolFactory()
        self.client_factory = client_factory or DefaultClientFactory()
        self.publisher_timeout = publisher_timeout
        self.subscriber_timeout = subscriber_timeout
        
        # Current stream state
        self.current_client: Optional[TrickleClient] = None
        self.current_params: Optional[StreamStartRequest] = None
        self._client_task: Optional[asyncio.Task] = None
        self._lifecycle_lock = asyncio.Lock()
    
    async def start_stream(self, params: StreamStartRequest) -> StreamResult:
        """Start a new stream with the given parameters."""
        try:
            async with self._lifecycle_lock:
                # Clear any previous error state when starting a new stream
                if self.state.is_error():
                    self.state.clear_error()
                    logger.info("Cleared previous error state for new stream")
                
                # Extract dimensions and framerate from params
                params_dict = params.params or {}
                width = params_dict.get("width", 512)
                height = params_dict.get("height", 512)
                max_framerate = params_dict.get("max_framerate", None)
                
                # Create new protocol
                new_protocol = self.protocol_factory.create_protocol(
                    subscribe_url=params.subscribe_url,
                    publish_url=params.publish_url,
                    control_url=params.control_url or "",
                    events_url=params.events_url or "",
                    data_url=params.data_url,
                    width=width,
                    height=height,
                    max_framerate=max_framerate,
                    publisher_timeout=self.publisher_timeout,
                    subscriber_timeout=self.subscriber_timeout,
                )
                
                # Handle existing client
                if self.current_client is not None:
                    logger.info("Reusing existing TrickleClient with new protocol")
                    if self.current_client.running:
                        logger.info("Client already running - stopping current stream before restart")
                        await self._stop_current_stream()
                    
                    # Stop current protocol before swapping
                    try:
                        if self.current_client.protocol:
                            await self.current_client.protocol.stop()
                            logger.info("Stopped previous protocol")
                    except Exception as e:
                        logger.warning(f"Error stopping previous protocol (continuing): {e}")
                    
                    # Replace protocol and reset client state
                    self.current_client.protocol = new_protocol
                    self._reset_client_state()
                    logger.info("Client state reset for new stream")
                else:
                    logger.info("Creating new TrickleClient")
                    self.current_client = self.client_factory.create_client(
                        protocol=new_protocol,
                        frame_processor=self.frame_processor,
                        control_handler=self._handle_control_message,
                    )
                
                # Update current params and state
                self.current_params = params
                self._update_stream_state(started=True)
                
                # Start the client in background
                self._client_task = asyncio.create_task(self._run_client(params.gateway_request_id))
            
            # Set params via update_params if provided
            if params.params:
                try:
                    logger.info("Setting params from start stream request")
                    self.frame_processor.update_params(params.params)
                    logger.info("Params set successfully from start request")
                except Exception as e:
                    logger.warning(f"Failed to set params from start request: {e}")
            
            # Emit start event
            if self.current_client and self.current_client.protocol:
                await self.current_client.protocol.emit_monitoring_event({
                    "type": "stream_started",
                    "timestamp": int(time.time() * 1000),
                    "params": params.params or {}
                })
            
            return StreamResult(
                success=True,
                message="Stream started successfully",
                request_id=params.gateway_request_id
            )
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            import traceback
            logger.error(f"Stream start error traceback: {traceback.format_exc()}")
            self.state.set_error(f"Stream start failed: {str(e)}")
            return StreamResult(
                success=False,
                message=f"Error starting stream: {str(e)}",
                error=e
            )
    
    async def stop_stream(self) -> StopResult:
        """Stop the current stream."""
        try:
            async with self._lifecycle_lock:
                if not self.current_client:
                    return StopResult(
                        success=False,
                        message="No active stream to stop"
                    )
                
                await self._stop_current_stream()
            
            self._update_stream_state(started=False)
            
            return StopResult(
                success=True,
                message="Stream stopped successfully"
            )
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            return StopResult(
                success=False,
                message=f"Error stopping stream: {str(e)}",
                error=e
            )
    
    async def update_params(self, params: Dict[str, Any]) -> StreamResult:
        """Update stream parameters."""
        try:
            if not self.current_client:
                return StreamResult(
                    success=False,
                    message="No active stream to update"
                )
            
            # Update frame processor parameters
            self.frame_processor.update_params(params)
            logger.info(f"Parameters updated: {params}")
            
            # Emit parameter update event
            if self.current_client.protocol:
                await self.current_client.protocol.emit_monitoring_event({
                    "type": "params_updated",
                    "timestamp": int(time.time() * 1000),
                    "params": params
                })
            
            return StreamResult(
                success=True,
                message="Parameters updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return StreamResult(
                success=False,
                message=f"Parameter update failed: {str(e)}",
                error=e
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current stream status."""
        # Get base state
        status_data = self.state.get_pipeline_state()
        
        # Add client information if active
        if self.current_client:
            status_data["client_active"] = True
            status_data["client_running"] = self.current_client.running
            # Add FPS information from protocol
            if self.current_client.protocol:
                status_data["fps"] = self.current_client.protocol.fps_meter.get_fps_stats()
        else:
            status_data["client_active"] = False
        
        # Add current parameters if available
        if self.current_params:
            status_data["current_params"] = {
                "subscribe_url": self.current_params.subscribe_url,
                "publish_url": self.current_params.publish_url,
                "gateway_request_id": getattr(self.current_params, 'gateway_request_id', None)
            }
        
        return status_data
    
    def _update_stream_state(self, started: bool):
        """Update stream state synchronously and predictably."""
        if started:
            logger.info(f"Lifecycle: Setting stream state to STARTED - active_streams=1, active_client=True")
            self.state.set_active_client(True)
            self.state.update_active_streams(1)
            self.state.set_state(PipelineState.IDLE)
        else:
            logger.info(f"Lifecycle: Setting stream state to STOPPED - active_streams=0, active_client=False")
            self.state.set_active_client(False)
            self.state.update_active_streams(0)
            # When stream stops, clear any error state and return to IDLE
            if self.state.is_error():
                logger.info("Lifecycle: Clearing previous error state")
                self.state.clear_error()
            self.state.set_state(PipelineState.IDLE)
            logger.info(f"Lifecycle: Stream state updated - active_streams={self.state.active_streams}, active_client={self.state.active_client}")
    
    def _reset_client_state(self):
        """Reset client coordination events for clean start."""
        if self.current_client:
            self.current_client.stop_event.clear()
            self.current_client.error_event.clear()
            self.current_client.running = False
            
            # Clear output queue of any stale data
            while not self.current_client.output_queue.empty():
                try:
                    self.current_client.output_queue.get_nowait()
                except:
                    break
    
    async def _handle_control_message(self, control_data: dict):
        """Handle control messages from trickle protocol."""
        try:
            self.frame_processor.update_params(control_data)
            logger.debug("Control message routed to frame processor")
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
    
    async def _run_client(self, request_id: str):
        """Run the trickle client."""
        try:
            if self.current_client:
                self.state.set_state(PipelineState.IDLE)
                await self.current_client.start(request_id)
                logger.info("Client started successfully")
        except Exception as e:
            logger.error(f"Error running client: {e}")
            import traceback
            logger.error(f"Client run error traceback: {traceback.format_exc()}")
            self.state.set_error(f"Client execution failed: {str(e)}")
        finally:
            logger.info("Client ended - cleaning up")
            async with self._lifecycle_lock:
                await self._stop_current_stream()
    
    async def _stop_current_stream(self):
        """Stop the current stream protocol while preserving client."""
        if self.current_client:
            logger.info("Stopping current stream protocol...")
            
            # Stop the client first
            if self.current_client.running:
                await self.current_client.stop()
                logger.info("Client stopped")
            
            # Stop the protocol
            if self.current_client.protocol:
                await self.current_client.protocol.stop()
                logger.info("Protocol stopped")
            
            # Cancel client task
            if self._client_task and not self._client_task.done():
                self._client_task.cancel()
                try:
                    await self._client_task
                except asyncio.CancelledError:
                    pass
            self._client_task = None
            
            # Update state to reflect stream stopped
            self._update_stream_state(started=False)
            
            logger.info("Current stream stopped and resources cleaned up")
    
    async def shutdown(self):
        """Completely shutdown and cleanup all resources."""
        if self.current_client:
            await self.current_client.stop()
        
        self.current_client = None
        self.current_params = None
        self._update_stream_state(started=False)
