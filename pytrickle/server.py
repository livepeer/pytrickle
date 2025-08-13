"""
Trickle App HTTP Server.

Provides a REST API for managing trickle video streams with support for
starting streams, updating parameters, and monitoring status.
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import os

from aiohttp import web
from pydantic import BaseModel, Field

from .client import TrickleClient
from .protocol import TrickleProtocol
from .api import StreamParamsUpdateRequest, StreamStartRequest, Version, HardwareInformation, HardwareStats
from .state import StreamState
from .utils.hardware import HardwareInfo
from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

class TrickleApp:
    """HTTP server application for trickle streaming with native async processor support."""
    
    def __init__(
        self, 
        frame_processor: 'FrameProcessor',
        port: int = 8080,
        capability_name: str = "",
        version: str = "0.0.1"
    ):
        """Initialize TrickleApp.
        
        Args:
            frame_processor: FrameProcessor for native async processing
            port: HTTP server port
            capability_name: Name of the capability
            version: Version string
        """
        self.frame_processor = frame_processor
        self.port = port
        self.state = StreamState()
        self.app = web.Application()
        self.current_client: Optional[TrickleClient] = None
        self.current_params: Optional[StreamStartRequest] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._client_task: Optional[asyncio.Task] = None
        self.version = version
        self.hardware_info = HardwareInfo()
        self.capability_name = capability_name or os.getenv("CAPABILITY_NAME", "")

        # Setup routes
        self._setup_routes()
    
    def _default_hardware_info(self) -> Dict[str, Any]:
        """Return default hardware info."""
        return self.hardware_info.get_gpu_compute_info()
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post("/api/stream/start", self._handle_start_stream)
        self.app.router.add_post("/api/stream/stop", self._handle_stop_stream)
        self.app.router.add_post("/api/stream/params", self._handle_update_params)
        self.app.router.add_get("/api/stream/status", self._handle_get_status)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/version", self._handle_version)
        self.app.router.add_get("/hardware/info", self._handle_hardware_info)
        self.app.router.add_get("/hardware/stats", self._handle_hardware_stats)

        # Alias for live-video-to-video endpoint (same as stream/start)
        self.app.router.add_post("/live-video-to-video", self._handle_start_stream)
    
    async def _parse_and_validate_request(self, request: web.Request, model_class):
        """Parse and validate request data using a Pydantic model."""
        if request.content_type.startswith("application/json"):
            data = await request.json()
        else:
            raise ValueError(f"Unknown content type: {request.content_type}")
        return model_class.model_validate(data)
    
    async def _handle_start_stream(self, request: web.Request) -> web.Response:
        """Handle stream start requests."""
        try:
            # Stop any existing stream
            if self.current_client:
                await self._stop_current_stream()
            
            # Parse and validate request
            params = await self._parse_and_validate_request(request, StreamStartRequest)
            self.current_params = params
            
            logger.info(f"Starting stream: {params.subscribe_url} -> {params.publish_url}")
            
            # Extract dimensions and framerate from params (already converted to int by validation)
            params_dict = params.params or {}
            width = params_dict.get("width", 512)
            height = params_dict.get("height", 512)
            max_framerate = params_dict.get("max_framerate", None)  # None will use default
            
            # Create protocol and client (align with current Client/Protocol API)
            protocol = TrickleProtocol(
                subscribe_url=params.subscribe_url,
                publish_url=params.publish_url,
                control_url=params.control_url or "",
                events_url=params.events_url or "",
                data_url=params.data_url,
                width=width,
                height=height,
                max_framerate=max_framerate,
            )

            # Create TrickleClient with native async processor
            logger.info("Creating TrickleClient with native async processor")
            self.current_client = TrickleClient(
                protocol=protocol,
                frame_processor=self.frame_processor,
            )
            
            # Track active client and start health monitoring
            self.state.set_active_client(True)
            self._start_health_monitoring()
            
            # Start the client in background
            self._client_task = asyncio.create_task(self._run_client(params.gateway_request_id))
            
            # Emit start event via protocol events publisher if available
            if self.current_client and self.current_client.protocol:
                await self.current_client.protocol.emit_monitoring_event({
                    "type": "stream_started",
                    "timestamp": int(time.time() * 1000),
                    "params": params.params or {}
                })
            
            return web.json_response({
                "status": "success",
                "message": "Stream started successfully",
                "request_id": params.gateway_request_id
            })
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            self.state.stream_errors.append(f"Stream start failed: {str(e)}")
            self.state.error_event.set()
            return web.json_response({
                "status": "error",
                "message": f"Error starting stream: {str(e)}"
            }, status=400)
    
    async def _handle_stop_stream(self, request: web.Request) -> web.Response:
        """Handle stream stop requests."""
        try:
            if not self.current_client:
                return web.json_response({
                    "status": "error",
                    "message": "No active stream to stop"
                }, status=400)
            
            await self._stop_current_stream()
            
            return web.json_response({
                "status": "success",
                "message": "Stream stopped successfully"
            })
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error stopping stream: {str(e)}"
            }, status=500)
    
    async def _handle_update_params(self, request: web.Request) -> web.Response:
        """Handle parameter update requests."""
        try:
            if not self.current_client:
                return web.json_response({
                    "status": "error",
                    "message": "No active stream to update"
                }, status=400)
            
            # Parse and validate request
            validated_params = await self._parse_and_validate_request(request, StreamParamsUpdateRequest)
            data = validated_params.model_dump()
            
            # Update frame processor parameters directly
            try:
                self.frame_processor.update_params(data)
                logger.info(f"Parameters updated: {data}")
            except Exception as e:
                logger.error(f"Error updating parameters: {e}")
                return web.json_response({
                    "status": "error",
                    "message": f"Parameter update failed: {str(e)}"
                }, status=500)
            
            # Emit parameter update event via protocol events publisher if available
            if self.current_client.protocol:
                await self.current_client.protocol.emit_monitoring_event({
                    "type": "params_updated",
                    "timestamp": int(time.time() * 1000),
                    "params": data
                })
            
            logger.info(f"Parameters updated: {data}")
            
            return web.json_response({
                "status": "success",
                "message": "Parameters updated successfully"
            })
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error updating parameters: {str(e)}"
            }, status=500)
    
    async def _handle_get_status(self, request: web.Request) -> web.Response:
        """Handle status requests with detailed state and frame processing statistics."""
        try:
            # Get base state
            status_data = self.state.get_state()
            
            # Add basic client information if active
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
            
            return web.json_response(status_data)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error getting status: {str(e)}"
            }, status=500)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests with simplified state."""
        try:
            state_data = self.state.get_state()
            #TODO: Should we return 503 if in error state?
            # status_code = 503 if state_data["error"] else 200
            return web.json_response(state_data, status=200)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }, status=500)
    
    async def _handle_version(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        capability_name = os.getenv("CAPABILITY_NAME", "")
        return web.json_response(Version(
            pipeline=os.getenv("PIPELINE", "byoc"),
            model_id=capability_name,
            version=self.version
        ).model_dump())
    
    async def _handle_hardware_info(self, request: web.Request) -> web.Response:
        """Handle hardware info requests."""

        return web.json_response(HardwareInformation(
            pipeline=os.getenv("PIPELINE", "byoc"),
            model_id=self.capability_name,
            gpu_info=self._default_hardware_info()
        ).model_dump())

    async def _handle_hardware_stats(self, request: web.Request) -> web.Response:
        """Handle hardware stats requests."""
        return web.json_response(HardwareStats(
            pipeline=os.getenv("PIPELINE", "byoc"),
            model_id=self.capability_name,
            gpu_stats=self.hardware_info.get_gpu_utilization_stats()
        ).model_dump())
    
    async def _run_client(self, request_id: str):
        """Run the trickle client."""
        try:
            if self.current_client:
                # Set pipeline ready when client starts successfully
                self.state.set_pipeline_ready()
                await self.current_client.start(request_id)
        except Exception as e:
            logger.error(f"Error running client: {e}")
            # Set error state on client failure
            self.state.error_event.set()
        finally:
            # Properly stop the current stream instead of just clearing references
            await self._stop_current_stream()
    
    def _start_health_monitoring(self):
        """Start background health monitoring task."""
        if self._health_monitor_task and not self._health_monitor_task.done():
            return
        self._health_monitor_task = asyncio.create_task(self._health_monitoring_loop())

    async def _health_monitoring_loop(self):
        """Background task to monitor component health."""
        try:
            while self.current_client:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                if self.current_client and self.current_client.protocol:
                    protocol = self.current_client.protocol
                    self.state.update_component_health(
                        protocol.component_name,
                        protocol.get_component_health()
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")

    async def _stop_current_stream(self):
        """Stop the current stream and wait for complete cleanup."""
        if self.current_client:
            logger.info("Stopping current stream...")
            
            # Stop the client and wait for cleanup
            await self.current_client.stop()
            
            logger.info("FrameProcessor returned to idle state")
                
            self.current_client = None
            self.current_params = None
            
            # Update state and stop monitoring
            self.state.set_active_client(False)
            
            # Cancel client task
            if self._client_task and not self._client_task.done():
                self._client_task.cancel()
                try:
                    await self._client_task
                except asyncio.CancelledError:
                    pass
            self._client_task = None
            
            # Cancel health monitoring task  
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Current stream stopped and resources cleaned up")
    
    async def stop(self):
        """Stop the current trickle client and clean up resources."""
        await self._stop_current_stream()
    
    async def start_server(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        
        # Set pipeline ready when server is up and ready to accept requests
        self.state.set_pipeline_ready()
        
        logger.info(f"Trickle app server started on port {self.port}")
        return runner
    
    async def run_forever(self):
        """Run the server forever."""
        runner = await self.start_server()
        try:
            # Keep the server running
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            await self._stop_current_stream()
            await runner.cleanup()

def create_app(
    frame_processor: 'FrameProcessor',
    port: int = 8080,
    capability_name: str = "",
    version: str = "0.0.1"
) -> TrickleApp:
    """Create a trickle app instance.
    
    Args:
        frame_processor: FrameProcessor for native async processing
        port: HTTP server port
        capability_name: Name of the capability
        version: Version string
        
    Returns:
        TrickleApp instance
        
    Example:
        processor = MyAsyncProcessor()
        await processor.start()
        app = create_app(frame_processor=processor, port=8080)
        await app.run_forever()
    """
    return TrickleApp(
        frame_processor=frame_processor,
        port=port,
        capability_name=capability_name,
        version=version
    ) 