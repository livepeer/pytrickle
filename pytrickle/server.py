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

from aiohttp import web
from pydantic import BaseModel, Field

from .client import TrickleClient
from .protocol import TrickleProtocol
from .frames import VideoFrame, VideoOutput, AudioFrame, AudioOutput
from .api import StreamParamsUpdateRequest, StreamStartRequest

logger = logging.getLogger(__name__)



class TrickleApp:
    """HTTP server application for trickle streaming."""
    
    def __init__(
        self, 
        frame_processor: Optional[Callable[[Union[VideoFrame, AudioFrame]], Union[VideoOutput, AudioOutput]]] = None,
        port: int = 8080
    ):
        self.frame_processor = frame_processor or self._default_frame_processor
        self.port = port
        self.app = web.Application()
        self.current_client: Optional[TrickleClient] = None
        self.current_params: Optional[StreamStartRequest] = None
        
        # Setup routes
        self._setup_routes()
    
    def _default_frame_processor(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        """Default frame processor that passes frames through unchanged."""
        if isinstance(frame, VideoFrame):
            return VideoOutput(frame, self.current_params.gateway_request_id if self.current_params else "default")
        elif isinstance(frame, AudioFrame):
            return AudioOutput([frame], self.current_params.gateway_request_id if self.current_params else "default")
        else:
            raise ValueError(f"Unknown frame type: {type(frame)}")
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post("/api/stream/start", self._handle_start_stream)
        self.app.router.add_post("/api/stream/stop", self._handle_stop_stream)
        self.app.router.add_post("/api/stream/params", self._handle_update_params)
        self.app.router.add_get("/api/stream/status", self._handle_get_status)
        self.app.router.add_get("/health", self._handle_health)
        
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
            
            # Extract dimensions from params (already converted to int by validation)
            params_dict = params.params or {}
            width = params_dict.get("width", 512)
            height = params_dict.get("height", 512)
            
            # Create protocol and client (align with current Client/Protocol API)
            protocol = TrickleProtocol(
                subscribe_url=params.subscribe_url,
                publish_url=params.publish_url,
                control_url=params.control_url or "",
                events_url=params.events_url or "",
                data_url=params.data_url,
                width=width,
                height=height,
            )

            self.current_client = TrickleClient(
                protocol=protocol,
                frame_processor=self.frame_processor,
            )
            
            # Start the client in background
            asyncio.create_task(self._run_client(params.gateway_request_id))
            
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
        """Handle status requests."""
        try:
            status = {
                "status": "online" if self.current_client else "offline",
                "timestamp": int(time.time() * 1000),
                "has_active_stream": self.current_client is not None,
                "current_params": self.current_params.model_dump() if self.current_params else None
            }
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error getting status: {str(e)}"
            }, status=500)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response({
            "status": "healthy",
            "service": "Pytrickle Streaming Server",
            "endpoints": {
                "start": "POST /api/stream/start - Start streaming",
                "stop": "POST /api/stream/stop - Stop streaming", 
                "params": "POST /api/stream/params - Update parameters",
                "status": "GET /api/stream/status - Get status",
                "health": "GET /health - Health check",
                "live_video": "POST /live-video-to-video - Start live video processing (alias for /api/stream/start)"
            }
        })
    
    async def _run_client(self, request_id: str):
        """Run the trickle client."""
        try:
            if self.current_client:
                await self.current_client.start(request_id)
        except Exception as e:
            logger.error(f"Error running client: {e}")
        finally:
            self.current_client = None
            self.current_params = None
    
    async def _stop_current_stream(self):
        """Stop the current stream."""
        if self.current_client:
            await self.current_client.stop()
            self.current_client = None
            self.current_params = None
            logger.info("Current stream stopped")
    
    async def start_server(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        
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

def create_app(frame_processor: Optional[Callable[[Union[VideoFrame, AudioFrame]], Union[VideoOutput, AudioOutput]]] = None) -> TrickleApp:
    """Create a trickle app instance."""
    return TrickleApp(frame_processor) 