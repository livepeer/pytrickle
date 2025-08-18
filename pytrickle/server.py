"""
Trickle App HTTP Server.

Provides a REST API for managing trickle video streams with support for
starting streams, updating parameters, and monitoring status.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass
import os

from aiohttp import web

from .client import TrickleClient
from .protocol import TrickleProtocol
from .api import StreamParamsUpdateRequest, StreamStartRequest, Version, HardwareInformation, HardwareStats
from .state import StreamState, PipelineState
from .utils.hardware import HardwareInfo
from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

@dataclass
class RouteConfig:
    """Configuration for custom routes."""
    method: str
    path: str
    handler: Callable
    kwargs: Optional[Dict[str, Any]] = None


class StreamServer:
    """HTTP server application for trickle streaming with native async processor support."""
    
    def __init__(
        self, 
        frame_processor: 'FrameProcessor',
        port: int = 8000,
        pipeline: str = "",
        capability_name: str = "",
        version: str = "0.0.1",
        # New configurability options
        custom_routes: Optional[List[Union[RouteConfig, Dict[str, Any]]]] = None,
        middleware: Optional[List[Callable]] = None,
        cors_config: Optional[Dict[str, Any]] = None,
        static_routes: Optional[List[Dict[str, Any]]] = None,
        app_context: Optional[Dict[str, Any]] = None,
        health_check_interval: float = 5.0,
        enable_default_routes: bool = True,
        route_prefix: str = "/api",
        host: str = "0.0.0.0",
        # Startup/shutdown hooks
        on_startup: Optional[List[Callable]] = None,
        on_shutdown: Optional[List[Callable]] = None,
        # Additional aiohttp app configuration
        app_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize StreamServer.
        
        Args:
            frame_processor: FrameProcessor for native async processing
            port: HTTP server port
            capability_name: Name of the capability
            version: Version string
            custom_routes: List of custom routes to add (RouteConfig objects or dicts)
            middleware: List of aiohttp middleware functions
            cors_config: CORS configuration dict (requires aiohttp-cors)
            static_routes: List of static route configurations
            app_context: Additional context to store in the app
            health_check_interval: Interval for health monitoring in seconds
            enable_default_routes: Whether to enable default streaming routes
            route_prefix: Prefix for default routes
            host: Host to bind the server to
            on_startup: List of startup handlers
            on_shutdown: List of shutdown handlers
            app_kwargs: Additional kwargs for aiohttp.web.Application
        """
        self.frame_processor = frame_processor
        self.port = port
        self.host = host
        self.state = StreamState()
        self.route_prefix = route_prefix
        self.enable_default_routes = enable_default_routes
        self.health_check_interval = health_check_interval
        
        # Create aiohttp application with optional configuration
        app_config = app_kwargs or {}
        self.app = web.Application(**app_config)
        
        # Store configuration
        self.current_client: Optional[TrickleClient] = None
        self.current_params: Optional[StreamStartRequest] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._client_task: Optional[asyncio.Task] = None
        self.version = version
        self.hardware_info = HardwareInfo()
        self.pipeline = pipeline or os.getenv("PIPELINE", "byoc")
        self.capability_name = capability_name or os.getenv("CAPABILITY_NAME", os.getenv("MODEL_ID",""))

        # Store app context
        if app_context:
            for key, value in app_context.items():
                self.app[key] = value

        # Setup middleware first (order matters)
        if middleware:
            self._setup_middleware(middleware)
        
        # Setup CORS if configured
        if cors_config:
            self._setup_cors(cors_config)
        
        # Setup startup/shutdown handlers
        if on_startup:
            for handler in on_startup:
                self.app.on_startup.append(handler)
        if on_shutdown:
            for handler in on_shutdown:
                self.app.on_shutdown.append(handler)

        # Setup routes
        if enable_default_routes:
            self._setup_routes()
        
        # Setup custom routes
        if custom_routes:
            self._setup_custom_routes(custom_routes)
        
        # Setup static routes
        if static_routes:
            self._setup_static_routes(static_routes)
        
        # Serialize start/stop operations to avoid overlapping lifecycle transitions
        self._lifecycle_lock = asyncio.Lock()
    
    def _setup_middleware(self, middleware_list: List[Callable]):
        """Setup middleware for the aiohttp application."""
        for middleware_func in middleware_list:
            self.app.middlewares.append(middleware_func)
    
    def _setup_cors(self, cors_config: Dict[str, Any]):
        """Setup CORS configuration."""
        try:
            from aiohttp_cors import setup as setup_cors, ResourceOptions
            
            # Convert dict config to ResourceOptions if needed
            defaults = {}
            for origin, config in cors_config.items():
                if isinstance(config, dict):
                    defaults[origin] = ResourceOptions(**config)
                else:
                    defaults[origin] = config
            
            cors = setup_cors(self.app, defaults=defaults)
            
            # Store cors object for later use (e.g., adding routes after setup)
            self.app['cors'] = cors
            
        except ImportError:
            logger.warning("aiohttp-cors not installed, CORS configuration ignored")
    
    def _setup_custom_routes(self, routes: List[Union[RouteConfig, Dict[str, Any]]]):
        """Setup custom routes."""
        for route_config in routes:
            if isinstance(route_config, dict):
                # Convert dict to RouteConfig
                kwargs = route_config.pop('kwargs', {})
                route = RouteConfig(**route_config, kwargs=kwargs)
            else:
                route = route_config
            
            # Add the route
            route_kwargs = route.kwargs or {}
            route_resource = self.app.router.add_route(
                route.method, 
                route.path, 
                route.handler,
                **route_kwargs
            )
            
            # Add to CORS if configured
            if 'cors' in self.app:
                self.app['cors'].add(route_resource)
    
    def _setup_static_routes(self, static_routes: List[Dict[str, Any]]):
        """Setup static file routes."""
        for static_config in static_routes:
            prefix = static_config['prefix']
            path = static_config['path']
            kwargs = static_config.get('kwargs', {})
            
            static_resource = self.app.router.add_static(prefix, path, **kwargs)
            
            # Add to CORS if configured
            if 'cors' in self.app:
                self.app['cors'].add(static_resource)

    def _default_hardware_info(self) -> Dict[str, Any]:
        """Return default hardware info."""
        return self.hardware_info.get_gpu_compute_info()
    
    def _setup_routes(self):
        """Setup default HTTP routes."""
        prefix = self.route_prefix.rstrip('/')
        
        # Core streaming routes
        stream_routes = [
            (f"{prefix}/stream/start", self._handle_start_stream),
            (f"{prefix}/stream/stop", self._handle_stop_stream), 
            (f"{prefix}/stream/params", self._handle_update_params),
            (f"{prefix}/stream/status", self._handle_get_status),
        ]
        
        # System routes
        system_routes = [
            ("/health", self._handle_health),
            ("/version", self._handle_version),
            ("/hardware/info", self._handle_hardware_info),
            ("/hardware/stats", self._handle_hardware_stats),
        ]
        
        # Compatibility routes
        compat_routes = [
            ("/live-video-to-video", self._handle_start_stream),  # Alias for stream/start
        ]
        
        # Add all routes
        all_routes = stream_routes + system_routes + compat_routes
        for path, handler in all_routes:
            if path.endswith(('start', 'stop', 'params', 'live-video-to-video')):
                route_resource = self.app.router.add_post(path, handler)
            else:
                route_resource = self.app.router.add_get(path, handler)
            
            # Add to CORS if configured
            if 'cors' in self.app:
                self.app['cors'].add(route_resource)
    
    # Add new public methods for dynamic configuration
    def add_route(self, method: str, path: str, handler: Callable, **kwargs) -> web.AbstractRoute:
        """Add a route dynamically after initialization."""
        route_resource = self.app.router.add_route(method, path, handler, **kwargs)
        
        # Add to CORS if configured
        if 'cors' in self.app:
            self.app['cors'].add(route_resource)
            
        return route_resource
    
    def add_static_route(self, prefix: str, path: str, **kwargs) -> web.AbstractRoute:
        """Add a static route dynamically after initialization."""
        static_resource = self.app.router.add_static(prefix, path, **kwargs)
        
        # Add to CORS if configured
        if 'cors' in self.app:
            self.app['cors'].add(static_resource)
            
        return static_resource
    
    def get_app(self) -> web.Application:
        """Get the underlying aiohttp application for advanced customization."""
        return self.app
    
    def add_middleware(self, middleware: Callable):
        """Add middleware dynamically (must be called before server start)."""
        self.app.middlewares.append(middleware)
    
    def set_context(self, key: str, value: Any):
        """Set a value in the app context."""
        self.app[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the app context."""
        return self.app.get(key, default)

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
            # Parse and validate request first
            params = await self._parse_and_validate_request(request, StreamStartRequest)
            
            logger.info(f"Starting stream: {params.subscribe_url} -> {params.publish_url}")
            
            # Extract dimensions and framerate from params (already converted to int by validation)
            params_dict = params.params or {}
            width = params_dict.get("width", 512)
            height = params_dict.get("height", 512)
            max_framerate = params_dict.get("max_framerate", None)  # None will use default
            
            async with self._lifecycle_lock:
                # Create new protocol for the stream
                new_protocol = TrickleProtocol(
                    subscribe_url=params.subscribe_url,
                    publish_url=params.publish_url,
                    control_url=params.control_url or "",
                    events_url=params.events_url or "",
                    data_url=params.data_url,
                    width=width,
                    height=height,
                    max_framerate=max_framerate,
                )
                
                # Reuse existing client or create new one if none exists
                if self.current_client is not None:
                    logger.info("Reusing existing TrickleClient with new protocol")
                    # If client is currently running, stop the current stream cleanly first
                    if self.current_client.running:
                        logger.info("Client already running - stopping current stream before restart")
                        await self._stop_current_stream()
                    
                    # Stop current protocol before swapping to avoid stopping an unstarted protocol later
                    try:
                        if self.current_client.protocol:
                            await self.current_client.protocol.stop()
                            logger.info("Stopped previous protocol")
                    except Exception as e:
                        logger.warning(f"Error stopping previous protocol (continuing): {e}")

                    # Replace protocol and reset client state for new stream
                    self.current_client.protocol = new_protocol
                    
                    # Reset client coordination events for clean start
                    self.current_client.stop_event.clear()
                    self.current_client.error_event.clear()
                    self.current_client.running = False
                    
                    # Clear output queue of any stale data from previous stream
                    while not self.current_client.output_queue.empty():
                        try:
                            self.current_client.output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    
                    logger.info("Client state reset for new stream")
                else:
                    logger.info("Creating new TrickleClient with native async processor")
                    self.current_client = TrickleClient(
                        protocol=new_protocol,
                        frame_processor=self.frame_processor,
                        control_handler=self._handle_control_message,
                    )
        
                # Update current params
                self.current_params = params
                
                # Track active client and start health monitoring
                self.state.set_active_client(True)
                # Update unified state for active streams
                self.state.update_active_streams(1)
                self.state.set_state(PipelineState.READY)
                self._start_health_monitoring()
                
                # Start the client in background
                self._client_task = asyncio.create_task(self._run_client(params.gateway_request_id))
            
            # Set params via update_params if provided in start request
            try:
                logger.info("Setting params from start stream request")
                self.frame_processor.update_params(params.params)
                logger.info("Params set successfully from start request")
            except Exception as e:
                logger.warning(f"Failed to set params from start request: {e}")
            
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
            self.state.set_state(PipelineState.ERROR)
            return web.json_response({
                "status": "error",
                "message": f"Error starting stream: {str(e)}"
            }, status=400)
    
    async def _handle_stop_stream(self, request: web.Request) -> web.Response:
        """Handle stream stop requests."""
        try:
            async with self._lifecycle_lock:
                if not self.current_client:
                    return web.json_response({
                        "status": "error",
                        "message": "No active stream to stop"
                    }, status=400)
                
                await self._stop_current_stream()
            
            # Update unified state for zero active streams
            self.state.update_active_streams(0)

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
    
    async def _handle_control_message(self, control_data: dict):
        """Handle control messages from trickle protocol.
        
        Routes control messages to the frame processor's update_params method.
        """
        try:
            self.frame_processor.update_params(control_data)
            logger.debug("Control message routed to frame processor")
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
    
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
            status_data = self.state.get_pipeline_state()
            
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
        """Handle health check requests using unified StreamState."""
        try:
            health_state = self.state.get_pipeline_state()
            status_code = 503 if self.state.is_error() else 200
            return web.json_response(health_state, status=status_code)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }, status=500)
    
    async def _handle_version(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response(Version(
            pipeline=self.pipeline,
            model_id=self.capability_name,
            version=self.version
        ).model_dump())
    
    async def _handle_hardware_info(self, request: web.Request) -> web.Response:
        """Handle hardware info requests."""

        return web.json_response(HardwareInformation(
            pipeline=self.pipeline,
            model_id=self.capability_name,
            gpu_info=self._default_hardware_info()
        ).model_dump())

    async def _handle_hardware_stats(self, request: web.Request) -> web.Response:
        """Handle hardware stats requests."""
        return web.json_response(HardwareStats(
            pipeline=self.pipeline,
            model_id=self.capability_name,
            gpu_stats=self.hardware_info.get_gpu_utilization_stats()
        ).model_dump())
    
    async def _run_client(self, request_id: str):
        """Run the trickle client."""
        try:
            if self.current_client:
                # Set pipeline ready when client starts successfully
                self.state.set_state(PipelineState.READY)
                await self.current_client.start(request_id)
                
                logger.info("Client started with frame processor coordination")
        except Exception as e:
            logger.error(f"Error running client: {e}")
            # Set error state on client failure
            self.state.set_state(PipelineState.ERROR)
        finally:
            # When protocol ends, ensure frame processor stops processing
            logger.info("Protocol ended - stopping frame processor")
            # Properly stop the current stream
            async with self._lifecycle_lock:
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
                await asyncio.sleep(self.health_check_interval)
                
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
        """Stop the current stream protocol while preserving client."""
        if self.current_client:
            logger.info("Stopping current stream protocol...")
            
            # Stop the client first to ensure clean shutdown
            if self.current_client.running:
                await self.current_client.stop()
                logger.info("Client stopped")
            
            # Stop the protocol if it exists
            if self.current_client.protocol:
                await self.current_client.protocol.stop()
                logger.info("Protocol stopped, client preserved")
            
            logger.info("FrameProcessor returned to idle state")

            
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
        """Fully clean up client and frame processor with complete shutdown."""
        if self.current_client:
            await self.current_client.stop()
            # TODO: Does client.stop() handle the frame processor? Does it need to?
        
        # Clear client references on full shutdown
        self.current_client = None
        self.current_params = None
        self.state.set_active_client(False)
    
    async def start_server(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        # Set pipeline ready when server is up and ready to accept requests
        self.state.set_state(PipelineState.READY)
        
        logger.info(f"Trickle app server started on {self.host}:{self.port}")
        return runner

    # Built-in State Management API
    def set_state(self, state: PipelineState) -> None:
        """Set pipeline state using enum value.
        
        Delegates to the StreamState's set_state method for consistent state management.
        
        Args:
            state: PipelineState enum value indicating desired state
        """
        self.state.set_state(state)
    
    def set_client_active(self, active: bool) -> None:
        """Set whether there's an active streaming client.
        
        Args:
            active: True if client is active, False otherwise
        """
        self.state.set_active_client(active)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state information.
        
        Returns:
            Dict containing current state information
        """
        return self.state.get_state()
    
    def update_active_streams(self, count: int):
        """Update active stream count."""
        self.state.update_active_streams(count)
    
    def get_health_state(self) -> Dict[str, Any]:
        """Get current health state."""
        return self.state.get_pipeline_state()
    
    def is_healthy(self) -> bool:
        """Check if the server is in a healthy state."""
        return not self.state.is_error()
    
    def get_health_summary(self) -> str:
        """Get a simple health status string."""
        state = self.state.get_pipeline_state()
        return state.get('state', 'unknown')

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
            await self.stop()
            await runner.cleanup()
