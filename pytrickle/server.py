"""
Trickle App HTTP Server.

Provides a REST API for managing trickle video streams with support for
starting streams, updating parameters, and monitoring status.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass
import os

from aiohttp import web


from .api import StreamParamsUpdateRequest, StreamStartRequest, Version, HardwareInformation, HardwareStats
from .state import StreamState, PipelineState
from .utils.hardware import HardwareInfo
from .frame_processor import FrameProcessor
from .lifecycle import StreamLifecycleManager, DefaultProtocolFactory, DefaultClientFactory, ProtocolFactory, ClientFactory

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
        publisher_timeout: Optional[float] = None,
        subscriber_timeout: Optional[float] = None,
        app_kwargs: Optional[Dict[str, Any]] = None,
        # Testability options
        protocol_factory: Optional[ProtocolFactory] = None,
        client_factory: Optional[ClientFactory] = None
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
        self.version = version
        self.hardware_info = HardwareInfo()
        self.pipeline = pipeline or os.getenv("PIPELINE", "byoc")
        self.capability_name = capability_name or os.getenv("CAPABILITY_NAME", os.getenv("MODEL_ID",""))
        self.publisher_timeout = publisher_timeout
        self.subscriber_timeout = subscriber_timeout
        
        # Create lifecycle manager with dependency injection
        self.lifecycle_manager = StreamLifecycleManager(
            frame_processor=frame_processor,
            state=self.state,
            protocol_factory=protocol_factory or DefaultProtocolFactory(),
            client_factory=client_factory or DefaultClientFactory(),
            publisher_timeout=publisher_timeout,
            subscriber_timeout=subscriber_timeout,
        )
        
        # Health monitoring
        self._health_monitor_task: Optional[asyncio.Task] = None

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
            
            # Delegate to lifecycle manager
            result = await self.lifecycle_manager.start_stream(params)
            
            if result.success:
                # Start health monitoring
                self._start_health_monitoring()
                
                return web.json_response({
                    "status": "success",
                    "message": result.message,
                    "request_id": result.request_id
                })
            else:
                return web.json_response({
                    "status": "error",
                    "message": result.message
                }, status=400)
            
        except Exception as e:
            logger.error(f"Error in start stream handler: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error starting stream: {str(e)}"
            }, status=400)
    
    async def _handle_stop_stream(self, request: web.Request) -> web.Response:
        """Handle stream stop requests."""
        try:
            # Delegate to lifecycle manager
            result = await self.lifecycle_manager.stop_stream()
            
            if result.success:
                return web.json_response({
                    "status": "success",
                    "message": result.message
                })
            else:
                return web.json_response({
                    "status": "error",
                    "message": result.message
                }, status=400)
            
        except Exception as e:
            logger.error(f"Error in stop stream handler: {e}")
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
            # Parse and validate request
            validated_params = await self._parse_and_validate_request(request, StreamParamsUpdateRequest)
            data = validated_params.model_dump()
            
            # Delegate to lifecycle manager
            result = await self.lifecycle_manager.update_params(data)
            
            if result.success:
                return web.json_response({
                    "status": "success",
                    "message": result.message
                })
            else:
                return web.json_response({
                    "status": "error",
                    "message": result.message
                }, status=400 if "No active stream" in result.message else 500)
            
        except Exception as e:
            logger.error(f"Error in update params handler: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error updating parameters: {str(e)}"
            }, status=500)
    
    async def _handle_get_status(self, request: web.Request) -> web.Response:
        """Handle status requests with detailed state and frame processing statistics."""
        try:
            # Delegate to lifecycle manager
            status_data = self.lifecycle_manager.get_status()
            return web.json_response(status_data)
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Error getting status: {str(e)}"
            }, status=500)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check requests for container orchestration.
        
        Returns exactly the format expected by Kubernetes/Docker health checks:
        - HTTP 200 with {"status": "LOADING"|"OK"|"ERROR"|"IDLE"}
        - HTTP 500 with {"detail": {"msg": "Failed to retrieve pipeline status."}}
        """
        try:
            # Get current status based on lifecycle state
            if self.state.error_event.is_set():
                status = "ERROR"
            elif self.state.active_streams > 0 or self.state.active_client:
                status = "OK"  # Active streams/client = OK
                                # Debug logging to understand why we're getting OK
                logger.info(f"Health: OK - active_streams={self.state.active_streams}, active_client={self.state.active_client}")
            elif self.state.pipeline_ready and self.state.startup_complete:
                status = "IDLE"  # Ready but no activity = IDLE
                logger.info(f"Health: IDLE - pipeline_ready={self.state.pipeline_ready}, startup_complete={self.state.startup_complete}")
            else:
                status = "LOADING"  # Still starting up
                logger.info(f"Health: LOADING - pipeline_ready={self.state.pipeline_ready}, startup_complete={self.state.startup_complete}")
            
            # Return simple format for orchestration
            return web.json_response({"status": status})
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return web.json_response({
                "detail": {
                    "msg": "Failed to retrieve pipeline status."
                }
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
    
    def _start_health_monitoring(self):
        """Start background health monitoring task."""
        if self._health_monitor_task and not self._health_monitor_task.done():
            return
        self._health_monitor_task = asyncio.create_task(self._health_monitoring_loop())

    async def _health_monitoring_loop(self):
        """Background task to monitor component health."""
        try:
            while self.lifecycle_manager.current_client:
                await asyncio.sleep(self.health_check_interval)
                
                client = self.lifecycle_manager.current_client
                if client and client.protocol:
                    protocol = client.protocol
                    self.state.update_component_health(
                        protocol.component_name,
                        protocol.get_component_health()
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
    async def stop(self):
        """Stop the server and clean up resources."""
        # Delegate to lifecycle manager for proper cleanup
        await self.lifecycle_manager.stop_stream()
        
        # Cancel health monitoring task  
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
    
    async def start_server(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        # Set pipeline ready when server is up and ready to accept requests
        self.state.set_state(PipelineState.IDLE)
        self.state.set_startup_complete()
        
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
