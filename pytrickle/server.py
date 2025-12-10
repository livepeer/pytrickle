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
import uuid

from aiohttp import web
from .version import __version__

from .api import StreamParamsUpdateRequest, StreamStartRequest, Version, HardwareInformation, HardwareStats
from .state import StreamState, PipelineState
from .utils.hardware import HardwareInfo
from .frame_processor import FrameProcessor
from .client import TrickleClient
from .protocol import TrickleProtocol
from .frame_skipper import FrameSkipConfig
from .frame_overlay import OverlayConfig

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
        route_prefix: str = "/",
        host: str = "0.0.0.0",
        # Startup/shutdown hooks
        on_startup: Optional[List[Callable]] = None,
        on_shutdown: Optional[List[Callable]] = None,
        # Additional aiohttp app configuration
        publisher_timeout: Optional[float] = None,
        subscriber_timeout: Optional[float] = None,
        app_kwargs: Optional[Dict[str, Any]] = None,
        # Frame skipping configuration
        frame_skip_config: Optional[FrameSkipConfig] = None,
        # Loading overlay configuration
        overlay_config: Optional['OverlayConfig'] = None,

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
            frame_skip_config: Optional frame skipping configuration (None = no frame skipping)
        """
        self.frame_processor = frame_processor
        self.port = port
        self.host = host
        self.state = StreamState()
        self.route_prefix = route_prefix
        self.enable_default_routes = enable_default_routes
        self.health_check_interval = health_check_interval
        self._startup_task: Optional[asyncio.Task] = None
        
        if isinstance(self.frame_processor, FrameProcessor):
            try:
                self.frame_processor.attach_state(self.state)
            except Exception as exc:
                logger.debug("Failed to attach stream state to frame processor: %s", exc)
        
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
        
        # Frame skipping configuration
        self.frame_skip_config = frame_skip_config
        
        # Loading overlay configuration
        self.overlay_config = overlay_config
        
        # Stream management - simple and direct
        self.current_client: Optional[TrickleClient] = None
        self.current_params: Optional[StreamStartRequest] = None
        self._client_task: Optional[asyncio.Task] = None
        self._param_update_lock = asyncio.Lock()
        
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
        

    def _start_pipeline_initialization(self) -> None:
        """Kick off background initialization to prepare the pipeline."""
        if isinstance(self.frame_processor, FrameProcessor):
            loop = asyncio.get_running_loop()
            self._startup_task = loop.create_task(self._initialize_frame_processor())
            self._startup_task.add_done_callback(self._on_startup_task_done)
        else:
            self.state.set_startup_complete()

    def _on_startup_task_done(self, task: asyncio.Task) -> None:
        """Cleanup callback for startup initialization task."""
        if task is not self._startup_task:
            return
        self._startup_task = None
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("Startup initialization task cancelled")
        except Exception as exc:
            logger.error("Startup initialization task failed: %s", exc)

    async def _initialize_frame_processor(self) -> None:
        """Ensure the frame processor is ready for incoming streams."""
        try:
            ensure_model_loaded = getattr(self.frame_processor, "ensure_model_loaded", None)
            if callable(ensure_model_loaded):
                await ensure_model_loaded()
            else:
                self.state.set_startup_complete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Failed to initialize frame processor: %s", exc)
            self.state.set_error(f"Model loading failed: {exc}")
        else:
            if not self.state.startup_complete:
                self.state.set_startup_complete()

    async def _run_param_update(self, params: Dict[str, Any]):
        """Run parameter update sequentially while tracking state."""
        params_payload = dict(params)
        try:
            # Handle manual loading overlay changes before parameter updates            
            show_loading = params_payload.get("show_loading", None)
            if show_loading is not None and self.current_client:
                
                # Parse string/bool values for manual loading overlay
                if isinstance(show_loading, str):
                    show_loading_bool = show_loading.lower() == ("true")
                else:
                    show_loading_bool = bool(show_loading)
                    
                self.current_client.loading_controller.set_manual_loading(show_loading_bool)
                logger.debug(f"Manual loading set to: {show_loading_bool} (from {show_loading})")
            
            async with self._param_update_lock:
                # Update protocol if needed
                if self.current_client and self.current_client.protocol:
                    await self.current_client.protocol.update_params(params_payload)

                await self.frame_processor.update_params(params_payload)
                
                logger.info(f"Parameters updated: {params}")

                if self.current_client and self.current_client.protocol:
                    await self.current_client.protocol.emit_monitoring_event(
                        {
                            "type": "params_updated",
                            "timestamp": int(time.time() * 1000),
                            "params": params,
                        }
                    )
        except asyncio.CancelledError:
            logger.debug("Parameter update task cancelled before completion")
            raise
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            self.state.set_error(f"Parameter update failed: {str(e)}")

    
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
            ("/app-version", self._handle_app_version),
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
            # Ensure pipeline initialization is complete before allowing stream start
            if not self.state.startup_complete or not self.state.pipeline_ready:
                startup_task = self._startup_task
                if startup_task:
                    try:
                        await asyncio.shield(startup_task)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error("Startup task failed before stream start: %s", exc)

                if not self.state.startup_complete or not self.state.pipeline_ready:
                    logger.warning("Stream start rejected - pipeline not ready yet")
                    return web.json_response({
                        "status": "error",
                        "message": "Pipeline is still initializing. Please wait for IDLE status.",
                        "current_state": self.state.get_pipeline_state()
                    }, status=503)  # 503 Service Unavailable
            
            # Parse and validate request first
            params = await self._parse_and_validate_request(request, StreamStartRequest)
            
            logger.info(f"Starting stream: {params.subscribe_url} -> {params.publish_url}")
            
            # Check if a stream is already active
            if self.current_client and self.current_client.running:
                return web.json_response({
                        "status": "error",
                        "message": "Pipeline is still in use",
                        "current_state": self.state.get_pipeline_state()
                    }, status=503)  # 503 Service Unavailable
            
            # Extract dimensions from params
            params_dict = params.params or {}
            width = params_dict.get("width", 512)
            height = params_dict.get("height", 512)
            max_framerate = params_dict.get("max_framerate", None)
            # Auto-detect output resolution is enabled by default unless explicitly disabled
            detect_out_resolution = params_dict.get("detect_out_resolution", True)
            
            # Create protocol
            protocol = TrickleProtocol(
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
                detect_out_resolution=detect_out_resolution,
            )
            
            # Create client
            self.current_client = TrickleClient(
                protocol=protocol,
                frame_processor=self.frame_processor,
                control_handler=self._handle_control_message,
                frame_skip_config=self.frame_skip_config,
                overlay_config=self.overlay_config,
            )
            
            # Update state
            self.current_params = params
            self.state.set_active_client(True)
            self.state.update_active_streams(1)
            
            # Start client in background (params will be passed to on_stream_start)
            # Note: We pass params via client.start instead of via update_params
            # to differentiate stream start from parameter updates
            self._client_task = asyncio.create_task(
                self._run_client(params.gateway_request_id, params.params)
            )
            
            # Start health monitoring
            self._start_health_monitoring()
            
            return web.json_response({
                "status": "success",
                "message": "Stream started successfully",
                "request_id": params.gateway_request_id,
                "current_state": self.state.get_pipeline_state()
            })
            
        except Exception as e:
            logger.error(f"Error in start stream handler: {e}")

            # Ensure we don't leave the server in a partially active state
            if self._client_task:
                self._client_task.cancel()
                try:
                    await self._client_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._client_task = None

            if self.current_client:
                try:
                    await self.current_client.stop()
                except Exception as stop_exc:
                    logger.debug("Failed to stop client after start error: %s", stop_exc)
                self.current_client = None

            self.current_params = None
            self.state.set_active_client(False)
            self.state.update_active_streams(0)

            return web.json_response({
                "status": "error",
                "message": f"Error starting stream: {str(e)}",
                "current_state": self.state.get_pipeline_state()
            }, status=400)
    
    async def _handle_stop_stream(self, request: web.Request) -> web.Response:
        """Handle stream stop requests."""
        try:
            if not self.current_client:
                return web.json_response({
                    "status": "error",
                    "message": "No active stream to stop"
                }, status=400)
            
            # Emit stop event before stopping
            if self.current_client.protocol:
                await self.current_client.protocol.emit_monitoring_event({
                    "type": "stream_stopped",
                    "timestamp": int(time.time() * 1000)
                })
            
            # Stop client
            await self.current_client.stop()
            
            # Cancel background task
            if self._client_task:
                self._client_task.cancel()
                try:
                    await self._client_task
                except asyncio.CancelledError:
                    pass
                self._client_task = None
            
            # Update state
            self.current_client = None
            self.current_params = None
            self.state.set_active_client(False)
            self.state.update_active_streams(0)
            
            return web.json_response({
                "status": "success",
                "message": "Stream stopped successfully"
            })
            
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
            await self.frame_processor.update_params(control_data)
            logger.debug("Control message routed to frame processor")
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
    
    async def _handle_update_params(self, request: web.Request) -> web.Response:
        """Handle parameter update requests."""
        try:
            # Parse and validate request
            validated_params = await self._parse_and_validate_request(request, StreamParamsUpdateRequest)
            data = validated_params.model_dump()
            
            if not self.current_client:
                return web.json_response({
                    "status": "error",
                    "message": "No active stream to update"
                }, status=400)
            
            # Update frame processor parameters
            await self._run_param_update(data)
            logger.info(f"Parameters updated: {data}")
            
            # Emit monitoring event
            if self.current_client.protocol:
                await self.current_client.protocol.emit_monitoring_event({
                    "type": "params_updated",
                    "timestamp": int(time.time() * 1000),
                    "params": data
                })
            
            return web.json_response({
                "status": "success",
                "message": "Parameters updated successfully"
            })
            
        except Exception as e:
            logger.error(f"Error in update params handler: {e}")
            return web.json_response({
                "status": "error",
                "message": f"Parameter update failed: {str(e)}"
            }, status=500)
    
    async def _handle_get_status(self, request: web.Request) -> web.Response:
        """Handle status requests with detailed state and frame processing statistics."""
        try:
            # Get base state
            status_data = self.state.get_pipeline_state()
            
            # Add client information
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
                    "gateway_request_id": self.current_params.gateway_request_id
                }
            
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
        """Handle version requests - returns package version."""
        return web.json_response(Version(
            pipeline=self.pipeline,
            model_id=self.capability_name,
            version=__version__
        ).model_dump())
    
    async def _handle_app_version(self, request: web.Request) -> web.Response:
        """Handle app version requests - returns customizable application version."""
        return web.json_response({
            "version": self.version
        })
    
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
    
    async def _run_client(self, request_id: str, params: Optional[Dict[str, Any]] = None):
        """Run the trickle client in background.
        
        Args:
            request_id: Unique identifier for the stream request
            params: Optional parameters to pass to on_stream_start callback
        """
        try:
            if self.current_client:
                await self.current_client.start(request_id, params)
                logger.info("Client stream completed successfully")
        except Exception as e:
            logger.error(f"Error running client: {e}")
            self.state.set_error(f"Client execution failed: {str(e)}")
        finally:
            # Clean up when client ends
            self.state.set_active_client(False)
            self.state.update_active_streams(0)
    
    async def stop(self):
        """Stop the server and clean up resources."""
        # Stop current client if running
        if self.current_client:
            await self.current_client.stop()
        
        # Cancel client task
        if self._client_task:
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass
        
        # Cancel health monitoring task  
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._startup_task:
            self._startup_task.cancel()
            try:
                await self._startup_task
            except asyncio.CancelledError:
                pass
            self._startup_task = None
    
    async def start_server(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self._start_pipeline_initialization()

        logger.info(f"Server started on {self.host}:{self.port}")
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
