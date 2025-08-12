"""
TrickleStreamHandler - Complete trickle stream handler with protocol management built-in.

This module provides a high-level stream handler that encapsulates all the
trickle protocol setup, event management, and task lifecycle management
that would otherwise need to be duplicated across applications.
"""

import asyncio
import json
import logging
from abc import abstractmethod
from typing import Optional, Dict, Any, Callable, Union

from . import ErrorCallback
from .frame_processor import FrameProcessor
from .client import TrickleClient
from .protocol import TrickleProtocol
from .subscriber import TrickleSubscriber
from .manager import StreamHandler

logger = logging.getLogger(__name__)


class TrickleStreamHandler(StreamHandler):
    """Complete trickle stream handler with protocol management built-in.
    
    This class handles all the common trickle streaming patterns:
    - Protocol setup and configuration
    - Event management (running, shutdown, error)
    - Task lifecycle management
    - Control channel handling
    - Monitoring and cleanup
    
    Applications only need to implement:
    - create_frame_processor(): Return the FrameProcessor instance
    - handle_control_message(): Handle control channel messages
    """
    
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: str = "",
        events_url: str = "",
        data_url: str = "",
        width: int = 512,
        height: int = 512,
        error_callback: Optional[ErrorCallback] = None,
        app_context: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize TrickleStreamHandler.
        
        Args:
            subscribe_url: URL to subscribe to input stream
            publish_url: URL to publish output stream
            control_url: Optional URL for control channel
            events_url: Optional URL for events/monitoring
            data_url: Optional URL for data publishing
            width: Stream width in pixels
            height: Stream height in pixels
            error_callback: Optional callback for error handling
            app_context: Optional application context dict
            **kwargs: Additional arguments passed to StreamHandler
        """
        super().__init__(width=width, height=height, **kwargs)
        
        # Store URLs and configuration
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.data_url = data_url
        self.app_context = app_context or {}
        
        # Event management
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Task management
        self._task: Optional[asyncio.Task] = None
        self._control_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Error handling
        self._error_callback = error_callback
        self._critical_error_occurred = False
        self._cleanup_lock = asyncio.Lock()
        
        # Trickle components (initialized in start())
        self.protocol: Optional[TrickleProtocol] = None
        self.client: Optional[TrickleClient] = None
        self.control_subscriber: Optional[TrickleSubscriber] = None
        
    @property
    def running(self) -> bool:
        """Whether the stream handler is currently running."""
        return (self.running_event.is_set() and 
                not self.shutdown_event.is_set() and 
                not self.error_event.is_set())
    
    @abstractmethod
    async def create_frame_processor(self) -> 'FrameProcessor':
        """Create the async frame processor.
        
        This method must be implemented by subclasses to provide
        the actual async frame processing logic.
        
        Returns:
            FrameProcessor instance for native async processing
        """
        pass
    
    @abstractmethod
    async def handle_control_message(self, params: Dict[str, Any]):
        """Handle control channel messages.
        
        This method must be implemented by subclasses to handle
        application-specific control messages like parameter updates.
        
        Args:
            params: Dictionary containing control message parameters
        """
        pass
    
    async def _on_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle critical errors that require stream shutdown."""
        if self.shutdown_event.is_set():
            return
            
        logger.error(f"Critical error in trickle stream: {error_type} - {exception}")
        self._critical_error_occurred = True
        
        if self.running:
            self.error_event.set()
            # Schedule cleanup without blocking
            asyncio.create_task(self.stop())
        
        # Call external error callback if provided
        if self._error_callback:
            try:
                if asyncio.iscoroutinefunction(self._error_callback):
                    await self._error_callback(error_type, exception)
                else:
                    self._error_callback(error_type, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    async def _emit_monitoring_event(self, data: Dict[str, Any], event_type: str):
        """Emit monitoring event if events URL is configured."""
        if not self.events_url or not self.events_url.strip():
            return
        if not self.protocol:
            return
            
        try:
            await self.protocol.emit_monitoring_event(data, event_type)
        except Exception as e:
            logger.warning(f"Failed to emit {event_type} event: {e}")
    
    async def _publish_data(self, text_data: str):
        """Publish text data via the data channel."""
        if not self.data_url or not self.data_url.strip():
            return
        if not self.client:
            return
            
        try:
            await self.client.publish_data(text_data)
        except Exception as e:
            logger.warning(f"Failed to publish data: {e}")
    
    async def _control_loop(self):
        """Handle control channel messages."""
        if not self.control_subscriber:
            return
        
        keepalive_message = {"keep": "alive"}
        
        try:
            while not self.shutdown_event.is_set() and not self.error_event.is_set():
                try:
                    segment = await self.control_subscriber.next()
                    if not segment or segment.eos():
                        break
                    
                    params_data = await segment.read()
                    if not params_data:
                        continue
                    
                    try:
                        params = json.loads(params_data.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.warning(f"Invalid control message: {e}")
                        continue
                    
                    # Skip keepalive messages
                    if params == keepalive_message:
                        continue
                    
                    # Handle control message via abstract method
                    await self.handle_control_message(params)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in control loop: {e}")
                    # Brief pause to prevent tight error loops
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=0.1)
                        break
                    except asyncio.TimeoutError:
                        continue
                        
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Control loop error: {e}")
            await self._on_error("control_loop_error", e)
    
    async def _monitoring_loop(self):
        """Send periodic monitoring/stats events."""
        try:
            while self.running:
                try:
                    # Get stats from subclass implementation
                    stats = await self._get_monitoring_stats()
                    if stats:
                        await self._emit_monitoring_event(stats, "stream_stats")
                except Exception as e:
                    logger.error(f"Error sending monitoring stats: {e}")
                
                # Wait for monitoring interval or shutdown
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=20.0)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Continue with next monitoring interval
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    async def _get_monitoring_stats(self) -> Optional[Dict[str, Any]]:
        """Get monitoring statistics. Can be overridden by subclasses.
        
        Returns:
            Dictionary with monitoring stats or None to skip this interval
        """
        return {
            "type": "stream_stats", 
            "timestamp": asyncio.get_event_loop().time(),
            "stream": {
                "running": self.running,
                "width": self.width,
                "height": self.height
            }
        }
    
    def _on_client_done(self, task: asyncio.Task):
        """Handle client task completion."""
        self.shutdown_event.set()
        if task.exception():
            logger.error(f"Client task finished with exception: {task.exception()}")
        
        # Schedule cleanup
        cleanup_task = asyncio.create_task(self.stop())
        cleanup_task.add_done_callback(lambda t: None)
    
    async def start(self) -> bool:
        """Start the trickle stream with all components."""
        if self.running:
            return False
        
        try:
            # Get frame processor from subclass
            frame_processor = await self.create_frame_processor()
            if not frame_processor:
                logger.error("Failed to create frame processor")
                return False
            
            # Set up trickle protocol
            self.protocol = TrickleProtocol(
                subscribe_url=self.subscribe_url,
                publish_url=self.publish_url,
                control_url=self.control_url,
                events_url=self.events_url,
                data_url=self.data_url,
                width=self.width,
                height=self.height,
                error_callback=self._on_error
            )
            
            # Set up trickle client
            self.client = TrickleClient(
                protocol=self.protocol,
                frame_processor=frame_processor,
                error_callback=self._on_error
            )
            
            # Set up control subscriber if needed
            if self.control_url and self.control_url.strip():
                self.control_subscriber = TrickleSubscriber(
                    self.control_url, 
                    error_callback=self._on_error
                )
            
            # Emit startup event
            await self._emit_monitoring_event(
                {"type": "stream_started"}, 
                "stream_trace"
            )
            
            # Start the main client task
            self._task = asyncio.create_task(self.client.start())
            self._task.add_done_callback(self._on_client_done)
            
            # Set running state
            self.running_event.set()
            
            # Start auxiliary tasks
            if self.control_subscriber:
                try:
                    self._control_task = asyncio.create_task(self._control_loop())
                except Exception as e:
                    logger.warning(f"Failed to start control task: {e}")
            
            try:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            except Exception as e:
                logger.warning(f"Failed to start monitoring task: {e}")
            
            logger.info("TrickleStreamHandler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start TrickleStreamHandler: {e}")
            await self._cleanup_after_failure()
            return False
    
    async def stop(self, *, called_by_manager: bool = False) -> bool:
        """Stop the trickle stream with proper cleanup."""
        async with self._cleanup_lock:
            self.shutdown_event.set()
            
            try:
                # Stop the client
                if self.client:
                    try:
                        await asyncio.wait_for(self.client.stop(), timeout=5.0)
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning(f"Client stop timeout/error: {e}")
                
                # Cancel all tasks
                await self._cancel_task_with_timeout(self._task, "Main task")
                await self._cancel_task_with_timeout(self._control_task, "Control task")
                await self._cancel_task_with_timeout(self._monitoring_task, "Monitoring task")
                
                # Close control subscriber
                if self.control_subscriber:
                    try:
                        await self.control_subscriber.shutdown()
                        await self.control_subscriber.close()
                    except Exception as e:
                        logger.warning(f"Control subscriber cleanup error: {e}")
                
                # Emit final event
                try:
                    await self._emit_monitoring_event(
                        {"type": "stream_stopped", "timestamp": asyncio.get_event_loop().time()},
                        "stream_trace"
                    )
                except Exception:
                    pass
                
                # Update health manager if not called by manager
                if not called_by_manager:
                    await self._update_health_manager()
                
                self._set_final_state()
                logger.info("TrickleStreamHandler stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping TrickleStreamHandler: {e}")
                if not called_by_manager:
                    await self._update_health_manager(emergency=True)
                self._set_final_state()
                return False
    
    async def _cleanup_after_failure(self):
        """Clean up resources after a startup failure."""
        try:
            if self.client:
                try:
                    await self.client.stop()
                except Exception:
                    pass
            
            if self.control_subscriber:
                try:
                    await self.control_subscriber.close()
                except Exception:
                    pass
                    
            self._set_final_state()
        except Exception as e:
            logger.error(f"Error in cleanup after failure: {e}")
    
    async def _cancel_task_with_timeout(self, task: Optional[asyncio.Task], task_name: str, timeout: float = 3.0):
        """Cancel a task with timeout protection."""
        if not task or task.done():
            return
        
        try:
            task.cancel()
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.CancelledError:
            # Expected when cancelling
            pass
        except asyncio.TimeoutError:
            logger.warning(f"Timeout cancelling {task_name}")
        except Exception as e:
            logger.warning(f"Error cancelling {task_name}: {e}")
    
    def _set_final_state(self):
        """Set final state for the stream handler."""
        self.running_event.clear()
        self.shutdown_event.set()
        self.error_event.set()
    
    async def _update_health_manager(self, emergency: bool = False):
        """Update health manager with stream removal."""
        try:
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                if emergency:
                    health_manager.set_error("Emergency stream shutdown")
                else:
                    health_manager.clear_error()
        except Exception as e:
            logger.error(f"Error updating health manager: {e}")
