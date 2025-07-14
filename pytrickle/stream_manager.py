"""
Stream management utilities for trickle streaming.

Provides high-level stream management functionality that can be reused
across different applications and frameworks.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass

from .client import TrickleClient
from .frames import VideoFrame, VideoOutput
from .exceptions import ErrorPropagator

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for a trickle stream."""
    subscribe_url: str
    publish_url: str
    control_url: Optional[str] = None
    events_url: Optional[str] = None
    width: int = 512
    height: int = 512
    max_retries: int = 3


class StreamHandler:
    """Base class for handling a single trickle stream."""
    
    def __init__(
        self,
        request_id: str,
        config: StreamConfig,
        frame_processor: Optional[Callable[[VideoFrame], VideoOutput]] = None,
        app_context: Optional[Dict[str, Any]] = None
    ):
        self.request_id = request_id
        self.config = config
        self.frame_processor = frame_processor or self._default_frame_processor
        self.app_context = app_context or {}
        self.running = False
        self.client: Optional[TrickleClient] = None
        self.error_propagator = ErrorPropagator()
        self._task: Optional[asyncio.Task] = None
        
    def _default_frame_processor(self, frame: VideoFrame) -> VideoOutput:
        """Default frame processor that passes frames through unchanged."""
        return VideoOutput(frame, self.request_id)
    
    async def start(self) -> bool:
        """Start the stream handler."""
        if self.running:
            logger.warning(f"Stream {self.request_id} is already running")
            return False
        
        try:
            logger.info(f"Starting stream handler for {self.request_id}")
            
            # Create trickle client
            self.client = TrickleClient(
                subscribe_url=self.config.subscribe_url,
                publish_url=self.config.publish_url,
                control_url=self.config.control_url,
                events_url=self.config.events_url,
                width=self.config.width,
                height=self.config.height,
                frame_processor=self.frame_processor
            )
            
            # Start the client in background
            self._task = asyncio.create_task(self._run_client())
            self.running = True
            
            logger.info(f"Stream handler started successfully for {self.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream handler {self.request_id}: {e}")
            self.running = False
            return False
    
    async def stop(self) -> bool:
        """Stop the stream handler."""
        if not self.running:
            return True
        
        try:
            logger.info(f"Stopping stream handler for {self.request_id}")
            
            # Stop the client
            if self.client:
                await self.client.stop()
                self.client = None
            
            # Cancel the task
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
            
            self.running = False
            logger.info(f"Stream handler stopped for {self.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream handler {self.request_id}: {e}")
            self.running = False
            return False
    
    async def _run_client(self):
        """Run the trickle client."""
        try:
            if self.client:
                await self.client.start(self.request_id)
        except Exception as e:
            logger.error(f"Error running client for {self.request_id}: {e}")
            # Propagate error to trigger shutdown
            await self.error_propagator.propagate_error(e, f"stream_handler:{self.request_id}")
        finally:
            self.client = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stream handler."""
        return {
            'request_id': self.request_id,
            'running': self.running,
            'subscribe_url': self.config.subscribe_url,
            'publish_url': self.config.publish_url,
            'control_url': self.config.control_url,
            'events_url': self.config.events_url,
            'width': self.config.width,
            'height': self.config.height,
            'has_client': self.client is not None,
            'has_task': self._task is not None and not self._task.done()
        }


class StreamManager:
    """Generic stream manager for handling multiple trickle streams."""
    
    def __init__(self, stream_handler_class: type = StreamHandler):
        self.stream_handler_class = stream_handler_class
        self.handlers: Dict[str, StreamHandler] = {}
        self.lock = asyncio.Lock()
    
    async def create_stream(
        self,
        request_id: str,
        config: StreamConfig,
        frame_processor: Optional[Callable[[VideoFrame], VideoOutput]] = None,
        app_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create and start a new stream."""
        async with self.lock:
            if request_id in self.handlers:
                logger.warning(f"Stream {request_id} already exists")
                return False
            
            try:
                handler = self.stream_handler_class(
                    request_id=request_id,
                    config=config,
                    frame_processor=frame_processor,
                    app_context=app_context
                )
                
                success = await handler.start()
                if success:
                    self.handlers[request_id] = handler
                    logger.info(f"Created and started stream {request_id}")
                    return True
                else:
                    logger.error(f"Failed to start stream {request_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        """Stop and remove a stream."""
        async with self.lock:
            if request_id not in self.handlers:
                logger.warning(f"Stream {request_id} not found")
                return False
            
            handler = self.handlers[request_id]
            success = await handler.stop()
            
            if success:
                del self.handlers[request_id]
                logger.info(f"Stopped and removed stream {request_id}")
            
            return success
    
    async def get_stream_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stream."""
        async with self.lock:
            if request_id not in self.handlers:
                return None
            
            return self.handlers[request_id].get_status()
    
    async def list_streams(self) -> Dict[str, Dict[str, Any]]:
        """List all active streams."""
        async with self.lock:
            return {
                request_id: handler.get_status()
                for request_id, handler in self.handlers.items()
            }
    
    async def cleanup_all(self):
        """Stop and cleanup all streams."""
        async with self.lock:
            if not self.handlers:
                logger.info("No streams to clean up")
                return
                
            logger.info(f"Cleaning up {len(self.handlers)} streams...")
            
            # Stop all streams concurrently
            cleanup_tasks = []
            for request_id in list(self.handlers.keys()):
                cleanup_tasks.append(self._stop_stream_with_timeout(request_id))
            
            # Wait for all cleanup tasks
            try:
                results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                logger.info(f"Successfully cleaned up {successful}/{len(results)} streams")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            
            # Force clear all handlers
            self.handlers.clear()
            logger.info("All streams cleaned up")
    
    async def _stop_stream_with_timeout(self, request_id: str) -> bool:
        """Stop a single stream with timeout."""
        try:
            if request_id in self.handlers:
                handler = self.handlers[request_id]
                success = await asyncio.wait_for(handler.stop(), timeout=10.0)
                return success
        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping stream {request_id}")
        except Exception as e:
            logger.error(f"Error stopping stream {request_id}: {e}")
        return False


# Convenience functions for common use cases
async def create_simple_stream(
    request_id: str,
    subscribe_url: str,
    publish_url: str,
    frame_processor: Callable[[VideoFrame], VideoOutput],
    control_url: Optional[str] = None,
    events_url: Optional[str] = None,
    width: int = 512,
    height: int = 512
) -> Optional[StreamHandler]:
    """Create a simple stream handler with minimal configuration."""
    config = StreamConfig(
        subscribe_url=subscribe_url,
        publish_url=publish_url,
        control_url=control_url,
        events_url=events_url,
        width=width,
        height=height
    )
    
    handler = StreamHandler(
        request_id=request_id,
        config=config,
        frame_processor=frame_processor
    )
    
    success = await handler.start()
    if success:
        return handler
    else:
        return None


async def run_stream_until_complete(
    request_id: str,
    subscribe_url: str,
    publish_url: str,
    frame_processor: Callable[[VideoFrame], VideoOutput],
    control_url: Optional[str] = None,
    events_url: Optional[str] = None,
    width: int = 512,
    height: int = 512
) -> bool:
    """Run a stream until completion or error."""
    handler = await create_simple_stream(
        request_id=request_id,
        subscribe_url=subscribe_url,
        publish_url=publish_url,
        frame_processor=frame_processor,
        control_url=control_url,
        events_url=events_url,
        width=width,
        height=height
    )
    
    if not handler:
        return False
    
    try:
        # Wait for completion or error
        await handler.error_propagator.wait_for_shutdown()
        return True
    except Exception as e:
        logger.error(f"Stream {request_id} error: {e}")
        return False
    finally:
        await handler.stop() 