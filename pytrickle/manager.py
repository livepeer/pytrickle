"""
Base stream manager for trickle streaming applications.

This module provides reusable stream management functionality that can be
extended for specific use cases while maintaining core streaming patterns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

from .state import StreamState

logger = logging.getLogger(__name__)

class StreamHandler:
    """Base stream handler with common functionality for all trickle streams."""
    
    def __init__(self, width: int = 512, height: int = 512, **kwargs):
        """Initialize stream handler with resolution tracking."""
        super().__init__(**kwargs)  # Support multiple inheritance
        self.width = width
        self.height = height
    
    @property
    def running(self) -> bool:
        """Whether the stream handler is currently running. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the 'running' property")
    
    async def start(self) -> bool:
        """Start the stream handler. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement start()")
    
    async def stop(self, *, called_by_manager: bool = False) -> bool:
        """Stop the stream handler. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement stop()")
    
    def update_resolution(self, width: int, height: int) -> bool:
        """
        Update the resolution and return True if it changed.
        
        Args:
            width: New width value
            height: New height value
            
        Returns:
            True if resolution changed, False otherwise
        """
        changed = (self.width != width) or (self.height != height)
        
        if changed:
            old_width, old_height = self.width, self.height
            self.width = width
            self.height = height
            logger.info(f"Resolution updated from {old_width}x{old_height} to {width}x{height}")
        
        return changed

class BaseStreamManager(ABC):
    """Base stream manager that provides core stream management functionality."""
    
    def __init__(self, stream_state: Optional[StreamState] = None):
        self.handlers: Dict[str, StreamHandler] = {}
        self.lock = asyncio.Lock()
        self.stream_state = stream_state
    
    @abstractmethod
    async def create_stream_handler(self, request_id: str, **kwargs) -> Optional[StreamHandler]:
        """Create a stream handler for the given request. Must be implemented by subclasses."""
        pass
    
    async def create_stream(self, request_id: str, **kwargs) -> bool:
        """Create and start a new stream."""
        async with self.lock:
            if request_id in self.handlers:
                logger.warning(f"Stream {request_id} already exists")
                return False
            
            try:
                # Clear error state if this is the first stream after error
                if self.stream_state and self.stream_state.is_error():
                    if len(self.handlers) == 0:
                        self.stream_state.clear_error()
                
                handler = await self.create_stream_handler(request_id, **kwargs)
                if not handler:
                    logger.error(f"Failed to create stream handler for {request_id}")
                    return False
                
                success = await handler.start()
                if success:
                    self.handlers[request_id] = handler
                    self._update_stream_state()
                    logger.info(f"Stream {request_id} started successfully")
                    return True
                else:
                    logger.error(f"Failed to start stream handler for {request_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                if self.stream_state:
                    self.stream_state.set_error(f"Error creating stream: {str(e)}")
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        """Stop and remove a stream."""
        async with self.lock:
            if request_id not in self.handlers:
                logger.warning(f"Stream {request_id} not found")
                return False
            
            handler = self.handlers[request_id]
            try:
                success = await handler.stop(called_by_manager=True)
                del self.handlers[request_id]
                self._update_stream_state()
                logger.info(f"Stream {request_id} stopped, success: {success}")
                return success
            except Exception as e:
                logger.error(f"Error stopping stream {request_id}: {e}")
                # Remove from handlers even if stop failed
                del self.handlers[request_id]
                self._update_stream_state()
                return False
    
    def _update_stream_state(self):
        """Update the stream state with current stream count."""
        if self.stream_state:
            stream_count = len(self.handlers)
            self.stream_state.update_active_streams(stream_count)
            if stream_count == 0 and self.stream_state.is_error():
                self.stream_state.clear_error()
    
    def build_stream_status(self, request_id: str, handler: StreamHandler) -> Dict[str, Any]:
        """Compose a status dictionary for a given handler.

        Subclasses may override this to add more fields without altering
        locking semantics. This method must be non-blocking.
        """
        return {
            'request_id': request_id,
            'running': handler.running,
            'width': handler.width,
            'height': handler.height,
        }

    async def get_stream_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stream."""
        # Snapshot handler reference under lock, build status outside lock
        async with self.lock:
            handler = self.handlers.get(request_id)
        if not handler:
            return None
        try:
            return self.build_stream_status(request_id, handler)
        except Exception:
            # If handler is in an inconsistent state concurrently, surface as missing
            return None
    
    async def list_streams(self) -> Dict[str, Dict[str, Any]]:
        """List all active streams."""
        # Snapshot mapping under lock, build statuses outside to avoid blocking
        async with self.lock:
            items = list(self.handlers.items())
        result: Dict[str, Dict[str, Any]] = {}
        for request_id, handler in items:
            try:
                status = self.build_stream_status(request_id, handler)
                if status:
                    result[request_id] = status
            except Exception:
                # Skip handlers that may be concurrently stopping
                continue
        return result
    
    async def cleanup_all(self):
        """Stop and cleanup all streams."""
        async with self.lock:
            if not self.handlers:
                return
            
            cleanup_tasks = []
            for request_id in list(self.handlers.keys()):
                cleanup_tasks.append(self._stop_stream_with_timeout(request_id))
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("Global cleanup timeout reached, forcing cleanup")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            
            self.handlers.clear()
            self._update_stream_state()
    
    async def _stop_stream_with_timeout(self, request_id: str) -> bool:
        """Stop a stream with timeout protection."""
        try:
            if request_id in self.handlers:
                handler = self.handlers[request_id]
                return await asyncio.wait_for(handler.stop(called_by_manager=True), timeout=8.0)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Failed to stop stream {request_id} gracefully: {e}")
        return False

class TrickleStreamManager(BaseStreamManager):
    """Basic trickle stream manager implementation."""
    
    def __init__(self, app_context: Optional[Dict] = None, stream_state: Optional[StreamState] = None):
        # Extract stream state from app_context if provided
        if app_context and not stream_state:
            stream_state = app_context.get('stream_state')
        
        super().__init__(stream_state)
        self.app_context = app_context or {}
        self.stream_handler_factory: Optional[Callable] = None
    
    def set_stream_handler_factory(self, factory: Callable):
        """Set the factory function for creating stream handlers."""
        self.stream_handler_factory = factory
    
    async def create_stream_handler(self, request_id: str, **kwargs) -> Optional[StreamHandler]:
        """Create a stream handler using the configured factory."""
        if not self.stream_handler_factory:
            logger.error("No stream handler factory configured")
            return None
        
        try:
            return await self.stream_handler_factory(request_id, **kwargs, app_context=self.app_context)
        except Exception as e:
            logger.error(f"Error in stream handler factory: {e}")
            return None