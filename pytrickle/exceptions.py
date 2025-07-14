"""
Trickle streaming exceptions and error handling.

Defines custom exceptions for better error classification and handling
in trickle streaming applications.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TrickleErrorType(Enum):
    """Types of trickle streaming errors."""
    CONNECTION_ERROR = "connection_error"
    STREAM_CLOSED = "stream_closed"
    AUTHENTICATION_ERROR = "authentication_error"
    PROTOCOL_ERROR = "protocol_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"


class TrickleException(Exception):
    """Base exception for trickle streaming errors."""
    
    def __init__(self, message: str, error_type: TrickleErrorType, retryable: bool = True, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.metadata = metadata or {}
        
    def __str__(self):
        return f"{self.error_type.value}: {super().__str__()}"


class TrickleConnectionError(TrickleException):
    """Exception for connection-related errors."""
    
    def __init__(self, message: str, url: str, retryable: bool = True):
        super().__init__(message, TrickleErrorType.CONNECTION_ERROR, retryable, {"url": url})
        self.url = url


class TrickleStreamClosedError(TrickleException):
    """Exception for stream closure errors."""
    
    def __init__(self, message: str, url: str):
        super().__init__(message, TrickleErrorType.STREAM_CLOSED, False, {"url": url})
        self.url = url


class TrickleTimeoutError(TrickleException):
    """Exception for timeout errors."""
    
    def __init__(self, message: str, timeout: float, retryable: bool = True):
        super().__init__(message, TrickleErrorType.TIMEOUT_ERROR, retryable, {"timeout": timeout})
        self.timeout = timeout


class TrickleMaxRetriesError(TrickleException):
    """Exception when maximum retries are exceeded."""
    
    def __init__(self, message: str, max_retries: int, last_error: Optional[Exception] = None):
        super().__init__(message, TrickleErrorType.CONNECTION_ERROR, False, {"max_retries": max_retries})
        self.max_retries = max_retries
        self.last_error = last_error


class ErrorPropagator:
    """Utility class for error propagation and shutdown signaling."""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.error_callbacks = []
        self.last_error: Optional[Exception] = None
        
    def add_error_callback(self, callback):
        """Add a callback to be called when an error occurs."""
        self.error_callbacks.append(callback)
        
    async def propagate_error(self, error: Exception, source: str = "unknown"):
        """Propagate an error and trigger shutdown if necessary."""
        self.last_error = error
        logger.error(f"Error from {source}: {error}")
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, source)
                else:
                    callback(error, source)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # For non-retryable errors, signal shutdown
        if isinstance(error, TrickleException) and not error.retryable:
            logger.info(f"Non-retryable error from {source}, signaling shutdown")
            self.shutdown_event.set()
        elif isinstance(error, TrickleMaxRetriesError):
            logger.info(f"Max retries exceeded from {source}, signaling shutdown")
            self.shutdown_event.set()
            
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_event.is_set()
        
    async def wait_for_shutdown(self, timeout: Optional[float] = None):
        """Wait for shutdown signal."""
        if timeout:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout)
        else:
            await self.shutdown_event.wait() 