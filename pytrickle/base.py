"""
Base classes for trickle protocol components.

Provides common functionality for subscriber and publisher components
to reduce code duplication and improve maintainability.
"""

import asyncio
import logging
from typing import Optional, Callable

from . import ErrorCallback

logger = logging.getLogger(__name__)

class TrickleComponent:
    """Base class for trickle protocol components with common error handling."""
    
    def __init__(self, error_callback: Optional[ErrorCallback] = None):
        self.error_event = asyncio.Event()  # Use Event instead of boolean
        self.shutdown_event = asyncio.Event()  # Event to signal shutdown
        self.error_callback = error_callback

    def _should_stop(self) -> bool:
        """Check if the component should stop due to error or shutdown signal."""
        return self.error_event.is_set() or self.shutdown_event.is_set()

    async def _notify_error(self, error_type: str, exception: Optional[Exception] = None):
        """Notify parent component of critical errors."""
        self.error_event.set()  # Set error event
        if self.error_callback:
            try:
                if asyncio.iscoroutinefunction(self.error_callback):
                    await self.error_callback(error_type, exception)
                else:
                    self.error_callback(error_type, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def shutdown(self):
        """Signal shutdown to stop background tasks."""
        self.shutdown_event.set() 