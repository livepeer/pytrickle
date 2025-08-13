"""
Base classes for trickle protocol components.

Provides common functionality for subscriber and publisher components
with integrated state tracking and health monitoring.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Callable

from . import ErrorCallback

logger = logging.getLogger(__name__)

class ComponentState(Enum):
    """States for trickle protocol components.
    
    Defines the lifecycle states that components can be in during their operation.
    """
    INIT = "INIT"                # Initial state, not yet started
    STARTING = "STARTING"        # Component is starting up
    RUNNING = "RUNNING"          # Component is actively running
    STOPPING = "STOPPING"       # Component is shutting down
    STOPPED = "STOPPED"          # Component has stopped
    ERROR = "ERROR"              # Component encountered an error
    SHUTDOWN = "SHUTDOWN"        # Component received shutdown signal

class TrickleComponent:
    """Base class for trickle protocol components with state tracking."""
    
    def __init__(self, error_callback: Optional[ErrorCallback] = None, component_name: str = "unknown"):
        self.error_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_callback = error_callback
        self.component_name = component_name
        
        # Component state tracking
        self.component_state = ComponentState.INIT
        self.last_error: Optional[str] = None
        self.last_activity: Optional[float] = time.time()

    def _should_stop(self) -> bool:
        """Check if the component should stop due to error or shutdown signal."""
        return self.error_event.is_set() or self.shutdown_event.is_set()

    def _update_state(self, state: ComponentState):
        """Update component state and activity timestamp."""
        self.component_state = state
        self.last_activity = time.time()
        logger.debug(f"Component {self.component_name} state: {state.value}")

    async def _notify_error(self, error_type: str, exception: Optional[Exception] = None):
        """Enhanced error notification with state tracking."""
        self.error_event.set()
        self.component_state = ComponentState.ERROR
        self.last_error = f"{error_type}: {str(exception) if exception else 'Unknown error'}"
        self.last_activity = time.time()
        
        logger.error(f"Component {self.component_name} error: {self.last_error}")
        
        if self.error_callback:
            try:
                if asyncio.iscoroutinefunction(self.error_callback):
                    await self.error_callback(error_type, exception)
                else:
                    self.error_callback(error_type, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def get_component_health(self) -> dict:
        """Get component health status."""
        return {
            "component": self.component_name,
            "state": self.component_state.value,
            "error": self.error_event.is_set(),
            "last_error": self.last_error,
            "last_activity": self.last_activity,
            "should_stop": self._should_stop()
        }

    async def shutdown(self):
        """Signal shutdown to stop background tasks."""
        self._update_state(ComponentState.SHUTDOWN)
        self.shutdown_event.set()