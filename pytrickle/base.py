"""
Base classes for trickle protocol components.

Provides common functionality for subscriber and publisher components
with integrated state tracking and health monitoring.
"""

import asyncio
import aiohttp
import logging
import time
from enum import Enum
from typing import Optional, List, Callable, Coroutine, Any

logger = logging.getLogger(__name__)

# Type alias for error callback functions (async only)
ErrorCallback = Callable[[str, Optional[Exception]], Coroutine[Any, Any, None]]

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
        
        # Background task tracking
        self._background_tasks: List[asyncio.Task] = []

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
                await self.error_callback(error_type, exception)
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

    def _track_background_task(self, task: asyncio.Task):
        """Track a background task with unified cleanup handling."""
        self._background_tasks.append(task)
        
        def cleanup_task(completed_task):
            try:
                if not completed_task.cancelled():
                    # Retrieve any exception to prevent "never retrieved" error
                    completed_task.result()
            except (asyncio.CancelledError, Exception):
                # Task was cancelled or had an exception - both are handled appropriately
                pass
            finally:
                try:
                    if completed_task in self._background_tasks:
                        self._background_tasks.remove(completed_task)
                except ValueError:
                    # Task already removed, that's fine
                    pass
        
        if task.done():
            cleanup_task(task)
        else:
            task.add_done_callback(cleanup_task)

    async def shutdown(self):
        """Signal shutdown to stop background tasks."""
        self._update_state(ComponentState.SHUTDOWN)
        self.shutdown_event.set()

def setup_asyncio_exception_handler():
    """Setup global asyncio exception handler to suppress expected aiohttp errors during shutdown."""
    
    def handle_exception(loop, context):
        """Handle uncaught asyncio task exceptions."""
        exception = context.get('exception')
        
        if isinstance(exception, aiohttp.ClientConnectionResetError):
            logger.debug(f"Suppressed expected aiohttp connection reset during shutdown: {exception}")
            return
        
        loop.default_exception_handler(context)
    
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(handle_exception)
        logger.debug("Global asyncio exception handler installed")
    except RuntimeError:
        logger.debug("No running asyncio loop, exception handler will be set later")