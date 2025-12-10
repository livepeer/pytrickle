"""
Stream state and error utilities for trickle streaming.

This module centralizes lifecycle management into a single enum-driven
state machine and exposes events for coordination. It replaces scattered
boolean flags to improve clarity and reduce redundant state.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class PipelineState(Enum):
    """Lifecycle states for a stream/pipeline.

    These states match the external API status responses exactly.
    Only one state should be active at a time.
    """

    LOADING = "LOADING"          # initial state, setting up resources, warming pipeline
    IDLE = "IDLE"               # pipeline ready but no active streams  
    OK = "OK"                   # pipeline ready with active streams
    ERROR = "ERROR"             # error state

class StreamState:
    """Unified state management for stream lifecycle using enums and events.

    This replaces multiple boolean flags with a single enum and a small
    set of events that consumers can wait on. Designed to be reusable by
    pytrickle and higher-level integrations like ComfyStream.
    """

    def __init__(self) -> None:
        self._state: PipelineState = PipelineState.LOADING

        # Coordination events
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Client tracking
        self.active_client: bool = False
        self.active_streams: int = 0
        self.startup_complete: bool = False

        # State transition mapping for maintainability
        self._state_transitions = {
            PipelineState.LOADING: self._set_loading,
            PipelineState.IDLE: self._set_idle,
            PipelineState.OK: self._set_ok,
            PipelineState.ERROR: self._set_error,
        }

    # State and derived flags
    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def running(self) -> bool:
        """Compatibility: True when the stream is in an active state."""
        return self.running_event.is_set() and not self.shutdown_event.is_set() and not self.error_event.is_set()

    @property
    def pipeline_ready(self) -> bool:
        """Compatibility: True when the pipeline is warmed/ready."""
        return self._state in (PipelineState.IDLE, PipelineState.OK)

    def _update_state_from_activity(self) -> None:
        """Update internal state based on current activity levels.
        
        Transitions between OK and IDLE states based on whether there are
        active streams or clients. Only transitions if not in ERROR state.
        """
        if not self.error_event.is_set():
            if self.active_streams > 0 or self.active_client:
                if self._state != PipelineState.OK:
                    self.set_state(PipelineState.OK)
            elif self.startup_complete and self.pipeline_ready:
                if self._state != PipelineState.IDLE:
                    self.set_state(PipelineState.IDLE)

    def set_active_client(self, active: bool):
        """Track whether there's an active streaming client."""
        self.active_client = active
        self._update_state_from_activity()

    def update_component_health(self, component_name: str, health_data: dict):
        """Update component health and log errors without persisting them.
        
        Trickle protocol errors (subscriber/publisher connection issues) are normal
        operational events and should not persist as system-level ERROR states.
        """
        if health_data.get("error"):
            # Log component errors for debugging but don't change system state
            logger.debug(f"Component {component_name} reported error (normal): {health_data.get('error')}")
            # Component errors are transient and don't affect overall pipeline health

    def get_state(self) -> dict:
        """Get a dict representation of the current state."""
        return {
            "status": self._get_current_status(),
            "pipeline_ready": self.pipeline_ready,
            "shutdown_initiated": self.shutdown_event.is_set(),
            "error": self.error_event.is_set(),
        }
    
    def _get_current_status(self) -> str:
        """Get current status based on state and activity."""
        # Determine status based on actual activity, but don't auto-transition internal state
        if self.error_event.is_set():
            return "ERROR"
        elif self.active_streams > 0 or self.active_client:
            return "OK"
        elif self.startup_complete:
            return "IDLE"
        else:
            return "LOADING"
    
    # Internal state transition methods
    def _set_loading(self) -> None:
        """Set LOADING state."""
        self._state = PipelineState.LOADING
        self.running_event.set()

    def _set_idle(self) -> None:
        """Set pipeline state to IDLE and signal waiting tasks."""
        self._state = PipelineState.IDLE
        self.running_event.set()

    def _set_ok(self) -> None:
        """Set pipeline state to OK (active streams)."""
        self._state = PipelineState.OK
        self.running_event.set()

    def _set_error(self) -> None:
        """Set ERROR state and trigger shutdown."""
        logger.error(f"State transition: {self._state.name} → ERROR")
        self._state = PipelineState.ERROR
        self.shutdown_event.set()
        self.error_event.set()

    def set_state(self, state: PipelineState) -> None:
        """Primary interface for setting pipeline state.
        
        Args:
            state: PipelineState enum value
        
        Example:
            set_state(PipelineState.IDLE)
        """
        transition_handler = self._state_transitions.get(state)
        if transition_handler:
            transition_handler()
        else:
            logger.warning(f"Unknown PipelineState enum: {state}")

    def set_startup_complete(self) -> None:
        """Mark startup as complete for health/status reporting."""
        self.startup_complete = True
        # When startup completes, transition to IDLE (ready state)
        if self._state == PipelineState.LOADING:
            self.set_state(PipelineState.IDLE)

    def update_active_streams(self, count: int) -> None:
        """Update number of active streams for health/status reporting."""
        self.active_streams = max(0, int(count))
        self._update_state_from_activity()

    def is_error(self) -> bool:
        return self.error_event.is_set()

    def set_error(self, message: str) -> None:
        """Enter ERROR phase with an error message."""
        import traceback
        logger.error(f"Stream state: ERROR - {message}")
        logger.error(f"Error occurred from state {self._state.name}, call stack:")
        logger.error(traceback.format_stack()[-3:-1])  # Show calling context
        self.set_state(PipelineState.ERROR)

    def clear_error(self) -> None:
        """Clear error state and return to appropriate state."""
        if self.is_error():
            # Clear the error events first
            self.error_event.clear()
            self.shutdown_event.clear()
            # Return to default LOADING state
            self.set_state(PipelineState.LOADING)
            logger.info("Stream state: ERROR cleared, returning to LOADING")

    def get_pipeline_state(self) -> dict:
        """Return health-like payload used by /health endpoint.
        
        Maps detailed pipeline state to simplified health status like ai-runner:
        - LOADING → "LOADING" (only during actual startup)
        - IDLE → "IDLE" (ready but no active streams)
        - OK → "OK" (ready with active streams)
        - ERROR → "ERROR" (error state)
        """
        # Determine status based on actual state and activity, not just startup_complete
        if self.error_event.is_set():
            status = "ERROR"
        elif self.active_streams > 0 or self.active_client:
            # If we have active streams/client, we're definitely OK regardless of startup_complete
            status = "OK"
        elif self.pipeline_ready and self.startup_complete:
            # Pipeline ready and startup complete but no streams = IDLE
            status = "IDLE"
        else:
            # Still loading/starting up
            status = "LOADING"
            
        return {
            "status": status,
            "pipeline_ready": self.pipeline_ready,
            "active_streams": self.active_streams,
            "startup_complete": self.startup_complete,
        }
