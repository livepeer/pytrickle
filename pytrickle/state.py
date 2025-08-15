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

class PipelineVersion:
    """Version information for the pipeline."""
    
    def __init__(self, version: str):
        self.version = version
    
    @property
    def state(self) -> str:
        return self.version

class PipelineState(Enum):
    """Lifecycle states for a stream/pipeline.

    Using a single enum avoids overlapping booleans. Only one state
    should be active at a time. Values map directly to external state strings.
    """

    INIT = "LOADING"              # initial state maps to loading externally
    LOADING = "LOADING"           # setting up resources
    WARMING_PIPELINE = "LOADING"  # warming the inference pipeline
    READY = "IDLE"               # pipeline warmed and ready (idle until streams)
    SHUTTING_DOWN = "IDLE"       # coordinated shutdown
    ERROR = "ERROR"              # error state
    STOPPED = "IDLE"             # stopped state

class StreamState:
    """Unified state management for stream lifecycle using enums and events.

    This replaces multiple boolean flags with a single enum and a small
    set of events that consumers can wait on. Designed to be reusable by
    pytrickle and higher-level integrations like ComfyStream.
    """

    def __init__(self) -> None:
        self._state: PipelineState = PipelineState.INIT

        # Coordination events
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.pipeline_ready_event = asyncio.Event()
        
        # Client tracking
        self.active_client: bool = False
        self.active_streams: int = 0
        self.startup_complete: bool = False

    # State and derived flags
    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether processing loops should keep running.

        Active in LOADING, WARMING_PIPELINE, READY. Inactive otherwise.
        """
        return self._state in {
            PipelineState.LOADING,
            PipelineState.WARMING_PIPELINE,
            PipelineState.READY,
        } and not self.shutdown_event.is_set() and not self.error_event.is_set()

    # Backwards-compat shorthand properties (read-only)
    @property
    def running(self) -> bool:
        """Compatibility: True when the stream is in an active state."""
        return self.is_active

    @property
    def pipeline_ready(self) -> bool:
        """Compatibility: True when the pipeline is warmed/ready."""
        return self._state is PipelineState.READY

    def set_active_client(self, active: bool):
        """Track whether there's an active streaming client."""
        self.active_client = active

    def update_component_health(self, component_name: str, health_data: dict):
        """Update component health and propagate errors to main state."""
        # If component has error, set main error event
        if health_data.get("error"):
            self.error_event.set()

    def get_state(self) -> dict:
        """Get a dict representation of the current state."""
        # Determine status: prioritize ERROR, then "OK" if actively streaming, otherwise pipeline state
        if self.error_event.is_set():
            status = "ERROR"
        elif self._state is PipelineState.READY and (self.active_client or self.active_streams > 0):
            status = "OK"
        else:
            status = self._state.value
        return {
            "status": status,
            "pipeline_ready": self.pipeline_ready,
            "shutdown_initiated": self.shutdown_event.is_set(),
            "error": self.error_event.is_set(),
        }

    def get_stream_state(self) -> dict:
        """Get current state (alias for get_state)."""
        return self.get_state()
    
    # State transitions
    def start(self) -> None:
        """Begin stream startup; transitions to LOADING."""
        if self._state is PipelineState.INIT:
            self._state = PipelineState.LOADING
            self.running_event.set()

    def set_loading(self) -> None:
        """Explicitly set LOADING state (alias for start)."""
        self.start()

    def set_pipeline_warming(self) -> None:
        """Transition to WARMING_PIPELINE."""
        if self._state in {PipelineState.INIT, PipelineState.LOADING}:
            self._state = PipelineState.WARMING_PIPELINE
            self.running_event.set()

    def set_pipeline_ready(self) -> None:
        """Set pipeline state to READY and signal waiting tasks."""
        self._state = PipelineState.READY
        self.running_event.set()
        self.pipeline_ready_event.set()

    def initiate_shutdown(self, *, due_to_error: bool = False) -> None:
        """Begin coordinated shutdown. Optionally mark as error."""
        self._state = PipelineState.ERROR if due_to_error else PipelineState.SHUTTING_DOWN
        self.shutdown_event.set()
        if due_to_error:
            self.error_event.set()

    def finalize(self) -> None:
        """Finalize and mark as STOPPED; clear running event."""
        self._state = PipelineState.STOPPED
        self.running_event.clear()

    def _reset_to_init(self) -> None:
        """Reset to initial state - clear events and set to INIT."""
        self._state = PipelineState.INIT
        self.running_event.clear()
        self.pipeline_ready_event.clear()
        self.shutdown_event.clear()
        self.error_event.clear()

    def set_state(self, state: PipelineState) -> None:
        """Primary interface for setting pipeline state.
        
        Args:
            state: PipelineState enum value
        
        Example:
            set_state(PipelineState.READY)
        """
        if state == PipelineState.WARMING_PIPELINE:
            self._set_pipeline_warming()
        elif state == PipelineState.READY:
            self._set_pipeline_ready()
        elif state == PipelineState.LOADING:
            self._set_loading()
        elif state == PipelineState.ERROR:
            self._initiate_shutdown(due_to_error=True)
        elif state == PipelineState.SHUTTING_DOWN:
            self._initiate_shutdown()
        elif state == PipelineState.STOPPED:
            self._finalize()
        elif state == PipelineState.INIT:
            self._reset_to_init()
        else:
            logger.warning(f"Unknown PipelineState enum: {state}")

    # ---- Unified health-style API (to replace StreamHealthManager)
    def set_startup_complete(self) -> None:
        """Mark startup as complete for health/status reporting."""
        self.startup_complete = True
        # If no explicit state chosen yet, move from INIT -> LOADING
        if self._state is PipelineState.INIT:
            self._state = PipelineState.LOADING
            self.running_event.set()

    def update_active_streams(self, count: int) -> None:
        """Update number of active streams for health/status reporting."""
        self.active_streams = max(0, int(count))

    def is_error(self) -> bool:
        return self.error_event.is_set()

    def set_error(self, message: str) -> None:
        """Enter ERROR phase with an error message."""
        self.set_state(PipelineState.ERROR)
        logger.error(f"Stream state: ERROR - {message}")

    def clear_error(self) -> None:
        """Clear error state and return to appropriate state."""
        if self.is_error():
            # Clear the error event first
            self.error_event.clear()
            # Return to a sensible default state
            self.set_state(PipelineState.LOADING)
            logger.info("Stream state: ERROR cleared, returning to LOADING")

    @property
    def pipeline_warming(self) -> bool:
        return self._state is PipelineState.WARMING_PIPELINE

    def get_pipeline_state(self) -> dict:
        """Return health-like payload used by /health endpoint."""
        # If startup not complete, always show LOADING
        if not self.startup_complete:
            outward = "LOADING"
        elif self._state is PipelineState.READY:
            outward = "OK" if self.active_streams > 0 else "IDLE"
        else:
            outward = self._state.value
        return {
            "status": outward,  # backward-compatible key
            "state": outward,
            "error_message": None,  # keep compatibility with previous health payload
            "pipeline_ready": self._state is PipelineState.READY,
            "active_streams": self.active_streams,
            "startup_complete": self.startup_complete,
            "pipeline_state": self._state.name,
            "additional_info": {},
        }
