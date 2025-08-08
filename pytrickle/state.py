"""
Stream state and error utilities for trickle streaming.

This module centralizes lifecycle management into a single enum-driven
state machine and exposes events for coordination. It replaces scattered
boolean flags to improve clarity and reduce redundant state.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto


logger = logging.getLogger(__name__)


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

