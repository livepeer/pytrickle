"""
Health management for trickle streaming applications.

StreamHealthManager uses `PipelineState` enums from `pytrickle.state`, 
and a single centralized state update function (i.e. "LOADING", "OK", "IDLE", "ERROR")
"""

import logging
from typing import Optional, Dict, Any

from .state import PipelineState

logger = logging.getLogger(__name__)


class StreamHealthManager:
    """Manages the health state of a streaming application using enums."""

    def __init__(self, service_name: str = "trickle-stream-service"):
        self.service_name = service_name
        # Internal source of truth for lifecycle (enum)
        self.pipeline_state: PipelineState = PipelineState.INIT
        # Derived, outward-facing state string
        self._state: str = PipelineState.LOADING.value
        self.error_message: Optional[str] = None
        self.active_streams: int = 0
        self.startup_complete: bool = False
        self.additional_info: Dict[str, Any] = {}
        # Initialize state from current fields
        self._update_state()

    # ------- Public operations (phase and health updates)
    def set_loading(self, reason: Optional[str] = None) -> None:
        """Enter LOADING phase."""
        if self.pipeline_state is not PipelineState.ERROR:
            self.pipeline_state = PipelineState.LOADING
            self.error_message = None
            logger.info(
                f"Health phase: LOADING{f' - {reason}' if reason else ''}"
            )
            self._update_state()

    def set_pipeline_warming(self, warming: Optional[bool] = None) -> None:
        """Enter or leave WARMING_PIPELINE phase.

        Back-compat: accepts an optional boolean. If True -> warming, if False ->
        leave warming (READY if startup complete else LOADING). If None -> enter
        warming.
        """
        if self.pipeline_state is PipelineState.ERROR:
            return
        if warming is False:
            if self.startup_complete:
                self.pipeline_state = PipelineState.READY
                logger.debug("Health state: READY (leaving warming)")
            else:
                self.pipeline_state = PipelineState.LOADING
                logger.debug("Health state: LOADING (leaving warming)")
            self.error_message = None
            self._update_state()
            return
        # warming is True or None -> set warming
        self.pipeline_state = PipelineState.WARMING_PIPELINE
        self.error_message = None
        logger.info("Health state: WARMING_PIPELINE")
        self._update_state()

    def set_pipeline_ready(self, ready: Optional[bool] = None) -> None:
        """Enter or leave READY phase.

        Back-compat: accepts an optional boolean. If True -> READY, if False ->
        LOADING (conservative default). If None -> READY.
        """
        if self.pipeline_state is PipelineState.ERROR:
            return
        if ready is False:
            self.pipeline_state = PipelineState.LOADING
            self.error_message = None
            logger.debug("Health state: LOADING (leaving ready)")
            self._update_state()
            return
        # ready is True or None -> set ready
        self.pipeline_state = PipelineState.READY
        self.error_message = None
        logger.debug("Health state: READY")
        self._update_state()

    def set_error(self, message: str) -> None:
        """Enter ERROR phase with an error message."""
        self.pipeline_state = PipelineState.ERROR
        self.error_message = message
        logger.error(f"Health state: ERROR - {message}")
        self._update_state()

    def clear_error(self) -> None:
        """Clear error state and recompute state from current conditions."""
        self.error_message = None
        if self.pipeline_state is PipelineState.ERROR:
            # Return to a sensible default and let state recompute
            self.pipeline_state = PipelineState.LOADING
        self._update_state()

    def set_startup_complete(self) -> None:
        """Mark startup as complete and recompute state."""
        self.startup_complete = True
        # If no explicit state chosen yet, move from INIT -> LOADING
        if self.pipeline_state is PipelineState.INIT:
            self.pipeline_state = PipelineState.LOADING
        self._update_state()

    def update_active_streams(self, count: int) -> None:
        """Update count of active streams and recompute state.

        Also resolves error state automatically when activity resumes.
        """
        self.active_streams = max(0, int(count))
        # If we were in ERROR but now have activity, clear the error
        if self.pipeline_state is PipelineState.ERROR and self.active_streams > 0:
            self.clear_error()
            return
        self._update_state()

    def set_additional_info(self, key: str, value: Any) -> None:
        self.additional_info[key] = value

    def get_additional_info(self, key: str) -> Any:
        return self.additional_info.get(key)

    # ------- Centralized state recomputation
    def _update_state(self) -> None:
        """Compute outward-facing state string from enum and fields.

        Rules:
        - Most phases map directly to their enum value
        - READY phase: "OK" if active_streams > 0, otherwise "IDLE"  
        - If startup not complete, force "LOADING" regardless of phase
        """
        # If startup not complete, always show LOADING
        if not self.startup_complete:
            self._state = "LOADING"
            return

        # Special case: READY depends on active streams
        if self.pipeline_state is PipelineState.READY:
            self._state = "OK" if self.active_streams > 0 else "IDLE"
            return

        # All other states use their enum value directly
        self._state = self.pipeline_state.value

    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current health state as a structured dict."""
        return {
            "state": self._state,
            "error_message": self.error_message,
            "pipeline_ready": self.pipeline_state is PipelineState.READY,
            "active_streams": self.active_streams,
            "startup_complete": self.startup_complete,
            "pipeline_state": self.pipeline_state.name,
            "additional_info": self.additional_info,
        }

    @property
    def state(self) -> str:
        """Outward-facing state string (e.g., "LOADING", "OK", "IDLE", "ERROR")."""
        return self._state

    def is_error(self) -> bool:
        """Whether the manager is currently in an error phase."""
        return self.pipeline_state is PipelineState.ERROR

    @property
    def pipeline_warming(self) -> bool:
        return self.pipeline_state is PipelineState.WARMING_PIPELINE

    @property
    def pipeline_ready(self) -> bool:
        return self.pipeline_state is PipelineState.READY

