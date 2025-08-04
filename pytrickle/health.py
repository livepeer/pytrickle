"""
Health management for trickle streaming applications.

This module provides a reusable health state manager that can track the health
of streaming applications including pipeline status, active streams, and errors.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class StreamHealthManager:
    """Manages the health state of a streaming application."""
    
    def __init__(self, service_name: str = "trickle-stream-service"):
        self.service_name = service_name
        self.state = "LOADING"  # Initial state during startup
        self.error_message = None
        self.is_pipeline_warming = False
        self.pipeline_ready = False
        self.active_streams = 0
        self.startup_complete = False
        self.additional_info = {}
        
    def set_loading(self, reason: Optional[str] = None):
        """Set state to LOADING (pipeline warming up)."""
        self.state = "LOADING"
        self.error_message = None
        logger.info(f"Health state: LOADING{f' - {reason}' if reason else ''}")
        
    def set_idle(self):
        """Set state to IDLE (no active streams)."""
        if self.state == "ERROR":
            return  # Don't change from ERROR state unless explicitly cleared
        self.state = "IDLE"
        self.error_message = None
        logger.debug("Health state: IDLE")
        
    def set_ok(self):
        """Set state to OK (streams active)."""
        if self.state == "ERROR":
            return  # Don't change from ERROR state unless explicitly cleared
        self.state = "OK"
        self.error_message = None
        logger.debug("Health state: OK")
        
    def set_error(self, message: str):
        """Set state to ERROR with error message."""
        self.state = "ERROR"
        self.error_message = message
        logger.error(f"Health state: ERROR - {message}")
        
    def clear_error(self):
        """Clear error state and recalculate appropriate state."""
        self.error_message = None
        # Reset state from ERROR before recalculating - this allows _update_state to work properly
        if self.state == "ERROR":
            self.state = "LOADING"  # Temporary state before recalculation
        self._update_state()
        
    def set_pipeline_warming(self, warming: bool):
        """Set pipeline warming state."""
        self.is_pipeline_warming = warming
        if warming:
            self.set_loading("Pipeline warming")
        else:
            self._update_state()
    
    def set_pipeline_ready(self, ready: bool):
        """Set pipeline ready state."""
        self.pipeline_ready = ready
        if ready and not self.is_pipeline_warming:
            self._update_state()
            
    def set_startup_complete(self):
        """Mark startup as complete."""
        self.startup_complete = True
        self._update_state()
        
    def update_active_streams(self, count: int):
        """Update count of active streams."""
        self.active_streams = count
        self._update_state()
        
    def set_additional_info(self, key: str, value: Any):
        """Set additional health information."""
        self.additional_info[key] = value
        
    def get_additional_info(self, key: str) -> Any:
        """Get additional health information."""
        return self.additional_info.get(key)
        
    def _update_state(self):
        """Internal method to update state based on current conditions."""
        if self.state == "ERROR":
            return  # Don't change from ERROR state
            
        # Check if we should be in LOADING state
        if not self.startup_complete or self.is_pipeline_warming:
            self.set_loading("Startup in progress" if not self.startup_complete else "Pipeline warming")
            return
            
        # Check if we have active streams
        if self.active_streams > 0:
            self.set_ok()
        else:
            self.set_idle()
            
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "status": self.state,
            "error_message": self.error_message,
            "pipeline_ready": self.pipeline_ready,
            "active_streams": self.active_streams,
            "startup_complete": self.startup_complete,
            "additional_info": self.additional_info
        }

# ComfyStreamHealthManager has been moved to comfystream.server.health
# This keeps pytrickle focused on reusable components