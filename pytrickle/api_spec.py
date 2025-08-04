"""
Core Pydantic models for Trickle streaming API parameters.

This module defines reusable BaseModel classes for validating and serializing
API request parameters used across trickle streaming applications.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

class StreamStartRequest(BaseModel):
    """Base request model for starting a trickle stream."""
    subscribe_url: str = Field(..., description="URL for subscribing to input video stream")
    publish_url: str = Field(..., description="URL for publishing output video stream")
    control_url: Optional[str] = Field(default=None, description="URL for control channel communication")
    events_url: Optional[str] = Field(default=None, description="URL for events channel communication")
    data_url: Optional[str] = Field(default=None, description="URL for publishing text/data output via data channel")
    gateway_request_id: str = Field(..., description="Unique identifier for the stream request")
    
    # Optional fields that may be present in the request
    manifest_id: Optional[str] = Field(default=None, description="Manifest identifier")
    model_id: Optional[str] = Field(default=None, description="Model identifier")
    stream_id: Optional[str] = Field(default=None, description="Stream identifier")

class StreamParamsUpdateRequest(BaseModel):
    """Base request model for updating stream parameters."""
    width: Optional[int] = Field(default=None, description="Width of the generated video")
    height: Optional[int] = Field(default=None, description="Height of the generated video")

class StreamResponse(BaseModel):
    """Standard response model for stream operations."""
    status: str = Field(..., description="Operation status (success/error)")
    message: str = Field(..., description="Human-readable message")
    request_id: Optional[str] = Field(default=None, description="Stream request ID")
    config: Optional[dict] = Field(default=None, description="Stream configuration details")

class StreamStatusResponse(BaseModel):
    """Response model for stream status queries."""
    processing_active: bool = Field(..., description="Whether stream processing is active")
    stream_count: int = Field(..., description="Number of active streams")
    message: Optional[str] = Field(default=None, description="Status message")
    current_stream: Optional[dict] = Field(default=None, description="Current stream details")
    all_streams: Optional[dict] = Field(default=None, description="All active streams")

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoints."""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    stream_manager_ready: Optional[bool] = Field(default=None, description="Whether stream manager is ready")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")

class ServiceInfoResponse(BaseModel):
    """Response model for service information endpoints."""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    description: str = Field(..., description="Service description")
    capabilities: list = Field(..., description="List of service capabilities")
    endpoints: dict = Field(..., description="Available API endpoints")

# Health status model for general use
class HealthStatus(BaseModel):
    """Model for health status information."""
    status: str = Field(..., description="Health status (LOADING, IDLE, OK, ERROR)")
    error_message: Optional[str] = Field(default=None, description="Error message if status is ERROR")
    pipeline_ready: bool = Field(default=False, description="Whether processing pipeline is ready")
    active_streams: int = Field(default=0, description="Number of active streams")
    startup_complete: bool = Field(default=False, description="Whether startup is complete")
    additional_info: Optional[Dict[str, Any]] = Field(default=None, description="Additional status information")