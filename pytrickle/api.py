"""
Core Pydantic models for Trickle streaming API parameters.

This module defines reusable BaseModel classes for validating and serializing
API request parameters used across trickle streaming applications.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from .utils.hardware import GPUComputeInfo, GPUUtilizationInfo

class StreamStartRequest(BaseModel):
    """Base request model for starting a trickle stream.
    
    This model is used for both /api/stream/start and /live-video-to-video endpoints.
    """
    subscribe_url: Optional[str] = Field(default=None, description="URL for subscribing to input video stream")
    publish_url: Optional[str] = Field(default=None, description="URL for publishing output video stream")
    control_url: Optional[str] = Field(default=None, description="URL for control channel communication")
    events_url: Optional[str] = Field(default=None, description="URL for events channel communication")
    data_url: Optional[str] = Field(default=None, description="URL for publishing text/data output via data channel")
    gateway_request_id: str = Field(..., description="Unique identifier for the stream request")
    # Keep params identical in shape to update request, but optional for start
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dynamic parameters object with string field names and any values"
    )

    @field_validator('params')
    @classmethod
    def validate_optional_params_dict(cls, value):
        """Validate and process optional params, preserving any modifications from validation."""
        if value is not None:
            # Call validate_params and return the processed dictionary
            return StreamParamsUpdateRequest.validate_params(value)
        return value

class StreamParamsUpdateRequest(BaseModel):
    """Base request model for updating stream parameters.
    
    This model accepts arbitrary string field names with any value types,
    allowing flexible parameter updates without nested structure.
    Width and height values are automatically converted to integers if provided.
    
    Note: max_framerate cannot be updated during runtime and must be set when starting the stream.
    """
    
    model_config = {"extra": "allow"}  # Allow arbitrary fields
    
    @classmethod
    def validate_params(cls, v):
        """Validation method with automatic type conversion for width/height."""
        if v is None:
            return v
        
        if not isinstance(v, dict):
            raise ValueError("Params must be a dictionary")
        
        # Ensure all keys are strings (values can be any type now)
        for key in v.keys():
            if not isinstance(key, str):
                raise ValueError(f"All field names must be strings, got {type(key)} for key: {key}")
        
        # Convert and validate dimensions if present
        v = cls._convert_dimensions(v)
        # Convert and validate framerate if present  
        v = cls._convert_framerate(v)
        return v
    
    @classmethod
    def _convert_dimensions(cls, params_dict: dict) -> dict:
        """Convert and validate width/height parameters."""
        result = params_dict.copy()
        dimensions = {"width", "height"}
        provided_dims = dimensions.intersection(result.keys())
        
        if provided_dims:
            if provided_dims != dimensions:
                raise ValueError("Both 'width' and 'height' must be provided together")
            
            try:
                for dim in dimensions:
                    value = int(result[dim])
                    if value <= 0:
                        raise ValueError("Width and height must be positive integers")
                    result[dim] = value
            except (ValueError, TypeError):
                raise ValueError("Width and height must be valid integers or integer strings")
        
        return result
    
    @classmethod
    def _convert_framerate(cls, params_dict: dict) -> dict:
        """Convert and validate max_framerate parameter."""
        result = params_dict.copy()
        if "max_framerate" in result:
            try:
                value = int(result["max_framerate"])
            except (ValueError, TypeError):
                raise ValueError("max_framerate must be a valid integer")
            
            if value <= 0:
                raise ValueError("max_framerate must be a positive integer")
            if value > 60:
                raise ValueError("max_framerate cannot exceed 60 FPS")
            result["max_framerate"] = value
        
        return result
    
    @classmethod
    def model_validate(cls, obj):
        """Custom validation to ensure all fields are string key-value pairs."""
        if isinstance(obj, dict):
            # Check for unsupported runtime parameters
            if "max_framerate" in obj:
                raise ValueError("max_framerate cannot be updated during runtime. Set it when starting the stream.")
            
            # Validate and get the processed dictionary with dimension conversions
            obj = cls.validate_params(obj)
        return super().model_validate(obj)
    
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

class Version(BaseModel):
    """Model for version information."""
    pipeline: str = "byoc"
    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Version string")


class HardwareInformation(BaseModel):
    """Response model for GPU information."""

    pipeline: str
    model_id: str
    gpu_info: Dict[int, GPUComputeInfo]


class HardwareStats(BaseModel):
    """Response model for real-time GPU statistics."""

    pipeline: str
    model_id: str
    gpu_stats: Dict[int, GPUUtilizationInfo]
