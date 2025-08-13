"""
Tests for Pydantic validation logic in pytrickle.api module.

Focuses on testing the params validation and dimension conversion behavior
in StreamStartRequest and StreamParamsUpdateRequest models.
"""

import pytest
from pydantic import ValidationError
from pytrickle.api import StreamStartRequest, StreamParamsUpdateRequest

class TestStreamParamsUpdateRequest:
    """Test StreamParamsUpdateRequest validation and dimension conversion."""
    
    def test_basic_params_validation(self):
        """Test basic params validation with string keys."""
        # Valid params
        params = {"intensity": 0.8, "quality": "high", "enabled": True}
        request = StreamParamsUpdateRequest(**params)
        assert request.intensity == 0.8
        assert request.quality == "high"
        assert request.enabled is True
    
    def test_invalid_params_type(self):
        """Test that non-dict params raise validation error."""
        # Pydantic catches this before our custom validation
        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            StreamParamsUpdateRequest.model_validate("not_a_dict")
    
    def test_invalid_key_type(self):
        """Test that non-string keys raise validation error."""
        # Pydantic catches non-string keys before our custom validation
        invalid_params = {123: "value", "valid_key": "value"}
        with pytest.raises(ValueError, match="All field names must be strings"):
            StreamParamsUpdateRequest.model_validate(invalid_params)
    
    def test_width_height_both_required(self):
        """Test that both width and height must be provided together."""
        # Only width
        with pytest.raises(ValueError, match="Both 'width' and 'height' must be provided together"):
            StreamParamsUpdateRequest.model_validate({"width": 1920})
        
        # Only height
        with pytest.raises(ValueError, match="Both 'width' and 'height' must be provided together"):
            StreamParamsUpdateRequest.model_validate({"height": 1080})
        
        # Both provided - should work
        request = StreamParamsUpdateRequest.model_validate({"width": 1920, "height": 1080})
        assert request.width == 1920
        assert request.height == 1080
    
    def test_width_height_type_conversion(self):
        """Test that width/height are automatically converted to integers."""
        # String values
        request = StreamParamsUpdateRequest.model_validate({"width": "1920", "height": "1080"})
        assert request.width == 1920
        assert request.height == 1080
        assert isinstance(request.width, int)
        assert isinstance(request.height, int)
        
        # Float values
        request = StreamParamsUpdateRequest.model_validate({"width": 1920.0, "height": 1080.0})
        assert request.width == 1920
        assert request.height == 1080
        assert isinstance(request.width, int)
        assert isinstance(request.height, int)
    
    def test_width_height_positive_validation(self):
        """Test that width and height must be positive integers."""
        # Zero values
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": 0, "height": 1080})
        
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": 1920, "height": 0})
        
        # Negative values
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": -1920, "height": 1080})
        
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": 1920, "height": -1080})
    
    def test_width_height_invalid_conversion(self):
        """Test that invalid width/height values raise validation error."""
        # Non-numeric strings
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": "invalid", "height": "1080"})
        
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": "1920", "height": "invalid"})
        
        # None values
        with pytest.raises(ValueError, match="Width and height must be valid integers"):
            StreamParamsUpdateRequest.model_validate({"width": None, "height": "1080"})
    
    def test_mixed_params_with_dimensions(self):
        """Test mixing regular params with width/height dimensions."""
        params = {
            "intensity": 0.8,
            "quality": "high",
            "width": "1920",
            "height": "1080",
            "enabled": True
        }
        
        request = StreamParamsUpdateRequest.model_validate(params)
        
        # Check regular params
        assert request.intensity == 0.8
        assert request.quality == "high"
        assert request.enabled is True
        
        # Check converted dimensions
        assert request.width == 1920
        assert request.height == 1080
        assert isinstance(request.width, int)
        assert isinstance(request.height, int)
    
    def test_validate_params_method(self):
        """Test the validate_params class method directly."""
        # Test with dimensions
        params = {"width": "1920", "height": "1080"}
        validated = StreamParamsUpdateRequest.validate_params(params)
        assert validated["width"] == 1920
        assert validated["height"] == 1080
        assert isinstance(validated["width"], int)
        assert isinstance(validated["height"], int)
        
        # Test without dimensions
        params = {"intensity": 0.8, "quality": "high"}
        validated = StreamParamsUpdateRequest.validate_params(params)
        assert validated == params  # No changes
        
        # Test None
        assert StreamParamsUpdateRequest.validate_params(None) is None


class TestStreamStartRequest:
    """Test StreamStartRequest validation, particularly params handling."""
    
    def test_basic_validation(self):
        """Test basic StreamStartRequest validation."""
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123"
        )
        
        assert request.subscribe_url == "http://localhost:3389/sample"
        assert request.publish_url == "http://localhost:3389/output"
        assert request.gateway_request_id == "test_request_123"
        assert request.params is None
    
    def test_params_none(self):
        """Test that params can be None."""
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params=None
        )
        
        assert request.params is None
    
    def test_params_with_dimensions(self):
        """Test that params with dimensions are properly converted."""
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params={
                "intensity": 0.8,
                "width": "1920",
                "height": "1080"
            }
        )
        
        # Check that dimensions were converted to integers
        assert request.params["width"] == 1920
        assert request.params["height"] == 1080
        assert isinstance(request.params["width"], int)
        assert isinstance(request.params["height"], int)
        
        # Check that other params remain unchanged
        assert request.params["intensity"] == 0.8
    
    def test_params_without_dimensions(self):
        """Test that params without dimensions are passed through unchanged."""
        params = {
            "intensity": 0.8,
            "quality": "high",
            "enabled": True,
            "custom_param": "custom_value"
        }
        
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params=params
        )
        
        # Params should be unchanged
        assert request.params == params
    
    def test_params_validation_error_propagation(self):
        """Test that params validation errors are properly propagated."""
        # Invalid params (missing height)
        with pytest.raises(ValidationError, match="Both 'width' and 'height' must be provided together"):
            StreamStartRequest(
                subscribe_url="http://localhost:3389/sample",
                publish_url="http://localhost:3389/output",
                gateway_request_id="test_request_123",
                params={"width": "1920"}  # Missing height
            )
        
        # Invalid params (non-string keys) - Pydantic catches this before our validation
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            StreamStartRequest(
                subscribe_url="http://localhost:3389/sample",
                publish_url="http://localhost:3389/output",
                gateway_request_id="test_request_123",
                params={123: "invalid_key"}
            )
    
    def test_params_dimension_conversion_edge_cases(self):
        """Test edge cases in dimension conversion."""
        # String dimensions
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params={
                "width": "1920",
                "height": "1080"
            }
        )
        
        assert request.params["width"] == 1920
        assert request.params["height"] == 1080
        
        # Float dimensions
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params={
                "width": 1920.0,
                "height": 1080.0
            }
        )
        
        assert request.params["width"] == 1920
        assert request.params["height"] == 1080
    
    def test_params_complex_scenario(self):
        """Test a complex scenario with multiple param types and dimensions."""
        complex_params = {
            "intensity": 0.8,
            "quality": "high",
            "width": "1920",
            "height": "1080",
            "fps": 30,
            "codec": "h264",
            "enabled": True,
            "custom_list": [1, 2, 3],
            "custom_dict": {"nested": "value"}
        }
        
        request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params=complex_params
        )
        
        # Check dimension conversions
        assert request.params["width"] == 1920
        assert request.params["height"] == 1080
        assert isinstance(request.params["width"], int)
        assert isinstance(request.params["height"], int)
        
        # Check other params remain unchanged
        assert request.params["intensity"] == 0.8
        assert request.params["quality"] == "high"
        assert request.params["fps"] == 30
        assert request.params["codec"] == "h264"
        assert request.params["enabled"] is True
        assert request.params["custom_list"] == [1, 2, 3]
        assert request.params["custom_dict"] == {"nested": "value"}


class TestValidationIntegration:
    """Test integration between StreamStartRequest and StreamParamsUpdateRequest validation."""
    
    def test_validation_consistency(self):
        """Test that both models handle the same params consistently."""
        test_params = {
            "intensity": 0.8,
            "width": "1920",
            "height": "1080"
        }
        
        # Test StreamParamsUpdateRequest
        update_request = StreamParamsUpdateRequest.model_validate(test_params)
        assert update_request.width == 1920
        assert update_request.height == 1080
        
        # Test StreamStartRequest
        start_request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params=test_params
        )
        
        # Both should have the same converted values
        assert start_request.params["width"] == update_request.width
        assert start_request.params["height"] == update_request.height
    
    def test_validation_error_consistency(self):
        """Test that both models raise the same validation errors for invalid params."""
        invalid_params = {"width": "1920"}  # Missing height
        
        # Both should raise the same error
        with pytest.raises(ValueError, match="Both 'width' and 'height' must be provided together"):
            StreamParamsUpdateRequest.model_validate(invalid_params)
        
        with pytest.raises(ValueError, match="Both 'width' and 'height' must be provided together"):
            StreamStartRequest(
                subscribe_url="http://localhost:3389/sample",
                publish_url="http://localhost:3389/output",
                gateway_request_id="test_request_123",
                params=invalid_params
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
