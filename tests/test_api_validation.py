"""
Tests for Pydantic validation logic in pytrickle.api module.

Focuses on testing the params validation, dimension conversion, and max_framerate
validation behavior in StreamStartRequest and StreamParamsUpdateRequest models.
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
    
    def test_max_framerate_rejected_in_updates(self):
        """Test that max_framerate cannot be updated during runtime."""
        # Test that max_framerate is rejected in runtime updates
        invalid_params = {"max_framerate": 60}
        with pytest.raises(ValueError, match="max_framerate cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(invalid_params)
        
        # Test that other parameters still work
        valid_params = {"intensity": 0.8, "effect": "enhanced"}
        request = StreamParamsUpdateRequest.model_validate(valid_params)
        assert request.model_dump()["intensity"] == 0.8
        
        # Test mix of valid and invalid parameters
        mixed_params = {"intensity": 0.9, "max_framerate": 45}
        with pytest.raises(ValueError, match="max_framerate cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(mixed_params)
        
        # Test max_framerate rejected with string value in updates
        string_update = {"max_framerate": "30"}
        with pytest.raises(ValueError, match="max_framerate cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(string_update)
    
    def test_framerate_conversion_method(self):
        """Test the _convert_framerate method directly."""
        # Test valid conversion
        params = {"max_framerate": "30", "other_param": "value"}
        converted = StreamParamsUpdateRequest._convert_framerate(params)
        assert converted["max_framerate"] == 30
        assert isinstance(converted["max_framerate"], int)
        assert converted["other_param"] == "value"
        
        # Test no framerate parameter
        params_no_fr = {"other_param": "value"}
        converted_no_fr = StreamParamsUpdateRequest._convert_framerate(params_no_fr)
        assert "max_framerate" not in converted_no_fr
        assert converted_no_fr["other_param"] == "value"
        
        # Test invalid framerate
        params_invalid = {"max_framerate": "invalid"}
        with pytest.raises(ValueError, match="max_framerate must be a valid integer"):
            StreamParamsUpdateRequest._convert_framerate(params_invalid)
            
        # Test negative framerate
        params_negative = {"max_framerate": -10}
        with pytest.raises(ValueError, match="max_framerate must be a positive integer"):
            StreamParamsUpdateRequest._convert_framerate(params_negative)
            
        # Test framerate exceeding maximum (60 FPS)
        params_too_high = {"max_framerate": 120}
        with pytest.raises(ValueError, match="max_framerate cannot exceed 60 FPS"):
            StreamParamsUpdateRequest._convert_framerate(params_too_high)


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
    
    def test_max_framerate_validation(self):
        """Test that max_framerate is correctly validated in stream start requests."""
        # Test valid max_framerate
        request = StreamStartRequest(
            subscribe_url="http://example.com/input",
            publish_url="http://example.com/output", 
            gateway_request_id="test123",
            params={
                "width": 512,
                "height": 512,
                "max_framerate": 30
            }
        )
        
        assert request.params["max_framerate"] == 30
        assert isinstance(request.params["max_framerate"], int)
        
        # Test string max_framerate gets converted to int
        request_str = StreamStartRequest(
            subscribe_url="http://example.com/input",
            publish_url="http://example.com/output", 
            gateway_request_id="test123",
            params={
                "width": 512,
                "height": 512,
                "max_framerate": "25"
            }
        )
        
        assert request_str.params["max_framerate"] == 25
        assert isinstance(request_str.params["max_framerate"], int)
        
        # Test invalid max_framerate (negative)
        with pytest.raises(ValidationError, match="max_framerate must be a positive integer"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={
                    "width": 512,
                    "height": 512,
                    "max_framerate": -5
                }
            )
            
        # Test non-numeric max_framerate
        with pytest.raises(ValidationError, match="max_framerate must be a valid integer"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={
                    "width": 512,
                    "height": 512,
                    "max_framerate": "invalid"
                }
            )
            
        # Test max_framerate exceeding 60 FPS limit
        with pytest.raises(ValidationError, match="max_framerate cannot exceed 60 FPS"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={
                    "width": 512,
                    "height": 512,
                    "max_framerate": 120
                }
            )
            
        # Test max_framerate at the limit (60) should work
        request_at_limit = StreamStartRequest(
            subscribe_url="http://example.com/input",
            publish_url="http://example.com/output", 
            gateway_request_id="test123",
            params={
                "width": 512,
                "height": 512,
                "max_framerate": 60
            }
        )
        assert request_at_limit.params["max_framerate"] == 60
    
    def test_max_framerate_limits(self):
        """Test specific max_framerate limit validation."""
        # Test common valid values
        valid_framerates = [1, 15, 24, 30, 45, 60]
        for fps in valid_framerates:
            request = StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output",
                gateway_request_id="test123",
                params={
                    "width": 512,
                    "height": 512,
                    "max_framerate": fps
                }
            )
            assert request.params["max_framerate"] == fps
        
        # Test invalid values (above 60)
        invalid_framerates = [61, 75, 100, 120, 240]
        for fps in invalid_framerates:
            with pytest.raises(ValidationError, match="max_framerate cannot exceed 60 FPS"):
                StreamStartRequest(
                    subscribe_url="http://example.com/input",
                    publish_url="http://example.com/output",
                    gateway_request_id="test123",
                    params={
                        "width": 512,
                        "height": 512,
                        "max_framerate": fps
                    }
                )
    
    def test_readme_example_parsing(self):
        """Test that the README example request format is correctly parsed."""
        # This is the exact format from the README
        request = StreamStartRequest(
            subscribe_url="http://127.0.0.1:3389/",
            publish_url="http://127.0.0.1:3389/",
            gateway_request_id="test",
            params={
                "width": 512,
                "height": 512,
                "max_framerate": 30
            }
        )
        
        # This should parse correctly
        assert request.params["max_framerate"] == 30
        assert request.params["width"] == 512
        assert request.params["height"] == 512
    
    def test_complete_max_framerate_flow(self):
        """Test the complete flow from HTTP request format to validation."""
        # Test the exact curl format from README
        request = StreamStartRequest(
            subscribe_url="http://127.0.0.1:3389/",
            publish_url="http://127.0.0.1:3389/",
            gateway_request_id="test",
            params={
                "width": 512,
                "height": 512,
                "max_framerate": 30
            }
        )
        
        # Validate the request
        assert request.params["max_framerate"] == 30
        
        # Simulate server parameter extraction
        params_dict = request.params or {}
        width = params_dict.get("width", 512)
        height = params_dict.get("height", 512)
        max_framerate = params_dict.get("max_framerate", None)
        
        assert width == 512
        assert height == 512
        assert max_framerate == 30
        
        # Test default case (no max_framerate)
        request_no_framerate = StreamStartRequest(
            subscribe_url="http://127.0.0.1:3389/",
            publish_url="http://127.0.0.1:3389/",
            gateway_request_id="test",
            params={
                "width": 512,
                "height": 512
                # No max_framerate specified
            }
        )
        
        params_dict_no_fr = request_no_framerate.params or {}
        max_framerate_default = params_dict_no_fr.get("max_framerate", None)
        
        assert max_framerate_default is None  # Should be None when not provided


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
