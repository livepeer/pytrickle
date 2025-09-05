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
        """Test that width and height are blocked in runtime updates."""
        # Width alone should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": 1920})
        
        # Height alone should be blocked  
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"height": 1080})
        
        # Both provided should also be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": 1920, "height": 1080})
    
    def test_width_height_type_conversion(self):
        """Test that width/height are blocked regardless of type in runtime updates."""
        # String values should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": "1920", "height": "1080"})
        
        # Float values should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": 1920.0, "height": 1080.0})
    
    def test_width_height_positive_validation(self):
        """Test that width and height are blocked in runtime updates regardless of validity."""
        # Zero values should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": 0, "height": 1080})
        
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": 1920, "height": 0})
        
        # Negative values should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": -1920, "height": 1080})
        
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": 1920, "height": -1080})
    
    def test_width_height_invalid_conversion(self):
        """Test that width/height are blocked in runtime updates regardless of validity."""
        # Non-numeric strings should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": "invalid", "height": "1080"})
        
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": "1920", "height": "invalid"})
        
        # None values should be blocked
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate({"width": None, "height": "1080"})
    
    def test_mixed_params_with_dimensions(self):
        """Test that width/height are blocked even when mixed with valid params."""
        params = {
            "intensity": 0.8,
            "quality": "high",
            "width": "1920",
            "height": "1080",
            "enabled": True
        }
        
        # Should be blocked due to width/height
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(params)
        
        # But params without dimensions should work
        valid_params = {
            "intensity": 0.8,
            "quality": "high",
            "enabled": True
        }
        request = StreamParamsUpdateRequest.model_validate(valid_params)
        assert request.model_dump()["intensity"] == 0.8
        assert request.model_dump()["quality"] == "high"
        assert request.model_dump()["enabled"] is True
    
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
    
    def test_width_height_rejected_in_updates(self):
        """Test that width and height cannot be updated during runtime."""
        # Test that width alone is rejected in runtime updates
        invalid_width = {"width": 1920}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(invalid_width)
        
        # Test that height alone is rejected in runtime updates
        invalid_height = {"height": 1080}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(invalid_height)
        
        # Test that both width and height are rejected in runtime updates
        invalid_dimensions = {"width": 1920, "height": 1080}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(invalid_dimensions)
        
        # Test that other parameters still work
        valid_params = {"intensity": 0.8, "effect": "enhanced"}
        request = StreamParamsUpdateRequest.model_validate(valid_params)
        assert request.model_dump()["intensity"] == 0.8
        
        # Test mix of valid and invalid parameters (width)
        mixed_params_width = {"intensity": 0.9, "width": 1280}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(mixed_params_width)
        
        # Test mix of valid and invalid parameters (height)
        mixed_params_height = {"intensity": 0.9, "height": 720}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(mixed_params_height)
        
        # Test width/height rejected with string values in updates
        string_width_update = {"width": "1920"}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(string_width_update)
        
        string_height_update = {"height": "1080"}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(string_height_update)
        
        # Test combination of blocked parameters (framerate + dimensions)
        all_blocked = {"max_framerate": 30, "width": 1920, "height": 1080}
        # Should catch max_framerate first (since it's checked first in the code)
        with pytest.raises(ValueError, match="max_framerate cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(all_blocked)
        
        # Test dimensions without framerate (to ensure dimensions check works independently)
        dimensions_only = {"width": 1920, "height": 1080, "intensity": 0.5}
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(dimensions_only)
    
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
    
    def test_width_height_validation_in_start_request(self):
        """Test that width/height validation still works properly in StreamStartRequest."""
        # Test that both width and height are required together
        with pytest.raises(ValidationError, match="Both 'width' and 'height' must be provided together"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={"width": 1920}  # Missing height
            )
        
        with pytest.raises(ValidationError, match="Both 'width' and 'height' must be provided together"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={"height": 1080}  # Missing width
            )
        
        # Test valid width/height conversion
        request = StreamStartRequest(
            subscribe_url="http://example.com/input",
            publish_url="http://example.com/output", 
            gateway_request_id="test123",
            params={"width": "1920", "height": "1080"}
        )
        assert request.params["width"] == 1920
        assert request.params["height"] == 1080
        assert isinstance(request.params["width"], int)
        assert isinstance(request.params["height"], int)
        
        # Test invalid width/height values
        with pytest.raises(ValidationError, match="Width and height must be valid integers"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={"width": 0, "height": 1080}
            )
        
        with pytest.raises(ValidationError, match="Width and height must be valid integers"):
            StreamStartRequest(
                subscribe_url="http://example.com/input",
                publish_url="http://example.com/output", 
                gateway_request_id="test123",
                params={"width": "invalid", "height": "1080"}
            )
        
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
        """Test that StreamStartRequest allows dimensions but StreamParamsUpdateRequest blocks them."""
        test_params = {
            "intensity": 0.8,
            "width": "1920",
            "height": "1080"
        }
        
        # Test StreamParamsUpdateRequest - should be blocked due to width/height
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(test_params)
        
        # Test StreamStartRequest - should work
        start_request = StreamStartRequest(
            subscribe_url="http://localhost:3389/sample",
            publish_url="http://localhost:3389/output",
            gateway_request_id="test_request_123",
            params=test_params
        )
        
        # StreamStartRequest should have converted values
        assert start_request.params["width"] == 1920
        assert start_request.params["height"] == 1080
        assert start_request.params["intensity"] == 0.8
    
    def test_validation_error_consistency(self):
        """Test that models handle invalid params appropriately."""
        invalid_params = {"width": "1920"}  # Missing height
        
        # StreamParamsUpdateRequest should block width/height entirely
        with pytest.raises(ValueError, match="width and height cannot be updated during runtime"):
            StreamParamsUpdateRequest.model_validate(invalid_params)
        
        # StreamStartRequest should validate dimensions properly and require both
        with pytest.raises(ValueError, match="Both 'width' and 'height' must be provided together"):
            StreamStartRequest(
                subscribe_url="http://localhost:3389/sample",
                publish_url="http://localhost:3389/output",
                gateway_request_id="test_request_123",
                params=invalid_params
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
