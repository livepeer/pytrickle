"""
Tests for prompt parsing and validation functionality.

Tests the centralized prompt validation system in comfystream that handles
prompts from various sources. Tests the core validation logic and integration
with server endpoints, but not the pytrickle API models themselves.
"""

import json
import pytest
import logging
from pytrickle.api import (
    validate_prompts_for_api,
    parse_prompts_field,
    extract_prompts_from_params,
)

# Set up logging to show validation in action
logging.basicConfig(level=logging.INFO, format='🔍 %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test prompt data - simulates real ComfyUI workflows
SIMPLE_WORKFLOW = {
    "1": {
        "inputs": {
            "images": ["2", 0]
        },
        "class_type": "SaveTensor",
        "_meta": {
            "title": "SaveTensor"
        }
    },
    "2": {
        "inputs": {},
        "class_type": "LoadTensor",
        "_meta": {
            "title": "LoadTensor"
        }
    }
}

COMPLEX_WORKFLOW = {
    "1": {
        "inputs": {
            "text": "a beautiful landscape",
            "clip": ["2", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "2": {
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.ckpt"
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"}
    },
    "3": {
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["2", 0],
            "positive": ["1", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"}
    },
    "4": {
        "inputs": {
            "text": "",
            "clip": ["2", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative)"}
    },
    "5": {
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent Image"}
    }
}

AUDIO_WORKFLOW = {
    "1": {
        "inputs": {},
        "class_type": "LoadAudioTensor",
        "_meta": {"title": "LoadAudioTensor"}
    },
    "2": {
        "inputs": {
            "audio": ["1", 0],
            "gain": 1.0
        },
        "class_type": "AudioGain",
        "_meta": {"title": "Audio Gain"}
    },
    "3": {
        "inputs": {
            "audio": ["2", 0]
        },
        "class_type": "SaveAudioTensor",
        "_meta": {"title": "SaveAudioTensor"}
    }
}

class TestPromptValidationCore:
    """Test core prompt validation functionality."""
    
    def test_import_validation_functions(self):
        """Test that prompt validation functions can be imported."""
        assert callable(validate_prompts_for_api)
        assert callable(extract_prompts_from_params)
    
    def test_validate_single_prompt_dict(self):
        """Test validation of a single prompt dictionary."""
        logger.info("🎯 Testing single prompt dictionary validation...")
        
        logger.info("✅ Using pytrickle validation module")
        
        logger.info(f"📝 Input: Workflow with {len(SIMPLE_WORKFLOW)} nodes")
        result = validate_prompts_for_api(SIMPLE_WORKFLOW)
        logger.info(f"✅ Output: {len(result)} validated prompt(s)")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == SIMPLE_WORKFLOW
        logger.info("🎉 Single prompt dictionary validation PASSED!")
    
    def test_validate_prompt_list(self):
        """Test validation of a list of prompt dictionaries."""
        
        prompt_list = [SIMPLE_WORKFLOW, COMPLEX_WORKFLOW]
        result = validate_prompts_for_api(prompt_list)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == SIMPLE_WORKFLOW
        assert result[1] == COMPLEX_WORKFLOW
    
    def test_validate_json_string_single_prompt(self):
        """Test validation of JSON string containing single prompt."""
        logger.info("🎯 Testing JSON string prompt validation...")
        
        logger.info("✅ Using pytrickle validation module")
        
        json_string = json.dumps(SIMPLE_WORKFLOW)
        logger.info(f"📝 Input: JSON string ({len(json_string)} chars)")
        logger.info(f"🔤 JSON Preview: {json_string[:50]}...")
        
        result = validate_prompts_for_api(json_string)
        logger.info(f"✅ Output: {len(result)} validated prompt(s)")
        logger.info("🔄 JSON → Dict conversion successful!")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == SIMPLE_WORKFLOW
        logger.info("🎉 JSON string validation PASSED!")
    
    def test_validate_json_string_prompt_list(self):
        """Test validation of JSON string containing list of prompts."""
        
        prompt_list = [SIMPLE_WORKFLOW, AUDIO_WORKFLOW]
        json_string = json.dumps(prompt_list)
        result = validate_prompts_for_api(json_string)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == SIMPLE_WORKFLOW
        assert result[1] == AUDIO_WORKFLOW
    
    def test_validate_mixed_json_strings_in_list(self):
        """Test validation of list containing both dicts and JSON strings."""
        logger.info("🎯 Testing MIXED list validation (dicts + JSON strings)...")
        
        logger.info("✅ Using pytrickle parse_prompts_field")
        
        # Test directly with the parse function that handles mixed types
        mixed_list = [
            json.dumps(SIMPLE_WORKFLOW),  # JSON string
            COMPLEX_WORKFLOW,             # Dict
            json.dumps(AUDIO_WORKFLOW)    # JSON string
        ]
        
        logger.info("📝 Input: Mixed list with 3 items:")
        logger.info("   📄 [0]: JSON string (SIMPLE_WORKFLOW)")
        logger.info("   📊 [1]: Dict object (COMPLEX_WORKFLOW)")  
        logger.info("   📄 [2]: JSON string (AUDIO_WORKFLOW)")
        
        result = parse_prompts_field(mixed_list)
        
        logger.info(f"✅ Output: {len(result)} parsed prompts")
        logger.info("🔄 Mixed format parsing successful!")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == SIMPLE_WORKFLOW
        assert result[1] == COMPLEX_WORKFLOW
        assert result[2] == AUDIO_WORKFLOW
        logger.info("🎉 Mixed format validation PASSED!")
    
    def test_invalid_prompt_formats(self):
        """Test validation with invalid prompt formats."""
        
        # Test invalid types
        with pytest.raises(ValueError, match="Invalid prompts"):
            validate_prompts_for_api(123)
        
        with pytest.raises(ValueError, match="Invalid prompts"):
            validate_prompts_for_api(True)
        
        with pytest.raises(ValueError, match="Invalid prompts"):
            validate_prompts_for_api(None)
    
    def test_invalid_json_strings(self):
        """Test validation with invalid JSON strings."""
        
        # Invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_prompts_for_api('{"invalid": json}')
        
        # JSON with invalid structure
        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_prompts_for_api('{"unclosed": "dict"')
    
    def test_empty_prompts_validation(self):
        """Test validation with empty prompt structures."""
        
        # Empty dict
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompts_for_api({})
        
        # Empty list
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompts_for_api([])
    
    def test_invalid_prompt_structure(self):
        """Test validation with invalid ComfyUI prompt structure."""
        
        # Missing class_type
        invalid_prompt = {
            "1": {
                "inputs": {},
                "_meta": {"title": "Test"}
                # Missing class_type
            }
        }
        
        with pytest.raises(ValueError, match="Invalid prompts"):
            validate_prompts_for_api(invalid_prompt)
        
        # Non-string node ID
        invalid_prompt_2 = {
            123: {  # Should be string
                "inputs": {},
                "class_type": "TestNode",
                "_meta": {"title": "Test"}
            }
        }
        
        with pytest.raises(ValueError, match="Invalid prompts"):
            validate_prompts_for_api(invalid_prompt_2)
        
        # Non-dict node data
        invalid_prompt_3 = {
            "1": "not_a_dict"  # Should be dict
        }
        
        with pytest.raises(ValueError, match="Invalid prompts"):
            validate_prompts_for_api(invalid_prompt_3)


class TestComfyStreamServerIntegration:
    """Test prompt validation integration with ComfyStream server endpoints."""
    
    def test_server_endpoint_prompt_validation(self):
        """Test that server endpoints can validate prompts using pytrickle validation."""
        logger.info("🎯 Testing SERVER ENDPOINT integration...")
        
        logger.info("✅ pytrickle validation available for server")
        
        # Simulate server endpoint receiving prompts
        incoming_params = {
            "prompts": json.dumps(SIMPLE_WORKFLOW),
            "width": 512,
            "height": 512
        }
        
        logger.info("🌐 Simulating server receiving request:")
        logger.info(f"   📏 width: {incoming_params['width']}")
        logger.info(f"   📏 height: {incoming_params['height']}")
        logger.info(f"   📝 prompts: JSON string ({len(incoming_params['prompts'])} chars)")
        
        # Validate prompts as server would
        logger.info("🔍 Server validating prompts...")
        validated_prompts = validate_prompts_for_api(incoming_params["prompts"])
        
        logger.info(f"✅ Server validation result: {len(validated_prompts)} validated prompt(s)")
        logger.info("🎯 Server endpoint integration working!")
        
        assert isinstance(validated_prompts, list)
        assert len(validated_prompts) == 1
        assert validated_prompts[0] == SIMPLE_WORKFLOW
        logger.info("🎉 Server endpoint validation PASSED!")
    
    def test_server_endpoint_error_handling(self):
        """Test server endpoint error handling for invalid prompts."""
        
        # Simulate server receiving invalid prompts
        invalid_params = {
            "prompts": "invalid json",
            "width": 512,
            "height": 512
        }
        
        with pytest.raises(ValueError):
            validate_prompts_for_api(invalid_params["prompts"])


class TestPromptValidationIntegration:
    """Test integration scenarios with prompt validation."""
    
    def test_extract_prompts_from_params_function(self):
        """Test the extract_prompts_from_params utility function."""
        
        # Test with prompts
        params_with_prompts = {
            "prompts": SIMPLE_WORKFLOW,
            "width": 512,
            "height": 512
        }
        
        result = extract_prompts_from_params(params_with_prompts)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == SIMPLE_WORKFLOW
        
        # Test without prompts
        params_without_prompts = {
            "width": 512,
            "height": 512,
            "intensity": 0.8
        }
        
        result = extract_prompts_from_params(params_without_prompts)
        assert result is None
    
    def test_prompt_validation_consistency(self):
        """Test consistent prompt validation across different input formats."""
            
        test_prompts = [SIMPLE_WORKFLOW, AUDIO_WORKFLOW]
        
        # Test as list
        result1 = validate_prompts_for_api(test_prompts)
        
        # Test as JSON string  
        result2 = validate_prompts_for_api(json.dumps(test_prompts))
        
        # Both should be equivalent
        assert result1 == result2
        assert len(result1) == 2
        assert len(result2) == 2
    
    def test_real_world_workflow_validation(self):
        """Test with realistic ComfyUI workflow scenarios."""
        logger.info("🎯 Testing REAL-WORLD ComfyUI workflow validation...")
        
        logger.info("✅ pytrickle validation available")
            
        logger.info("🎨 Creating realistic Stable Diffusion workflow...")
        # Simulate a real SD workflow from ComfyUI
        realistic_sd_workflow = {
            "1": {
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint - BASE"}
            },
            "2": {
                "inputs": {
                    "text": "a majestic mountain landscape at sunset, highly detailed, 8k",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"}
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative)"}
            },
            "4": {
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            "5": {
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            }
        }
        
        logger.info(f"📊 Workflow stats: {len(realistic_sd_workflow)} nodes")
        logger.info("   🔧 CheckpointLoaderSimple (model loading)")
        logger.info("   ✏️  CLIPTextEncode (positive prompt)")
        logger.info("   ⛔ CLIPTextEncode (negative prompt)")
        logger.info("   🖼️  EmptyLatentImage (canvas)")
        logger.info("   🎲 KSampler (generation)")
        
        # Test validation
        logger.info("🔍 Validating complex SD workflow...")
        validated_prompts = validate_prompts_for_api(realistic_sd_workflow)
        
        logger.info(f"✅ Validation result: {len(validated_prompts)} validated prompt(s)")
        logger.info("🎨 Complex workflow validation successful!")
        
        assert isinstance(validated_prompts, list)
        assert len(validated_prompts) == 1
        assert "CheckpointLoaderSimple" in str(validated_prompts[0])
        assert "KSampler" in str(validated_prompts[0])
        logger.info("🎉 Real-world workflow validation PASSED!")
    
    def test_error_handling_with_partial_valid_data(self):
        """Test error handling when some data is valid but prompts are invalid."""
            
        # Test invalid JSON should raise error
        with pytest.raises(ValueError):
            validate_prompts_for_api('{"invalid": json syntax}')


class TestBackwardCompatibility:
    """Test backward compatibility with existing prompt parsing functions."""
    
    def test_parsing_functions_handle_legacy_formats(self):
        """Test that prompt parsing functions handle various legacy formats."""
        
        # Test various formats that might come from different sources
        formats_to_test = [
            SIMPLE_WORKFLOW,  # Direct dict
            [SIMPLE_WORKFLOW],  # List of dicts
            json.dumps(SIMPLE_WORKFLOW),  # JSON string
            json.dumps([SIMPLE_WORKFLOW])  # JSON string of list
        ]
        
        for test_format in formats_to_test:
            result = parse_prompts_field(test_format)
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == SIMPLE_WORKFLOW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
