"""
Tests for detect_out_resolution feature.

This feature enables Super Resolution workflows by allowing the encoder to
detect output resolution from the first processed frame's tensor shape,
rather than using the input resolution from the decoder.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, call
from fractions import Fraction

from pytrickle.api import StreamStartRequest, StreamParamsUpdateRequest
from pytrickle.frames import VideoOutput, AudioOutput, VideoFrame, AudioFrame
from pytrickle.encoder import encode_av, default_output_metadata


class TestAutoDetectOutputResolutionAPI:
    """Test API validation for detect_out_resolution parameter."""

    def test_auto_detect_default_not_in_params(self):
        """Test that detect_out_resolution is not required in params."""
        request = StreamStartRequest(
            gateway_request_id="test",
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            params={"width": 512, "height": 512}
        )
        # Should not raise, detect_out_resolution defaults to True in server
        assert request.params.get("detect_out_resolution") is None

    def test_auto_detect_true_boolean(self):
        """Test detect_out_resolution with True boolean value."""
        request = StreamStartRequest(
            gateway_request_id="test",
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            params={"width": 512, "height": 512, "detect_out_resolution": True}
        )
        assert request.params["detect_out_resolution"] is True

    def test_auto_detect_false_boolean(self):
        """Test detect_out_resolution with False boolean value."""
        request = StreamStartRequest(
            gateway_request_id="test",
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            params={"width": 512, "height": 512, "detect_out_resolution": False}
        )
        assert request.params["detect_out_resolution"] is False

    def test_auto_detect_string_true_values(self):
        """Test detect_out_resolution with string 'true' variants."""
        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
            request = StreamStartRequest(
                gateway_request_id="test",
                subscribe_url="http://localhost/input",
                publish_url="http://localhost/output",
                params={"width": 512, "height": 512, "detect_out_resolution": value}
            )
            assert request.params["detect_out_resolution"] is True, f"Failed for value: {value}"

    def test_auto_detect_string_false_values(self):
        """Test detect_out_resolution with string 'false' variants."""
        for value in ["false", "False", "FALSE", "0", "no", "No", "NO"]:
            request = StreamStartRequest(
                gateway_request_id="test",
                subscribe_url="http://localhost/input",
                publish_url="http://localhost/output",
                params={"width": 512, "height": 512, "detect_out_resolution": value}
            )
            assert request.params["detect_out_resolution"] is False, f"Failed for value: {value}"

    def test_auto_detect_int_values(self):
        """Test detect_out_resolution with integer values."""
        # 1 should be True
        request = StreamStartRequest(
            gateway_request_id="test",
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            params={"width": 512, "height": 512, "detect_out_resolution": 1}
        )
        assert request.params["detect_out_resolution"] is True

        # 0 should be False
        request = StreamStartRequest(
            gateway_request_id="test",
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            params={"width": 512, "height": 512, "detect_out_resolution": 0}
        )
        assert request.params["detect_out_resolution"] is False

    def test_auto_detect_invalid_string_raises(self):
        """Test that invalid string values raise ValueError."""
        with pytest.raises(ValueError, match="detect_out_resolution must be a boolean"):
            StreamStartRequest(
                gateway_request_id="test",
                subscribe_url="http://localhost/input",
                publish_url="http://localhost/output",
                params={"width": 512, "height": 512, "detect_out_resolution": "invalid"}
            )

    def test_auto_detect_invalid_type_raises(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError, match="detect_out_resolution must be a boolean"):
            StreamStartRequest(
                gateway_request_id="test",
                subscribe_url="http://localhost/input",
                publish_url="http://localhost/output",
                params={"width": 512, "height": 512, "detect_out_resolution": [True]}
            )


class TestEncoderAutoDetectResolution:
    """Test encoder behavior with detect_out_resolution."""

    def create_video_output(self, height: int, width: int, channels: int = 3) -> VideoOutput:
        """Create a VideoOutput with specified dimensions."""
        # Tensor shape: [B, H, W, C] -> squeeze to [H, W, C]
        tensor = torch.rand(1, height, width, channels)
        frame = VideoFrame(tensor, timestamp=0, time_base=Fraction(1, 90000))
        return VideoOutput(frame, request_id="test")

    def test_auto_detect_enabled_uses_tensor_dimensions(self):
        """Test that with auto_detect=True, encoder uses tensor dimensions."""
        # Input dimensions from decoder metadata
        input_width, input_height = 512, 512
        # Output dimensions from Super Resolution processor (2x upscale)
        output_width, output_height = 1024, 1024

        metadata = {
            'video': {
                'target_width': input_width,
                'target_height': input_height,
            },
            'audio': None
        }

        frames_processed = []
        stream_created_with = {}

        def mock_input_queue():
            if len(frames_processed) == 0:
                frames_processed.append(1)
                return self.create_video_output(output_height, output_width)
            return None

        def mock_output_callback(read_file, write_file, url):
            pass

        def mock_get_metadata():
            return metadata

        # Patch av.open and add_stream to capture stream creation parameters
        with patch('pytrickle.encoder.av.open') as mock_av_open:
            mock_container = MagicMock()
            mock_av_open.return_value = mock_container

            def capture_add_stream(codec, options=None):
                stream_created_with['codec'] = codec
                stream_created_with['options'] = options
                mock_stream = MagicMock()
                mock_stream.time_base = Fraction(1, 90000)
                mock_stream.codec_context.time_base = Fraction(1, 90000)
                mock_stream.encode.return_value = []
                return mock_stream

            mock_container.add_stream.side_effect = capture_add_stream

            encode_av(
                input_queue=mock_input_queue,
                output_callback=mock_output_callback,
                get_metadata=mock_get_metadata,
                video_codec='libx264',
                audio_codec=None,
                detect_out_resolution=True
            )

        # Verify stream was created with output (tensor) dimensions, not input dimensions
        assert 'options' in stream_created_with
        assert stream_created_with['options']['video_size'] == f'{output_width}x{output_height}'

    def test_auto_detect_disabled_uses_metadata_dimensions(self):
        """Test that with auto_detect=False, encoder uses metadata dimensions."""
        # Input dimensions from decoder metadata
        input_width, input_height = 512, 512
        # Output dimensions from Super Resolution processor (would be ignored)
        output_width, output_height = 1024, 1024

        metadata = {
            'video': {
                'target_width': input_width,
                'target_height': input_height,
            },
            'audio': None
        }

        frames_processed = []
        stream_created_with = {}

        def mock_input_queue():
            if len(frames_processed) == 0:
                frames_processed.append(1)
                return self.create_video_output(output_height, output_width)
            return None

        def mock_output_callback(read_file, write_file, url):
            pass

        def mock_get_metadata():
            return metadata

        with patch('pytrickle.encoder.av.open') as mock_av_open:
            mock_container = MagicMock()
            mock_av_open.return_value = mock_container

            def capture_add_stream(codec, options=None):
                stream_created_with['codec'] = codec
                stream_created_with['options'] = options
                mock_stream = MagicMock()
                mock_stream.time_base = Fraction(1, 90000)
                mock_stream.codec_context.time_base = Fraction(1, 90000)
                mock_stream.encode.return_value = []
                return mock_stream

            mock_container.add_stream.side_effect = capture_add_stream

            encode_av(
                input_queue=mock_input_queue,
                output_callback=mock_output_callback,
                get_metadata=mock_get_metadata,
                video_codec='libx264',
                audio_codec=None,
                detect_out_resolution=False  # Disabled
            )

        # Verify stream was created with input (metadata) dimensions
        assert 'options' in stream_created_with
        assert stream_created_with['options']['video_size'] == f'{input_width}x{input_height}'

    def test_auto_detect_logs_resolution_difference(self):
        """Test that encoder logs when detected resolution differs from input."""
        input_width, input_height = 512, 512
        output_width, output_height = 1024, 1024

        metadata = {
            'video': {
                'target_width': input_width,
                'target_height': input_height,
            },
            'audio': None
        }

        frames_processed = []

        def mock_input_queue():
            if len(frames_processed) == 0:
                frames_processed.append(1)
                return self.create_video_output(output_height, output_width)
            return None

        def mock_get_metadata():
            return metadata

        with patch('pytrickle.encoder.av.open') as mock_av_open, \
             patch('pytrickle.encoder.logger') as mock_logger:
            mock_container = MagicMock()
            mock_av_open.return_value = mock_container

            mock_stream = MagicMock()
            mock_stream.time_base = Fraction(1, 90000)
            mock_stream.codec_context.time_base = Fraction(1, 90000)
            mock_stream.encode.return_value = []
            mock_container.add_stream.return_value = mock_stream

            encode_av(
                input_queue=mock_input_queue,
                output_callback=lambda *args: None,
                get_metadata=mock_get_metadata,
                video_codec='libx264',
                audio_codec=None,
                detect_out_resolution=True
            )

        # Check that the resolution difference was logged
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        resolution_logged = any(
            'Auto-detected output resolution' in str(c) and 
            '1024x1024' in str(c) and 
            '512x512' in str(c)
            for c in info_calls
        )
        assert resolution_logged, f"Expected resolution difference log, got: {info_calls}"

    def test_auto_detect_same_resolution_no_extra_log(self):
        """Test that encoder doesn't log 'differs' when resolutions match."""
        width, height = 512, 512

        metadata = {
            'video': {
                'target_width': width,
                'target_height': height,
            },
            'audio': None
        }

        frames_processed = []

        def mock_input_queue():
            if len(frames_processed) == 0:
                frames_processed.append(1)
                return self.create_video_output(height, width)
            return None

        def mock_get_metadata():
            return metadata

        with patch('pytrickle.encoder.av.open') as mock_av_open, \
             patch('pytrickle.encoder.logger') as mock_logger:
            mock_container = MagicMock()
            mock_av_open.return_value = mock_container

            mock_stream = MagicMock()
            mock_stream.time_base = Fraction(1, 90000)
            mock_stream.codec_context.time_base = Fraction(1, 90000)
            mock_stream.encode.return_value = []
            mock_container.add_stream.return_value = mock_stream

            encode_av(
                input_queue=mock_input_queue,
                output_callback=lambda *args: None,
                get_metadata=mock_get_metadata,
                video_codec='libx264',
                audio_codec=None,
                detect_out_resolution=True
            )

        # Check that no 'differs' message was logged
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        differs_logged = any('differs from input' in str(c) for c in info_calls)
        assert not differs_logged, f"Unexpected 'differs' log when resolutions match: {info_calls}"


class TestSuperResolutionWorkflow:
    """Integration-style tests for Super Resolution workflow scenarios."""

    def test_2x_upscale_scenario(self):
        """Test typical 2x Super Resolution upscale scenario."""
        # Simulate 720p input being upscaled to 1440p
        input_width, input_height = 1280, 720
        output_width, output_height = 2560, 1440  # 2x upscale

        metadata = {
            'video': {
                'target_width': input_width,
                'target_height': input_height,
            },
            'audio': None
        }

        frames = []
        stream_config = {}

        def mock_input_queue():
            if len(frames) < 3:
                # Super Resolution processor outputs 2x upscaled frames
                tensor = torch.rand(1, output_height, output_width, 3)
                frame = VideoFrame(tensor, timestamp=len(frames) * 3000, time_base=Fraction(1, 90000))
                frames.append(frame)
                return VideoOutput(frame, request_id="test")
            return None

        with patch('pytrickle.encoder.av.open') as mock_av_open:
            mock_container = MagicMock()
            mock_av_open.return_value = mock_container

            def capture_add_stream(codec, options=None):
                stream_config['options'] = options
                mock_stream = MagicMock()
                mock_stream.time_base = Fraction(1, 90000)
                mock_stream.codec_context.time_base = Fraction(1, 90000)
                mock_stream.encode.return_value = []
                return mock_stream

            mock_container.add_stream.side_effect = capture_add_stream

            encode_av(
                input_queue=mock_input_queue,
                output_callback=lambda *args: None,
                get_metadata=lambda: metadata,
                video_codec='libx264',
                audio_codec=None,
                detect_out_resolution=True
            )

        # Verify encoder was configured for 2x output resolution
        assert stream_config['options']['video_size'] == f'{output_width}x{output_height}'
        # Verify all frames were processed
        assert len(frames) == 3

    def test_4x_upscale_scenario(self):
        """Test 4x Super Resolution upscale scenario."""
        # Simulate 480p input being upscaled to 1920p (4x)
        input_width, input_height = 640, 480
        output_width, output_height = 2560, 1920  # 4x upscale

        metadata = {
            'video': {
                'target_width': input_width,
                'target_height': input_height,
            },
            'audio': None
        }

        stream_config = {}
        frame_count = [0]

        def mock_input_queue():
            if frame_count[0] == 0:
                frame_count[0] += 1
                tensor = torch.rand(1, output_height, output_width, 3)
                frame = VideoFrame(tensor, timestamp=0, time_base=Fraction(1, 90000))
                return VideoOutput(frame, request_id="test")
            return None

        with patch('pytrickle.encoder.av.open') as mock_av_open:
            mock_container = MagicMock()
            mock_av_open.return_value = mock_container

            def capture_add_stream(codec, options=None):
                stream_config['options'] = options
                mock_stream = MagicMock()
                mock_stream.time_base = Fraction(1, 90000)
                mock_stream.codec_context.time_base = Fraction(1, 90000)
                mock_stream.encode.return_value = []
                return mock_stream

            mock_container.add_stream.side_effect = capture_add_stream

            encode_av(
                input_queue=mock_input_queue,
                output_callback=lambda *args: None,
                get_metadata=lambda: metadata,
                video_codec='libx264',
                audio_codec=None,
                detect_out_resolution=True
            )

        assert stream_config['options']['video_size'] == f'{output_width}x{output_height}'


class TestProtocolAutoDetectParameter:
    """Test TrickleProtocol detect_out_resolution parameter."""

    def test_protocol_default_auto_detect_true(self):
        """Test that TrickleProtocol defaults detect_out_resolution to True."""
        from pytrickle.protocol import TrickleProtocol

        protocol = TrickleProtocol(
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
        )
        assert protocol.detect_out_resolution is True

    def test_protocol_auto_detect_can_be_disabled(self):
        """Test that detect_out_resolution can be set to False."""
        from pytrickle.protocol import TrickleProtocol

        protocol = TrickleProtocol(
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            detect_out_resolution=False,
        )
        assert protocol.detect_out_resolution is False

    def test_protocol_auto_detect_can_be_enabled_explicitly(self):
        """Test that detect_out_resolution can be set to True explicitly."""
        from pytrickle.protocol import TrickleProtocol

        protocol = TrickleProtocol(
            subscribe_url="http://localhost/input",
            publish_url="http://localhost/output",
            detect_out_resolution=True,
        )
        assert protocol.detect_out_resolution is True
