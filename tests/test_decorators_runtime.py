import logging
from fractions import Fraction

import numpy as np
import pytest
import torch

from pytrickle.decorators import (
    video_handler,
    audio_handler,
)
from pytrickle.frames import VideoFrame, AudioFrame


@pytest.mark.asyncio
async def test_video_handler_metadata_and_tensor_normalization():
    # Define video handler that returns a torch tensor (same shape)
    @video_handler
    async def brighten(frame: VideoFrame):
        return frame.tensor + 0.1

    assert getattr(brighten, "_trickle_handler", False)
    info = getattr(brighten, "_trickle_handler_info", None)
    assert info is not None
    assert info.handler_type == "video"
    assert isinstance(info.description, str)
    assert info.description.startswith("Video handler:")

    # Make a test frame
    tensor = torch.zeros((64, 64, 3), dtype=torch.float32)
    frame = VideoFrame.from_tensor(tensor, timestamp=123)

    out = await brighten(frame)
    assert isinstance(out, VideoFrame)
    assert out.timestamp == frame.timestamp
    assert out.time_base == frame.time_base
    assert torch.allclose(out.tensor, tensor + 0.1)


@pytest.mark.asyncio
async def test_video_handler_numpy_normalization_to_torch():
    # Define video handler that returns a numpy array (same shape)
    @video_handler
    async def as_numpy(frame: VideoFrame):
        return (frame.tensor.detach().cpu().numpy()).astype(np.float32)

    # Make a test frame
    tensor = torch.zeros((32, 32, 3), dtype=torch.float32)
    frame = VideoFrame.from_tensor(tensor)

    out = await as_numpy(frame)
    assert isinstance(out, VideoFrame)
    assert out.tensor.shape == tensor.shape
    # The wrapper converts numpy to torch via from_numpy, so values should match
    assert torch.allclose(out.tensor, tensor)


@pytest.mark.asyncio
async def test_audio_handler_tensor_and_list_return_shapes():
    # Handler returns tensor of samples to be normalized into AudioFrame list
    @audio_handler
    async def louder(frame: AudioFrame):
        # Create a small mono signal
        samples = torch.ones((1, 256), dtype=torch.float32) * 0.5
        return samples

    # Build an AudioFrame using utility (no av dependency path)
    base = torch.zeros((1, 256), dtype=torch.float32)
    in_frame = AudioFrame.from_tensor(base, sample_rate=16000, timestamp=42)

    out_list = await louder(in_frame)
    assert isinstance(out_list, list)
    assert len(out_list) == 1
    out = out_list[0]
    assert isinstance(out, AudioFrame)
    assert out.timestamp == in_frame.timestamp
    assert out.rate == in_frame.rate
    # Ensure samples were replaced (non-zero now)
    assert np.allclose(out.samples, np.ones_like(out.samples) * out.samples.max()) is False or np.any(out.samples != in_frame.samples)

    # Also check handler that returns a single AudioFrame is wrapped into list
    @audio_handler
    async def identity(frame: AudioFrame):
        return frame

    out_list2 = await identity(in_frame)
    assert isinstance(out_list2, list)
    assert len(out_list2) == 1
    assert isinstance(out_list2[0], AudioFrame)


def test_invalid_video_signature_logs_warning(caplog):
    caplog.set_level(logging.WARNING, logger="pytrickle.decorators")

    async def bad(x: int):  # wrong param name/type
        return None

    # Apply decorator dynamically to capture logs during decoration
    decorated = video_handler(bad)  # type: ignore
    assert callable(decorated)

    # Ensure a warning about signature was emitted
    msgs = [r.message for r in caplog.records]
    assert any("has no parameter matching VideoFrame/AudioFrame" in str(m) for m in msgs)


@pytest.mark.asyncio
async def test_video_handler_invalid_shape_returns_none(caplog):
    # Handler returns wrong-shaped tensor (channels = 2), should not be normalized
    @video_handler
    async def bad_shape_torch(frame: VideoFrame):
        h, w = frame.tensor.shape[0], frame.tensor.shape[1]
        return torch.zeros((h, w, 2), dtype=torch.float32)

    base = torch.zeros((10, 12, 3), dtype=torch.float32)
    frame = VideoFrame.from_tensor(base)

    caplog.set_level(logging.WARNING, logger="pytrickle.decorators")
    out = await bad_shape_torch(frame)
    assert out is None  # wrapper declines to normalize invalid shapes
    # Ensure a breadcrumb warning was emitted
    msgs = [r.message for r in caplog.records]
    assert any("unexpected shape" in str(m) for m in msgs)

    # Same with numpy
    @video_handler
    async def bad_shape_numpy(frame: VideoFrame):
        h, w = frame.tensor.shape[0], frame.tensor.shape[1]
        import numpy as _np
        return _np.zeros((h, w, 2), dtype=_np.float32)

    caplog.clear()
    out2 = await bad_shape_numpy(frame)
    assert out2 is None
    msgs2 = [r.message for r in caplog.records]
    assert any("unexpected shape" in str(m) for m in msgs2)


@pytest.mark.asyncio
async def test_audio_handler_mismatched_return_results_in_none(caplog):
    # Handler returns an unsupported type (string)
    @audio_handler
    async def bad_return(frame: AudioFrame):
        return "not-valid"

    base = torch.zeros((1, 128), dtype=torch.float32)
    in_frame = AudioFrame.from_tensor(base, sample_rate=16000, timestamp=1)

    caplog.set_level(logging.WARNING, logger="pytrickle.decorators")
    out = await bad_return(in_frame)
    # Decorator wrapper should return None for unsupported types
    assert out is None
    # Ensure a rate-limited warning is emitted for unsupported types
    msgs = [r.message for r in caplog.records]
    assert any("Audio handler 'bad_return' returned unsupported type" in str(m) for m in msgs)
