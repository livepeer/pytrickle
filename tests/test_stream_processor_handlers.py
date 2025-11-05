from typing import Any, Dict, Optional

import pytest
import torch

from pytrickle.decorators import (
    video_handler,
    audio_handler,
    param_updater,
    on_stream_start,
    on_stream_stop,
)
from pytrickle.frames import VideoFrame, AudioFrame
from pytrickle.stream_processor import StreamProcessor


class MyHandlers:
    def __init__(self) -> None:
        self.started: bool = False
        self.stopped: bool = False
        self.params: Optional[Dict[str, Any]] = None

    @video_handler
    async def handle_video(self, frame: VideoFrame) -> VideoFrame:
        return frame.replace_tensor(frame.tensor + 0.2)

    @param_updater
    async def update(self, params: Dict[str, Any]) -> None:
        self.params = params

    @on_stream_start
    async def start(self) -> None:
        self.started = True

    @on_stream_stop
    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_from_handlers_discovers_bound_methods_and_runs():
    inst = MyHandlers()

    sp = StreamProcessor.from_handlers(inst, validate_signature=True)

    # Lifecycle
    await sp._frame_processor.on_stream_start()
    assert inst.started is True

    # Video processing
    base = torch.zeros((16, 16, 3), dtype=torch.float32)
    frame = VideoFrame.from_tensor(base, timestamp=7)
    out = await sp._frame_processor.process_video_async(frame)
    assert isinstance(out, VideoFrame)
    assert torch.allclose(out.tensor, base + 0.2)

    # Params update
    params = {"alpha": 0.5, "beta": 2}
    await sp._frame_processor.update_params(params)
    assert inst.params == params

    # Lifecycle stop
    await sp._frame_processor.on_stream_stop()
    assert inst.stopped is True


def test_stream_processor_rejects_sync_video_processor():
    # Define a sync function intentionally
    def sync_video(frame: VideoFrame) -> VideoFrame:
        return frame

    with pytest.raises(ValueError):
        StreamProcessor(video_processor=sync_video)  # type: ignore


class _HandlersReturnNone:
    @video_handler
    async def handle_video(self, frame: VideoFrame):
        return None


@pytest.mark.asyncio
async def test_internal_processor_video_none_passthrough():
    inst = _HandlersReturnNone()
    sp = StreamProcessor.from_handlers(inst)
    base = torch.rand((8, 8, 3), dtype=torch.float32)
    frame = VideoFrame.from_tensor(base, timestamp=1)
    out = await sp._frame_processor.process_video_async(frame)
    # With current semantics, None from handler results in pass-through
    assert isinstance(out, VideoFrame)
    assert torch.allclose(out.tensor, base)


class _HandlersWrongShape:
    @video_handler
    async def handle_video(self, frame: VideoFrame):
        # Return a tensor with invalid last channel dimension (2), should pass-through
        h, w = frame.tensor.shape[0], frame.tensor.shape[1]
        return torch.zeros((h, w, 2), dtype=torch.float32)


@pytest.mark.asyncio
async def test_internal_processor_wrong_shape_passthrough():
    inst = _HandlersWrongShape()
    sp = StreamProcessor.from_handlers(inst)
    base = torch.rand((9, 7, 3), dtype=torch.float32)
    frame = VideoFrame.from_tensor(base, timestamp=5)
    out = await sp._frame_processor.process_video_async(frame)
    # Invalid shape from handler should result in pass-through of original frame
    assert isinstance(out, VideoFrame)
    assert torch.allclose(out.tensor, base)


class _AudioHandlersWrongType:
    @audio_handler
    async def handle_audio(self, frame: AudioFrame):
        # Return wrong type; internal processor should pass through original
        return {"unexpected": True}


@pytest.mark.asyncio
async def test_internal_processor_audio_wrong_type_passthrough():
    inst = _AudioHandlersWrongType()
    sp = StreamProcessor.from_handlers(inst)
    base = torch.zeros((1, 256), dtype=torch.float32)
    frame = AudioFrame.from_tensor(base, sample_rate=16000, timestamp=3)
    out_list = await sp._frame_processor.process_audio_async(frame)
    assert isinstance(out_list, list) and len(out_list) == 1
    out = out_list[0]
    # Should be the original frame unchanged
    assert isinstance(out, AudioFrame)
    assert out.timestamp == frame.timestamp
    assert out.rate == frame.rate
    assert out.samples.shape == frame.samples.shape
