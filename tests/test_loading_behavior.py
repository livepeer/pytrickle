import pytest
import torch

from fractions import Fraction
from types import MethodType

from pytrickle.client import TrickleClient
from pytrickle.frames import VideoFrame
from pytrickle.stream_processor import _InternalFrameProcessor, VideoProcessingResult
from pytrickle.frame_overlay import OverlayConfig, OverlayMode


def _make_video_frame(value: float = 0.5, timestamp: int = 0) -> VideoFrame:
    tensor = torch.full((3, 4, 4), value, dtype=torch.float32)
    return VideoFrame(tensor=tensor, timestamp=timestamp, time_base=Fraction(1, 30))


async def _noop_audio_handler(frame):
    return [frame]


async def _noop_param_updater(params):
    return None


async def _video_processor(frame: VideoFrame) -> VideoFrame:
    tensor = frame.tensor + 1.0
    return frame.replace_tensor(tensor)


class _DummyProtocol:
    fps_meter = None
    error_callback = None

    async def start(self):
        return None

    async def stop(self):
        return None


async def _run_single_frame(processor, frame, overlay_config=None):
    """Push a single frame through the TrickleClient video loop."""
    client = TrickleClient(
        protocol=_DummyProtocol(),
        frame_processor=processor,
        overlay_config=overlay_config
    )
    await client.video_input_queue.put(frame)
    await client.video_input_queue.put(None)
    await client._process_video_frames()
    return await client.output_queue.get()


@pytest.mark.asyncio
async def test_overlay_mode_injects_overlay_frame(monkeypatch):
    overlay_marker = _make_video_frame(9.0)

    def fake_overlay(*, original_frame, message, frame_counter, progress):
        return overlay_marker

    monkeypatch.setattr(
        "pytrickle.frame_overlay.build_frame_overlay",
        fake_overlay,
    )

    async def withheld_processor(frame: VideoFrame):
        return VideoProcessingResult.WITHHELD

    processor = _InternalFrameProcessor(
        video_processor=withheld_processor,
        audio_processor=_noop_audio_handler,
        model_loader=None,
        param_updater=_noop_param_updater,
        name="test-overlay",
    )
    
    # Create loading config for manual overlay mode
    overlay_config = OverlayConfig(
        mode=OverlayMode.OVERLAY,
        enabled=True,
        auto_timeout_seconds=None  # Disable auto-timeout for manual mode
    )
    
    # Create client with manual loading enabled
    client = TrickleClient(
        protocol=_DummyProtocol(),
        frame_processor=processor,
        overlay_config=overlay_config
    )
    client.loading_controller.set_manual_loading(True)
    
    await client.video_input_queue.put(_make_video_frame(1.0, timestamp=123))
    await client.video_input_queue.put(None)
    await client._process_video_frames()
    output = await client.output_queue.get()
    
    assert torch.allclose(output.frame.tensor, overlay_marker.tensor)


@pytest.mark.asyncio
async def test_passthrough_mode_keeps_processed_frame():
    processor = _InternalFrameProcessor(
        video_processor=_video_processor,
        audio_processor=_noop_audio_handler,
        model_loader=None,
        param_updater=_noop_param_updater,
        name="test-passthrough",
    )
    
    # Create loading config for passthrough mode
    overlay_config = OverlayConfig(
        mode=OverlayMode.PASSTHROUGH,
        enabled=True
    )
    
    # Create client with manual loading enabled in passthrough mode
    client = TrickleClient(
        protocol=_DummyProtocol(),
        frame_processor=processor,
        overlay_config=overlay_config
    )
    client.loading_controller.set_manual_loading(True)
    
    frame = _make_video_frame(2.0, timestamp=456)
    await client.video_input_queue.put(frame)
    await client.video_input_queue.put(None)
    await client._process_video_frames()
    output = await client.output_queue.get()
    
    # Processed frame should be the handler result (value + 1)
    assert torch.allclose(output.frame.tensor, frame.tensor + 1.0)


@pytest.mark.asyncio
async def test_auto_overlay_engages_when_frames_stall(monkeypatch):
    overlay_marker = _make_video_frame(7.0)

    def fake_overlay(**kwargs):
        return overlay_marker

    monkeypatch.setattr(
        "pytrickle.frame_overlay.build_frame_overlay",
        fake_overlay,
    )

    async def stalled_processor(frame: VideoFrame):
        return VideoProcessingResult.WITHHELD

    processor = _InternalFrameProcessor(
        video_processor=stalled_processor,
        audio_processor=_noop_audio_handler,
        model_loader=None,
        param_updater=_noop_param_updater,
        name="test-auto-overlay",
    )
    
    # Create loading config with auto-timeout set to 0 (immediate)
    overlay_config = OverlayConfig(
        mode=OverlayMode.OVERLAY,
        message="Waiting...",
        enabled=True,
        auto_timeout_seconds=0.0
    )

    output = await _run_single_frame(
        processor, _make_video_frame(1.0, timestamp=789), overlay_config=overlay_config
    )
    assert torch.allclose(output.frame.tensor, overlay_marker.tensor)


@pytest.mark.asyncio
async def test_client_start_resets_auto_loading_state(monkeypatch):
    processor = _InternalFrameProcessor(
        video_processor=_video_processor,
        audio_processor=_noop_audio_handler,
        model_loader=None,
        param_updater=_noop_param_updater,
        name="test-reset",
    )

    client = TrickleClient(protocol=_DummyProtocol(), frame_processor=processor)

    async def noop_loop(self):
        return None

    for loop_name in (
        "_ingress_loop",
        "_processing_loop",
        "_egress_loop",
        "_control_loop",
        "_send_data_loop",
    ):
        monkeypatch.setattr(client, loop_name, MethodType(noop_loop, client))

    # Access loading controller state (it's now in the controller, not the client)
    client.loading_controller._loading_active = True
    client.loading_controller._last_video_frame_time = -1.0

    await client.start("first-run")

    assert client.loading_controller._loading_active is False
    assert client.loading_controller._last_video_frame_time != -1.0

    client.loading_controller._loading_active = True
    client.loading_controller._last_video_frame_time = -5.0

    await client.start("second-run")

    assert client.loading_controller._loading_active is False
    assert client.loading_controller._last_video_frame_time != -5.0
