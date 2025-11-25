import asyncio

import pytest

from pytrickle.frame_processor import FrameProcessor
from pytrickle.stream_processor import StreamProcessor
from pytrickle.state import StreamState


class ModelLoadingTestProcessor(FrameProcessor):
    """Simple processor that tracks load invocations."""

    def __init__(self, load_delay: float = 0.1, should_fail: bool = False):
        super().__init__()
        self.load_delay = load_delay
        self.should_fail = should_fail
        self.load_call_count = 0

    async def load_model(self, **kwargs):
        self.load_call_count += 1
        if self.should_fail:
            raise RuntimeError("Simulated model loading failure")
        await asyncio.sleep(self.load_delay)

    async def process_video_async(self, frame):
        return frame

    async def process_audio_async(self, frame):
        return [frame]

    async def update_params(self, params):
        return None


@pytest.mark.asyncio
async def test_ensure_model_loaded_transitions_and_idempotent():
    processor = ModelLoadingTestProcessor(load_delay=0.05)
    state = StreamState()
    processor.attach_state(state)

    assert state.get_pipeline_state()["status"] == "LOADING"

    await asyncio.gather(*(processor.ensure_model_loaded() for _ in range(3)))

    assert processor.load_call_count == 1
    assert processor._model_loaded
    assert state.get_pipeline_state()["status"] == "IDLE"


@pytest.mark.asyncio
async def test_ensure_model_loaded_handles_failures():
    processor = ModelLoadingTestProcessor(should_fail=True)
    state = StreamState()
    processor.attach_state(state)

    with pytest.raises(RuntimeError, match="Simulated model loading failure"):
        await processor.ensure_model_loaded()

    assert state.get_pipeline_state()["status"] == "LOADING"
    assert not processor._model_loaded
    assert processor.load_call_count == 1


@pytest.mark.asyncio
async def test_ensure_model_loaded_without_state():
    processor = ModelLoadingTestProcessor(load_delay=0.01)
    await processor.ensure_model_loaded()
    assert processor._model_loaded
    assert processor.load_call_count == 1


@pytest.mark.asyncio
async def test_sentinel_triggers_model_load_and_filters_params():
    received_params = []
    load_count = 0

    async def model_loader(**_):
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.01)

    async def param_updater(params):
        received_params.append(params)

    processor = StreamProcessor(
        model_loader=model_loader,
        param_updater=param_updater,
        name="sentinel-filter",
        port=8100,
    )

    await processor._frame_processor.update_params({"load_model": True, "intensity": 0.7})
    await processor._frame_processor.update_params({"intensity": 0.9})

    assert load_count == 1
    assert processor._frame_processor._model_loaded
    assert received_params == [{"intensity": 0.7}, {"intensity": 0.9}]


@pytest.mark.asyncio
async def test_sentinel_errors_surface():
    async def failing_loader(**_):
        raise RuntimeError("boom")

    processor = StreamProcessor(
        model_loader=failing_loader,
        name="sentinel-error",
        port=8101,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await processor._frame_processor.update_params({"load_model": True})

    assert processor._frame_processor.state.get_pipeline_state()["status"] == "LOADING"
    assert not processor._frame_processor._model_loaded


@pytest.mark.asyncio
async def test_concurrent_sentinel_requests_only_load_once():
    load_count = 0

    async def loader(**_):
        nonlocal load_count
        load_count += 1
        await asyncio.sleep(0.02)

    processor = StreamProcessor(model_loader=loader, name="sentinel-concurrency", port=8102)

    await asyncio.gather(
        *(processor._frame_processor.update_params({"load_model": True}) for _ in range(5))
    )

    assert load_count == 1
    assert processor._frame_processor._model_loaded


@pytest.mark.asyncio
async def test_stream_processor_automatic_trigger_and_health_check():
    model_loaded = False

    async def loader(**_):
        nonlocal model_loaded
        await asyncio.sleep(0.2)
        model_loaded = True

    processor = StreamProcessor(model_loader=loader, name="auto-load", port=8103)

    server_task = asyncio.create_task(processor.run_forever())
    try:
        import aiohttp

        async def fetch_status():
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8103/health") as resp:
                    return resp.status, await resp.json()

        await asyncio.sleep(0.1)
        status, payload = await fetch_status()
        assert status == 200 and payload == {"status": "LOADING"}

        await asyncio.sleep(0.4)
        status, payload = await fetch_status()
        assert status == 200 and payload == {"status": "IDLE"}
        assert model_loaded
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
