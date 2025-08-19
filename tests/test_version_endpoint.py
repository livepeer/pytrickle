import pytest
from aiohttp.test_utils import TestServer, TestClient

from pytrickle.stream_processor import _InternalFrameProcessor
from pytrickle.server import StreamServer

# Import example processor functions
from examples.process_video_example import load_model, process_video, update_params


@pytest.mark.asyncio
async def test_version_endpoint_returns_expected_payload():
    # Build an internal frame processor from the example functions
    processor = _InternalFrameProcessor(
        video_processor=process_video,
        audio_processor=None,
        model_loader=load_model,
        param_updater=update_params,
        name="trickle-stream-example",
    )

    # Create the app server; pass desired model_id via capability_name
    server = StreamServer(
        frame_processor=processor,
        port=0,  # use ephemeral port for safety (not used with TestServer)
        capability_name="trickle-stream-example",
        enable_default_routes=True,
    )

    app = server.get_app()

    test_server = TestServer(app)
    async with test_server:
        client = TestClient(test_server)
        async with client:
            resp = await client.get("/version")
            assert resp.status == 200
            data = await resp.json()
            assert data == {
                "pipeline": "byoc",
                "model_id": "trickle-stream-example",
                "version": "0.0.1",
            }
