import sys

import torch

from examples.trickle_media_tools import (
    build_ffmpeg_publish_command,
    build_ffplay_subscribe_command,
    build_stream_url,
)
from examples.trickle_protocol_passthrough import frame_to_output
from pytrickle.frames import AudioFrame, VideoFrame, AudioOutput, VideoOutput


def test_build_stream_url_normalizes_components():
    url = build_stream_url("http://localhost:3389/", "/input")
    assert url == "http://localhost:3389/input"


def test_build_ffmpeg_publish_command_includes_python_invocation(tmp_path):
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"fake")

    ffmpeg_cmd, python_cmd = build_ffmpeg_publish_command(
        str(sample),
        base_url="http://127.0.0.1:3389/",
        stream_name="input",
        loop_input=True,
    )

    assert ffmpeg_cmd[:5] == ["ffmpeg", "-loglevel", "warning", "-re", "-stream_loop"]
    assert python_cmd[:3] == [sys.executable, "-m", "examples.trickle_media_tools"]


def test_build_ffplay_subscribe_command_returns_expected_pipeline():
    subscriber_cmd, ffplay_cmd = build_ffplay_subscribe_command(
        base_url="http://127.0.0.1:3389",
        stream_name="output",
        start_seq=-2,
    )

    assert subscriber_cmd[-2:] == ["--start-seq", "-2"]
    assert ffplay_cmd[-1] == "-"


def test_frame_to_output_video_and_audio_conversion():
    video = VideoFrame.from_tensor(torch.zeros((4, 4, 3), dtype=torch.float32))
    audio = AudioFrame.from_tensor(torch.zeros((1, 1024), dtype=torch.float32))

    video_output = frame_to_output(video, "req-1")
    audio_output = frame_to_output(audio, "req-2")

    assert isinstance(video_output, VideoOutput)
    assert video_output.request_id == "req-1"
    assert isinstance(audio_output, AudioOutput)
    assert audio_output.request_id == "req-2"

