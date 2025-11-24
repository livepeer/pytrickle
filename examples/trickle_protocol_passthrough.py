"""
Direct TrickleProtocol passthrough example.

This mirrors the low-level http-trickle samples by bypassing StreamProcessor and
talking to trickle channels directly. It subscribes to one channel, converts the
ingress frames into ``VideoOutput``/``AudioOutput`` objects, and republishes the
frames to another channel.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import AsyncGenerator, Optional

from pytrickle.frames import AudioFrame, AudioOutput, OutputFrame, VideoFrame, VideoOutput
from pytrickle.protocol import TrickleProtocol

logger = logging.getLogger(__name__)


def frame_to_output(frame: VideoFrame | AudioFrame, request_id: str) -> Optional[OutputFrame]:
    """Convert an ingress frame into the appropriate OutputFrame."""
    if isinstance(frame, VideoFrame):
        return VideoOutput(frame, request_id)
    if isinstance(frame, AudioFrame):
        return AudioOutput([frame], request_id)
    return None


async def passthrough_stream(
    subscribe_url: str,
    publish_url: str,
    *,
    request_id: str = "protocol-passthrough",
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_framerate: Optional[int] = None,
):
    """Bridge frames from ``subscribe_url`` into ``publish_url`` using TrickleProtocol."""
    protocol = TrickleProtocol(
        subscribe_url=subscribe_url,
        publish_url=publish_url,
        width=width,
        height=height,
        max_framerate=max_framerate,
    )
    stop_event = asyncio.Event()

    async def output_generator() -> AsyncGenerator[OutputFrame, None]:
        async for frame in protocol.ingress_loop(stop_event):
            output = frame_to_output(frame, request_id)
            if output:
                yield output

    await protocol.start()
    try:
        await protocol.egress_loop(output_generator())
    finally:
        stop_event.set()
        await protocol.stop()


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m examples.trickle_protocol_passthrough``."""
    parser = argparse.ArgumentParser(description="Direct TrickleProtocol passthrough utility.")
    parser.add_argument("--subscribe", required=True, help="Full trickle subscribe URL (e.g. http://127.0.0.1:3389/input)")
    parser.add_argument("--publish", required=True, help="Full trickle publish URL (e.g. http://127.0.0.1:3389/output)")
    parser.add_argument("--request-id", default="protocol-passthrough", help="Request ID stamped on VideoOutput/AudioOutput.")
    parser.add_argument("--width", type=int, default=None, help="Optional ingress width hint.")
    parser.add_argument("--height", type=int, default=None, help="Optional ingress height hint.")
    parser.add_argument("--max-framerate", type=int, default=None, help="Optional decoder framerate cap.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    asyncio.run(
        passthrough_stream(
            args.subscribe,
            args.publish,
            request_id=args.request_id,
            width=args.width,
            height=args.height,
            max_framerate=args.max_framerate,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

