"""
Utilities for publishing and subscribing trickle streams with FFmpeg/FFplay.

This module mirrors the workflows documented in the http-trickle project by
providing:
- Helper functions that build the exact shell commands for FFmpeg publishers
  and FFplay subscribers.
- A CLI (``python -m examples.trickle_media_tools``) that streams
  MPEG-TS data from ``stdin`` to a trickle channel, or reads a trickle stream
  and writes the payload to ``stdout`` so it can be piped into FFplay.

Typical usage::

    # Publish a video file into the "input" channel
    ffmpeg -re -i demo.mp4 -c copy -f mpegts - \\
        | python -m examples.trickle_media_tools \\
            publisher --url http://127.0.0.1:3389 --stream input

    # Subscribe to the processed "output" channel and view with ffplay
    python -m examples.trickle_media_tools \\
        subscriber --url http://127.0.0.1:3389 --stream output \\
        | ffplay -probesize 32 -fflags nobuffer -flags low_delay -
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shlex
import sys
from typing import Iterable, List, Sequence, Tuple

from pytrickle.publisher import TricklePublisher
from pytrickle.subscriber import TrickleSubscriber

logger = logging.getLogger(__name__)

DEFAULT_TRICKLE_URL = os.environ.get("TRICKLE_URL", "http://127.0.0.1:3389")
DEFAULT_PUBLISH_STREAM = "input"
DEFAULT_SUBSCRIBE_STREAM = "output"
DEFAULT_CHUNK_SIZE = 256 * 1024  # 256 KiB chunks balance latency vs throughput
DEFAULT_MIME_TYPE = "video/mp2t"


def build_stream_url(base_url: str, stream_name: str) -> str:
    """Return a normalized trickle channel URL."""
    if not base_url:
        raise ValueError("base_url is required")
    if not stream_name:
        raise ValueError("stream_name is required")
    return f"{base_url.rstrip('/')}/{stream_name.strip('/')}"


def build_ffmpeg_publish_command(
    input_path: str,
    *,
    base_url: str = DEFAULT_TRICKLE_URL,
    stream_name: str = DEFAULT_PUBLISH_STREAM,
    loglevel: str = "warning",
    loop_input: bool = False,
    extra_args: Sequence[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Build the FFmpeg + PyTrickle pipeline used to publish sample files.

    Returns:
        Tuple containing the FFmpeg command list and the Python publisher command.
    """
    if not input_path:
        raise ValueError("input_path is required")

    ffmpeg_cmd: List[str] = [
        "ffmpeg",
        "-loglevel",
        loglevel,
        "-re",
    ]
    if loop_input:
        ffmpeg_cmd.extend(["-stream_loop", "-1"])
    ffmpeg_cmd.extend(["-i", input_path, "-c", "copy", "-f", "mpegts", "-"])
    if extra_args:
        ffmpeg_cmd[1:1] = list(extra_args)

    python_cmd = [
        sys.executable,
        "-m",
        "examples.trickle_media_tools",
        "publisher",
        "--url",
        base_url,
        "--stream",
        stream_name,
    ]

    return ffmpeg_cmd, python_cmd


def build_ffplay_subscribe_command(
    *,
    base_url: str = DEFAULT_TRICKLE_URL,
    stream_name: str = DEFAULT_SUBSCRIBE_STREAM,
    start_seq: int = -1,
) -> Tuple[List[str], List[str]]:
    """
    Build the PyTrickle subscriber + FFplay pipeline for quick playback tests.

    Returns:
        Tuple of (PyTrickle subscriber command, ffplay command).
    """
    python_cmd = [
        sys.executable,
        "-m",
        "examples.trickle_media_tools",
        "subscriber",
        "--url",
        base_url,
        "--stream",
        stream_name,
        "--start-seq",
        str(start_seq),
    ]
    ffplay_cmd = [
        "ffplay",
        "-probesize",
        "32",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-",
    ]
    return python_cmd, ffplay_cmd


async def _stdin_chunks(chunk_size: int) -> asyncio.AsyncIterator[bytes]:
    """Yield binary chunks from stdin using a background thread."""
    loop = asyncio.get_running_loop()
    reader = sys.stdin.buffer.read

    while True:
        chunk = await loop.run_in_executor(None, reader, chunk_size)
        if not chunk:
            break
        yield chunk


async def publish_from_stdin(
    *,
    base_url: str,
    stream_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mime_type: str = DEFAULT_MIME_TYPE,
):
    """Stream binary data from stdin directly into a trickle channel."""
    url = build_stream_url(base_url, stream_name)
    publisher = TricklePublisher(url=url, mime_type=mime_type)
    await publisher.start()

    try:
        async with await publisher.next() as segment:
            async for chunk in _stdin_chunks(chunk_size):
                await segment.write(chunk)
    except (BrokenPipeError, ConnectionError) as exc:
        logger.warning("Publisher terminated early: %s", exc)
    finally:
        await publisher.close()


async def stream_to_stdout(
    *,
    base_url: str,
    stream_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    start_seq: int = -1,
):
    """Subscribe to a trickle channel and write the payload to stdout."""
    url = build_stream_url(base_url, stream_name)
    subscriber = TrickleSubscriber(url=url, start_seq=start_seq)
    writer = sys.stdout.buffer
    loop = asyncio.get_running_loop()

    async with subscriber as trickle_sub:
        try:
            while True:
                segment = await trickle_sub.next()
                if not segment:
                    break
                while True:
                    chunk = await segment.read(chunk_size)
                    if not chunk:
                        break
                    await loop.run_in_executor(None, writer.write, chunk)
                    await loop.run_in_executor(None, writer.flush)
        except BrokenPipeError as exc:
            logger.warning("Subscriber output closed: %s", exc)


def _format_command(cmd: Sequence[str]) -> str:
    """Return a human-friendly shell command string."""
    return " ".join(shlex.quote(part) for part in cmd)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser so tests can validate the interface."""
    parser = argparse.ArgumentParser(
        description="Publish or subscribe trickle streams with FFmpeg/FFplay helpers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pub = sub.add_parser("publisher", help="Read MPEG-TS from stdin and publish to trickle.")
    pub.add_argument("--url", default=DEFAULT_TRICKLE_URL, help="Base trickle URL (default: %(default)s)")
    pub.add_argument("--stream", default=DEFAULT_PUBLISH_STREAM, help="Channel name to publish into.")
    pub.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size (bytes) to buffer before pushing upstream.",
    )
    pub.add_argument("--mime-type", default=DEFAULT_MIME_TYPE, help="Content-Type used when publishing.")

    subp = sub.add_parser("subscriber", help="Read a trickle channel and write MPEG-TS to stdout.")
    subp.add_argument("--url", default=DEFAULT_TRICKLE_URL, help="Base trickle URL (default: %(default)s)")
    subp.add_argument("--stream", default=DEFAULT_SUBSCRIBE_STREAM, help="Channel name to subscribe to.")
    subp.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size (bytes) to read from trickle segments.",
    )
    subp.add_argument(
        "--start-seq",
        type=int,
        default=-1,
        help="Initial sequence number (default: wait for next segment).",
    )

    docs = sub.add_parser("commands", help="Print ready-to-run FFmpeg/FFplay command pipelines.")
    docs.add_argument("input", help="Sample file passed to FFmpeg.")
    docs.add_argument("--url", default=DEFAULT_TRICKLE_URL, help="Base trickle URL (default: %(default)s)")
    docs.add_argument("--ingest-stream", default=DEFAULT_PUBLISH_STREAM, help="Channel FFmpeg publishes into.")
    docs.add_argument("--output-stream", default=DEFAULT_SUBSCRIBE_STREAM, help="Channel to read from.")

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m examples.trickle_media_tools``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "publisher":
        asyncio.run(
            publish_from_stdin(
                base_url=args.url,
                stream_name=args.stream,
                chunk_size=args.chunk_size,
                mime_type=args.mime_type,
            )
        )
        return 0

    if args.command == "subscriber":
        asyncio.run(
            stream_to_stdout(
                base_url=args.url,
                stream_name=args.stream,
                chunk_size=args.chunk_size,
                start_seq=args.start_seq,
            )
        )
        return 0

    if args.command == "commands":
        ffmpeg_cmd, publisher_cmd = build_ffmpeg_publish_command(
            args.input,
            base_url=args.url,
            stream_name=args.ingest_stream,
        )
        subscriber_cmd, ffplay_cmd = build_ffplay_subscribe_command(
            base_url=args.url,
            stream_name=args.output_stream,
        )
        print("FFmpeg publisher:")
        print("  " + _format_command(ffmpeg_cmd) + " | " + _format_command(publisher_cmd))
        print("\nFFplay subscriber:")
        print("  " + _format_command(subscriber_cmd) + " | " + _format_command(ffplay_cmd))
        return 0

    parser.error(f"Unknown command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())

