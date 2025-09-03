"""
Video encoder for trickle streaming.

Handles encoding of processed video frames back to video streams using PyAV,
with support for multiple codecs and audio/video synchronization.
"""

import av
import time
import datetime
import logging
import os
from typing import Optional, Callable
from fractions import Fraction

from PIL import Image

from .frames import VideoOutput, AudioOutput, InputFrame, DEFAULT_WIDTH, DEFAULT_HEIGHT

logger = logging.getLogger(__name__)

# use mpegts default time base
OUT_TIME_BASE = Fraction(1, 90_000)
GOP_SECS = 2

# how many out-of-sync video packets to drop. Roughly 4s at 25fps
MAX_DROPPED_VIDEO = 100

def encode_av(
    input_queue: Callable,
    output_callback: Callable,
    get_metadata: Callable,
    video_codec: Optional[str] = 'libx264',
    audio_codec: Optional[str] = 'libfdk_aac'
):
    """
    Encode video/audio frames to output stream.
    
    Args:
        input_queue: Function to get the next frame from the input queue
        output_callback: Callback function to handle output stream data
        get_metadata: Function to get stream metadata
        video_codec: Video codec to use for encoding
        audio_codec: Audio codec to use for encoding
    """
    logger.info("Starting encoder")

    # Wait for metadata to become available (with timeout)
    decoded_metadata = None
    max_wait_time = 30  # seconds
    wait_interval = 0.1  # seconds
    total_waited = 0
    
    while total_waited < max_wait_time:
        decoded_metadata = get_metadata()
        if decoded_metadata:
            break
        time.sleep(wait_interval)
        total_waited += wait_interval
    
    if not decoded_metadata:
        logger.error(f"No metadata received after waiting {max_wait_time} seconds, exiting encoder")
        return

    # Ensure metadata has the expected structure
    if not isinstance(decoded_metadata, dict):
        logger.error(f"Invalid metadata format: expected dict, got {type(decoded_metadata)}")
        return
    
    if 'video' not in decoded_metadata and 'audio' not in decoded_metadata:
        logger.error("Metadata missing both video and audio streams")
        return

    video_meta = decoded_metadata.get('video')
    audio_meta = decoded_metadata.get('audio')
    
    logger.info(f"Encoder received metadata after waiting {total_waited:.1f}s: video={video_meta is not None} audio={audio_meta is not None}")

    def custom_io_open(url: str, flags: int, options: dict):
        read_fd, write_fd = os.pipe()
        read_file = os.fdopen(read_fd, 'rb', buffering=0)
        write_file = os.fdopen(write_fd, 'wb', buffering=0)
        output_callback(read_file, write_file, url)
        return write_file

    # Open the output container in write mode
    output_container = av.open("%d.ts", format='segment', mode='w', io_open=custom_io_open)

    # Create corresponding output streams if input streams exist
    output_video_stream = None
    output_audio_stream = None

    if video_meta and video_codec:
        # Add a new stream to the output using the desired video codec
        target_width = video_meta.get('target_width', DEFAULT_WIDTH)
        target_height = video_meta.get('target_height', DEFAULT_HEIGHT)
        video_opts = {'video_size': f'{target_width}x{target_height}', 'bf': '0'}
        if video_codec == 'libx264':
            video_opts = video_opts | {'preset': 'superfast', 'tune': 'zerolatency', 'forced-idr': '1'}
        output_video_stream = output_container.add_stream(video_codec, options=video_opts)
        output_video_stream.time_base = OUT_TIME_BASE
        
        # Ensure the codec context also has the time base
        if output_video_stream.codec_context.time_base is None:
            output_video_stream.codec_context.time_base = OUT_TIME_BASE

    if audio_meta and audio_codec:
        # Add a new stream to the output using the desired audio codec
        output_audio_stream = output_container.add_stream(audio_codec)
        output_audio_stream.time_base = OUT_TIME_BASE
        output_audio_stream.sample_rate = 48000
        output_audio_stream.layout = 'mono'
        
        # Ensure the codec context also has the time base
        if output_audio_stream.codec_context.time_base is None:
            output_audio_stream.codec_context.time_base = OUT_TIME_BASE
            
        # Optional: set other encoding parameters, e.g.:
        # output_audio_stream.bit_rate = 128_000

    # Now read packets from the input, decode, then re-encode, and mux.
    start = datetime.datetime.now()
    last_kf = None
    first_video_ts = None
    dropped_video = 0

    while True:
        avframe = input_queue()
        if avframe is None:
            break

        if isinstance(avframe, VideoOutput):
            if not output_video_stream:
                # received video but no video output, so drop
                continue
            avframe.log_timestamps["frame_end"] = time.time()
            _log_frame_timestamps("Video", avframe.frame)

            tensor = avframe.tensor.squeeze(0)
            image_np = (tensor * 255).byte().cpu().numpy()
            # Explicitly specify RGB mode - decoder produces RGB, encoder expects RGB
            image = Image.fromarray(image_np, mode='RGB')

            frame = av.video.frame.VideoFrame.from_image(image)
            
            # Ensure the codec context has a time base
            if output_video_stream.codec_context.time_base is None:
                logger.warning("Video codec context time_base is None, using stream time_base")
                dest_time_base = output_video_stream.time_base
            else:
                dest_time_base = output_video_stream.codec_context.time_base
                
            frame.pts = _rescale_ts(avframe.timestamp, avframe.time_base, dest_time_base)
            frame.time_base = dest_time_base
            current = avframe.timestamp * avframe.time_base

            # No audio synchronization delays - video and audio encoded independently

            if not last_kf or float(current - last_kf) >= GOP_SECS:
                frame.pict_type = av.video.frame.PictureType.I
                last_kf = current
                if first_video_ts is None:
                    first_video_ts = current
            encoded_packets = output_video_stream.encode(frame)
            for ep in encoded_packets:
                output_container.mux(ep)
            continue

        if isinstance(avframe, AudioOutput):
            if not output_audio_stream:
                # received audio but no audio output, so drop
                continue
                
            # Process audio frames immediately without buffering or video sync delays
            # This prevents audio quality issues during video frame skipping
            av_broken = False
            for af in avframe.frames:
                af.log_timestamps["frame_end"] = time.time()
                _log_frame_timestamps("Audio", af)
                
                # Encode audio immediately without sync delays or buffering
                frame = av.audio.frame.AudioFrame.from_ndarray(af.samples, format=af.format, layout=af.layout)
                frame.sample_rate = af.rate
                
                # Ensure the codec context has a time base
                if output_audio_stream.codec_context.time_base is None:
                    logger.warning("Audio codec context time_base is None, using stream time_base")
                    dest_time_base = output_audio_stream.time_base
                else:
                    dest_time_base = output_audio_stream.codec_context.time_base
                    
                frame.pts = _rescale_ts(af.timestamp, af.time_base, dest_time_base)
                frame.time_base = dest_time_base
                encoded_packets = output_audio_stream.encode(frame)
                for ep in encoded_packets:
                    output_container.mux(ep)
            continue

        logger.warning(f"Unsupported output frame type {type(avframe)}")

    # After reading all packets, flush encoders
    logger.info("Stopping encoder")
    if output_video_stream:
        encoded_packets = output_video_stream.encode(None)
        for ep in encoded_packets:
            output_container.mux(ep)

    if output_audio_stream:
        encoded_packets = output_audio_stream.encode(None)
        for ep in encoded_packets:
            output_container.mux(ep)

    # Close the output container to finish writing
    output_container.close()

def _rescale_ts(pts: int, orig_tb: Fraction, dest_tb: Fraction):
    """Rescale timestamp from one time base to another."""
    if orig_tb == dest_tb:
        return pts
    if dest_tb is None:
        logger.error(f"Destination time base is None, cannot rescale timestamp {pts}")
        return pts  # Return original timestamp as fallback
    return int(round(float((Fraction(pts) * orig_tb) / dest_tb)))

def _log_frame_timestamps(frame_type: str, frame: InputFrame):
    """Log frame processing timestamps for performance monitoring."""
    ts = frame.log_timestamps

    def log_duration(start_key: str, end_key: str):
        if start_key in ts and end_key in ts:
            duration = ts[end_key] - ts[start_key]
            logger.debug(f"frame_type={frame_type} start_tag={start_key} end_tag={end_key} duration_s={duration}s")

    log_duration('frame_init', 'pre_process_frame')
    log_duration('pre_process_frame', 'post_process_frame')
    log_duration('post_process_frame', 'frame_end')
    log_duration('frame_init', 'frame_end') 