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
import math
from typing import Optional, Callable
from fractions import Fraction
from collections import deque
from PIL import Image

from .frames import VideoOutput, AudioOutput, InputFrame, DEFAULT_WIDTH, DEFAULT_HEIGHT

logger = logging.getLogger(__name__)

# use mpegts default time base
OUT_TIME_BASE = Fraction(1, 90_000)
GOP_SECS = 2

# maximum buffer duration in seconds
AUDIO_BUFFER_SECS = 2
# maximum buffer size in number of packets. Roughly 2s at 20ms / frame
AUDIO_BUFFER_LEN = 100
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
    audio_received = False
    first_video_ts = None
    dropped_video = 0
    dropped_audio = 0
    audio_buffer = deque()

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

            # if there is pending audio, check whether video is too far behind
            if len(audio_buffer) > 0:
                audio_ts = audio_buffer[0].timestamp * audio_buffer[0].time_base
                delta = audio_ts - current
                if delta > AUDIO_BUFFER_SECS:
                    # video is too far behind audio, drop the frame to try and catch up
                    dropped_video += 1
                    if dropped_video > MAX_DROPPED_VIDEO:
                        # dropped too many frames, may be unlikely to catch up so stop the stream
                        logger.error(f"A/V is out of sync, exiting from video audio_ts={float(audio_ts)} video_ts={float(current)} delta={float(delta)}")
                        break
                    # drop the frame by skipping the rest of the following code
                    continue

            if not last_kf or float(current - last_kf) >= GOP_SECS:
                frame.pict_type = av.video.frame.PictureType.I
                last_kf = current
                if first_video_ts is None:
                    first_video_ts = current
            encoded_packets = output_video_stream.encode(frame)
            for ep in encoded_packets:
                output_container.mux(ep)
            logger.debug(f"encoded video packets={len(encoded_packets)} pts={frame.pts} time_base={frame.time_base} ts={float(current)}")
            continue

        if isinstance(avframe, AudioOutput):
            if not output_audio_stream:
                # received audio but no audio output, so drop
                continue
            if output_video_stream and first_video_ts is None:
                # Buffer up audio until we receive video, up to a point
                # because video could be extremely delayed and we don't
                # want to send out audio-only segments since that confuses
                # downstream tools

                # shortcut to assume len(audio_buffer) is at least 1
                if len(avframe.frames) <= 0:
                    continue
                for af in avframe.frames:
                    audio_buffer.append(af)
                while len(audio_buffer) > AUDIO_BUFFER_LEN:
                    af = audio_buffer.popleft()
                    dropped_audio += 1
                continue

            while len(audio_buffer) > 0:
                # if we hit this point then we have a video frame
                # Check whether audio is too far behind first video packet.
                af = audio_buffer[0]
                first_audio_ts = af.timestamp * af.time_base
                if first_video_ts - first_audio_ts > AUDIO_BUFFER_SECS:
                    audio_buffer.popleft()
                    dropped_audio += 1
                    continue

                # NB: video being too far behind audio is handled within the video code
                #     so we can drop video frames if necessary

                break

            if len(audio_buffer) > 0:
                first_ts = float(audio_buffer[0].timestamp * audio_buffer[0].time_base)
                last_ts = float(audio_buffer[-1].timestamp * audio_buffer[-1].time_base)
                avframe.frames[:0] = audio_buffer
                logger.info(f"Flushing {len(audio_buffer)} audio frames from={first_ts} to={last_ts} dropped_audio={dropped_audio} dropped_video={dropped_video}")
                audio_buffer.clear()

            av_broken = False
            for af in avframe.frames:
                af.log_timestamps["frame_end"] = time.time()
                _log_frame_timestamps("Audio", af)
                if not audio_received and first_video_ts is not None:
                    first_audio_ts = af.timestamp * af.time_base
                    delta = first_audio_ts - first_video_ts
                    if abs(delta) > AUDIO_BUFFER_SECS:
                        # A/V is out of sync badly enough so exit for now
                        av_broken = True
                        logger.error(f"A/V is out of sync, exiting from audio audio_ts={float(first_audio_ts)} video_ts={float(first_video_ts)} delta={float(delta)}")
                        break
                    logger.info(f"Received first audio_ts={float(first_audio_ts)} video_ts={float(first_video_ts)} delta={float(delta)}")
                    audio_received = True
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
            if av_broken:
                # too far out of sync, so stop encoding
                break
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

def default_output_metadata(height: int, width: int):
    """Generate default metadata for output streams."""
    return {
        'video': {
            'codec': 'h264',
            'width': width,
            'height': height,
            'pix_fmt': 'yuv420p',
            'time_base': OUT_TIME_BASE,
            'framerate': Fraction(24, 1),
            'sar': calc_aspect_ratio(height, width),
            'dar': calc_aspect_ratio(height, width),
            'format': 'yuv420p',
            'target_width': width,
            'target_height': height,
        },
        'audio': {
            'codec': 'aac',
            'sample_rate': 48000,
            'format': 'fltp',
            'channels': 2,
            'layout': 'stereo',
            'time_base': OUT_TIME_BASE,
        }
    }

def calc_aspect_ratio(height: int, width: int) -> Fraction:
    """Calculate aspect ratio as a Fraction."""
    if height <= 0 or width <= 0:
        return Fraction(1, 1)
    
    # Calculate the greatest common divisor (GCD) of width and height
    common_divisor = math.gcd(width, height)

    # Divide width and height by their GCD to get the simplified ratio
    simplified_width = width // common_divisor
    simplified_height = height // common_divisor

    # Create and return a Fraction object
    return Fraction(simplified_width, simplified_height)