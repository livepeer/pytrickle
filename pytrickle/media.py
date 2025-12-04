"""
Media processing for trickle streaming.

Handles video encoding/decoding, frame conversion, and media stream processing
for trickle-based video streaming applications.
"""

import os
import time
import asyncio
import logging
import threading
from typing import Callable, Optional, Union

from .subscriber import TrickleSubscriber  
from .publisher import TricklePublisher
from .decoder import decode_av, DEFAULT_MAX_FRAMERATE
from .encoder import encode_av
from .frames import DEFAULT_WIDTH, DEFAULT_HEIGHT

logger = logging.getLogger(__name__)

MAX_DECODER_RETRIES = 3
DECODER_RETRY_RESET_SECONDS = 120  # reset retry counter after 2 minutes

MAX_ENCODER_RETRIES = 3
ENCODER_RETRY_RESET_SECONDS = 120  # reset retry counter after 2 minutes

async def run_subscribe(
    subscribe_url: str, 
    frame_callback: Callable,
    put_metadata: Callable,
    monitoring_callback: Optional[Callable] = None,
    target_width: Optional[int] = DEFAULT_WIDTH,
    target_height: Optional[int] = DEFAULT_HEIGHT,
    max_framerate: Union[int, Callable[[], int], None] = DEFAULT_MAX_FRAMERATE,
    subscriber_timeout: Optional[float] = None,
):
    """
    Run subscription loop to receive and decode video streams.
    
    Args:
        subscribe_url: URL to subscribe to for incoming video stream
        frame_callback: Callback function to process decoded frames
        put_metadata: Callback to store stream metadata
        monitoring_callback: Optional callback for monitoring events
        target_width: Target width for decoded frames
        target_height: Target height for decoded frames
        max_framerate: Maximum framerate or callable returning framerate for decoded frames
        subscriber_timeout: Optional timeout for subscriber connection
    """
    # Ensure default values are applied if None
    if target_width is None:
        target_width = DEFAULT_WIDTH
    if target_height is None:
        target_height = DEFAULT_HEIGHT
    if max_framerate is None:
        max_framerate = DEFAULT_MAX_FRAMERATE
        
    try:
        in_pipe, out_pipe = os.pipe()
        write_fd = await _asyncify_fd_writer(out_pipe)
        parse_task = asyncio.create_task(
            _decode_in(in_pipe, frame_callback, put_metadata, write_fd, target_width, target_height, max_framerate)
        )
        subscribe_task = asyncio.create_task(
            _subscribe(subscribe_url, write_fd, monitoring_callback, subscriber_timeout)
        )
        await asyncio.gather(subscribe_task, parse_task)
        logger.info("run_subscribe complete")
    except Exception as e:
        logger.exception("run_subscribe got error", stack_info=True)
    finally:
        put_metadata(None)  # in case decoder quit without writing anything
        frame_callback(None)  # stops inference if this function exits early

async def _subscribe(
    subscribe_url: str,
    out_pipe,
    monitoring_callback: Optional[Callable] = None,
    subscriber_timeout: Optional[float] = None,
):
    """Subscribe to trickle stream and write data to pipe."""
    first_segment = True

    if subscriber_timeout is not None:
        subscriber_ctx = TrickleSubscriber(url=subscribe_url, connect_timeout_seconds=subscriber_timeout)
    else:
        subscriber_ctx = TrickleSubscriber(url=subscribe_url)
    async with subscriber_ctx as subscriber:
        logger.info(f"Launching subscribe loop for {subscribe_url}")
        while True:
            segment = None
            try:
                segment = await subscriber.next()
                if not segment:
                    break  # complete
                while True:
                    chunk = await segment.read()
                    if not chunk:
                        break  # end of segment
                    out_pipe.write(chunk)
                    await out_pipe.drain()
                if first_segment:
                    first_segment = False
                    if monitoring_callback:
                        await monitoring_callback({
                            "type": "receive_first_ingest_segment",
                            "timestamp": int(time.time() * 1000)
                        })
            except Exception as e:
                logger.info(f"Failed to read segment - {e}")
                break  # end of stream?
            finally:
                if segment:
                    await segment.close()
                else:
                    # stream is complete
                    out_pipe.close()
                    break

async def _asyncify_fd_writer(write_fd: int):
    """Convert file descriptor to asyncio StreamWriter."""
    loop = asyncio.get_event_loop()
    write_protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader(), loop=loop)
    write_transport, _ = await loop.connect_write_pipe(
        lambda: write_protocol, os.fdopen(write_fd, 'wb')
    )
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
    return writer

async def _decode_in(
    in_pipe: int, 
    frame_callback: Callable, 
    put_metadata: Callable, 
    write_fd, 
    target_width: int, 
    target_height: int,
    max_framerate: Union[int, Callable[[], int], None] = DEFAULT_MAX_FRAMERATE,
):
    """Decode video stream from pipe."""
    # Ensure default values are applied if None
    if target_width is None:
        target_width = DEFAULT_WIDTH
    if target_height is None:
        target_height = DEFAULT_HEIGHT
    if max_framerate is None:
        max_framerate = DEFAULT_MAX_FRAMERATE
        
    def decode_runner():
        retry_count = 0
        last_retry_time = time.time()
        while retry_count < MAX_DECODER_RETRIES:
            try:
                decode_av(f"pipe:{in_pipe}", frame_callback, put_metadata, target_width, target_height, max_framerate)
                break  # clean exit
            except Exception as e:
                msg = str(e)
                if f"Invalid data found when processing input: 'pipe:{in_pipe}'" in msg:
                    logger.info("Stream closed before initialization")
                    break
                current_time = time.time()
                # Reset retry counter if enough time has elapsed
                if current_time - last_retry_time > DECODER_RETRY_RESET_SECONDS:
                    logger.info("Resetting decoder retry count")
                    retry_count = 0
                retry_count += 1
                last_retry_time = current_time
                if retry_count < MAX_DECODER_RETRIES:
                    logger.exception(f"Error in decode_av, retrying {retry_count}/{MAX_DECODER_RETRIES}")
                else:
                    logger.exception("Error in decode_av, maximum retries reached")

        try:
            # force write end of pipe to close to terminate trickle subscriber
            write_fd.close()
        except Exception:
            # happens sometimes but ignore
            pass

        os.close(in_pipe)
        logger.info("Decoding finished")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, decode_runner)

async def run_publish(
    publish_url: str, 
    frame_generator: Callable,
    get_metadata: Callable,
    monitoring_callback: Optional[Callable] = None,
    publisher_timeout: Optional[float] = None,
    detect_out_resolution: bool = True,
):
    """
    Run publishing loop to encode and publish video streams.
    
    Args:
        publish_url: URL to publish the processed video stream  
        frame_generator: Generator function that yields output frames
        get_metadata: Function to get stream metadata
        monitoring_callback: Optional callback for monitoring events
        publisher_timeout: Optional timeout for publisher connection
        detect_out_resolution: If True, detect output resolution from first frame's tensor shape.
                                        If False, use target_width/target_height from decoder metadata.
                                        Default is True to support Super Resolution workflows.
    """
    first_segment = True
    publisher = None
    
    try:
        publisher = TricklePublisher(url=publish_url, mime_type="video/mp2t")
        if publisher_timeout is not None:
            publisher.connect_timeout_seconds = publisher_timeout

        loop = asyncio.get_running_loop()
        
        async def callback(pipe_file, pipe_name):
            nonlocal first_segment
            # trickle publish a segment with the contents of `pipe_file`
            async with await publisher.next() as segment:
                # convert pipe_fd into an asyncio friendly StreamReader
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe_file)
                while True:
                    sz = 32 * 1024  # read in chunks of 32KB
                    data = await reader.read(sz)
                    if not data:
                        break
                    await segment.write(data)
                    if first_segment:
                        first_segment = False
                        if monitoring_callback:
                            await monitoring_callback({
                                "type": "send_first_processed_segment",
                                "timestamp": int(time.time() * 1000)
                            })
                transport.close()

        def sync_callback(pipe_reader, pipe_writer, pipe_name):
            def do_schedule():
                _schedule_callback(callback(pipe_reader, pipe_name), pipe_writer, pipe_name, loop)
            loop.call_soon_threadsafe(do_schedule)

        # hold tasks since `loop.create_task` is a weak reference that gets GC'd
        live_tasks = set()
        live_pipes = set()
        live_tasks_lock = threading.Lock()

        def _schedule_callback(coro, pipe_writer, pipe_name, loop):
            task = loop.create_task(coro)
            with live_tasks_lock:
                live_tasks.add(task)
                live_pipes.add(pipe_writer)
            
            def task_done2(t: asyncio.Task, p):
                try:
                    t.result()
                except Exception as e:
                    logger.error(f"Task {pipe_name} crashed: {e}")
                with live_tasks_lock:
                    live_tasks.remove(t)
                    live_pipes.remove(p)
            
            def task_done(t2: asyncio.Task):
                return task_done2(task, pipe_writer)
            
            task.add_done_callback(task_done)

        encode_thread = threading.Thread(
            target=_encode_in, 
            args=(live_pipes, live_tasks_lock, frame_generator, sync_callback, get_metadata),
            kwargs={"audio_codec": "libopus", "detect_out_resolution": detect_out_resolution}
        )
        encode_thread.start()
        logger.debug("run_publish: encoder thread started")

        # Wait for encode thread to complete
        def joins():
            encode_thread.join()
        await asyncio.to_thread(joins)

        # wait for IO tasks to complete
        while True:
            with live_tasks_lock:
                current_tasks = list(live_tasks)
            if not current_tasks:
                break  # nothing left to wait on
            await asyncio.wait(current_tasks, return_when=asyncio.ALL_COMPLETED)
            # loop in case another task was added while awaiting

        logger.info("run_publish complete")

    except Exception as e:
        logger.error(f"postprocess got error {e}", e)
        raise e
    finally:
        if publisher:
            await publisher.close()

def _encode_in(task_pipes, task_lock, frame_generator, sync_callback, get_metadata, **kwargs):
    """Encode frames with retry logic."""
    # encode_av has a tendency to crash, so restart as necessary
    retry_count = 0
    last_retry_time = time.time()
    while retry_count < MAX_ENCODER_RETRIES:
        try:
            encode_av(frame_generator, sync_callback, get_metadata, **kwargs)
            break  # clean exit
        except Exception as exc:
            current_time = time.time()
            # Reset retry counter if enough time has elapsed
            if current_time - last_retry_time > ENCODER_RETRY_RESET_SECONDS:
                logger.info("Resetting encoder retry count")
                retry_count = 0
            retry_count += 1
            last_retry_time = current_time
            if retry_count < MAX_ENCODER_RETRIES:
                logger.exception(f"Error in encode_av, retrying {retry_count}/{MAX_ENCODER_RETRIES}")
            else:
                logger.exception("Error in encode_av, maximum retries reached")
            # close leftover writer ends of any pipes to prevent hanging
            pipe_count = 0
            total_pipes = 0
            with task_lock:
                pipes = list(task_pipes)
                total_pipes = len(pipes)
                for p in pipes:
                    try:
                        p.close()
                        pipe_count += 1
                    except Exception as e:
                        logger.exception("Error closing pipe on task list")
            logger.info(f"Closed pipes - {pipe_count}/{total_pipes}") 