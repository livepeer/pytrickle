"""
Trickle Protocol implementation for video streaming.

Provides a high-level interface for trickle-based video streaming,
integrating subscription, publishing, and media processing.
"""

import asyncio
import logging
import queue
import json
import threading
from typing import AsyncGenerator, Optional, Callable, Union, Coroutine, Any

from .frames import InputFrame, OutputFrame, AudioFrame, AudioOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT
from .media import run_subscribe, run_publish
from .subscriber import TrickleSubscriber
from .publisher import TricklePublisher

logger = logging.getLogger(__name__)

class LastValueCache:
    """Thread-safe cache for storing the last value."""
    
    def __init__(self):
        self._value = None
        self._lock = threading.Lock()
    
    def put(self, value):
        """Store a value in the cache."""
        with self._lock:
            self._value = value
    
    def get(self):
        """Retrieve the last stored value."""
        with self._lock:
            return self._value

class TrickleProtocol:
    """High-level trickle protocol implementation."""
    
    def __init__(
        self, 
        subscribe_url: str, 
        publish_url: str, 
        control_url: Optional[str] = None, 
        events_url: Optional[str] = None,
        secondary_publish_url: Optional[str] = None,
        secondary_publish_type: str = "text",
        width: Optional[int] = DEFAULT_WIDTH, 
        height: Optional[int] = DEFAULT_HEIGHT,
        error_callback: Optional[Union[Callable[[str, Optional[Exception]], None], Callable[[str, Optional[Exception]], Coroutine[Any, Any, None]]]] = None
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.secondary_publish_url = secondary_publish_url
        self.secondary_publish_type = secondary_publish_type
        self.width = width
        self.height = height
        self.error_callback = error_callback
        
        # Internal queues for frame processing
        self.subscribe_queue = queue.Queue()
        self.publish_queue = queue.Queue()
        self.secondary_publish_queue = queue.Queue()
        
        # Control and events components
        self.control_subscriber: Optional[TrickleSubscriber] = None
        self.events_publisher: Optional[TricklePublisher] = None
        self.secondary_publisher: Optional[TricklePublisher] = None
        
        # Background tasks
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        self.secondary_publish_task: Optional[asyncio.Task] = None
        
        # Error state tracking with Events
        self.error_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()

    async def _notify_error(self, error_type: str, exception: Optional[Exception] = None):
        """Notify parent component of critical errors."""
        self.error_event.set()  # Set error event
        if self.error_callback:
            try:
                if asyncio.iscoroutinefunction(self.error_callback):
                    await self.error_callback(error_type, exception)
                else:
                    self.error_callback(error_type, exception)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def _on_component_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle errors from subscriber/publisher components."""
        logger.error(f"Component error: {error_type} - {exception}")
        await self._notify_error(error_type, exception)

    async def start(self):
        """Start the trickle protocol."""
        logger.info(f"Starting trickle protocol: subscribe={self.subscribe_url}, publish={self.publish_url}")
        
        # Initialize queues
        self.subscribe_queue = queue.Queue()
        self.publish_queue = queue.Queue()
        self.secondary_publish_queue = queue.Queue()
        
        # Metadata cache to pass video metadata from decoder to encoder
        metadata_cache = LastValueCache()
        
        # Start subscribe and publish tasks
        self.subscribe_task = asyncio.create_task(
            run_subscribe(
                self.subscribe_url, 
                self.subscribe_queue.put, 
                metadata_cache.put, 
                self.emit_monitoring_event, 
                self.width or DEFAULT_WIDTH, 
                self.height or DEFAULT_HEIGHT
            )
        )
        
        self.publish_task = asyncio.create_task(
            run_publish(
                self.publish_url, 
                self.publish_queue.get, 
                metadata_cache.get, 
                self.emit_monitoring_event
            )
        )
        
        # Initialize control subscriber if URL provided
        if self.control_url and self.control_url.strip():
            self.control_subscriber = TrickleSubscriber(self.control_url, error_callback=self._on_component_error)
            
        # Initialize events publisher if URL provided
        if self.events_url and self.events_url.strip():
            self.events_publisher = TricklePublisher(self.events_url, "application/json", error_callback=self._on_component_error)
            await self.events_publisher.start()
            
        # Initialize secondary publisher if URL provided
        if self.secondary_publish_url and self.secondary_publish_url.strip():
            # Determine mime type based on secondary publish type
            if self.secondary_publish_type == "text":
                mime_type = "application/json"
            elif self.secondary_publish_type in ["video", "audio"]:
                mime_type = "video/mp2t"  # Use same encoding as main publish channel
            else:
                logger.warning(f"Unknown secondary publish type: {self.secondary_publish_type}, defaulting to text")
                mime_type = "application/json"
            
            self.secondary_publisher = TricklePublisher(self.secondary_publish_url, mime_type, error_callback=self._on_component_error)
            await self.secondary_publisher.start()
            
            # For video/audio types, start a publish task similar to main channel
            if self.secondary_publish_type in ["video", "audio"]:
                self.secondary_publish_task = asyncio.create_task(
                    run_publish(
                        self.secondary_publish_url,
                        self.secondary_publish_queue.get,
                        metadata_cache.get,
                        self.emit_monitoring_event
                    )
                )

    async def stop(self):
        """Stop the trickle protocol."""
        logger.info("Stopping trickle protocol")
        
        if not self.subscribe_task or not self.publish_task:
            return  # already stopped

        # Send sentinel None values to stop the trickle tasks gracefully
        self.subscribe_queue.put(None)
        self.publish_queue.put(None)
        if self.secondary_publish_task:
            self.secondary_publish_queue.put(None)

        # Close control and events components
        if self.control_subscriber:
            await self.control_subscriber.close()
            self.control_subscriber = None

        if self.events_publisher:
            await self.events_publisher.close()
            self.events_publisher = None
            
        if self.secondary_publisher:
            await self.secondary_publisher.close()
            self.secondary_publisher = None

        # Wait for tasks to complete with timeout
        tasks = [self.subscribe_task, self.publish_task]
        if self.secondary_publish_task:
            tasks.append(self.secondary_publish_task)
            
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Tasks did not complete within timeout, canceling...")
            for task in tasks:
                task.cancel()

        self.subscribe_task = None
        self.publish_task = None
        self.secondary_publish_task = None

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        """Generate frames from the ingress stream."""
        def dequeue_frame():
            try:
                frame = self.subscribe_queue.get(timeout=0.1)
                return frame if frame else None
            except queue.Empty:
                return None

        while not done.is_set():
            frame = await asyncio.to_thread(dequeue_frame)
            if frame is None:
                continue
            if frame is None:  # Sentinel value
                break
            
            # Handle audio frames (pass through for now)
            if isinstance(frame, AudioFrame):
                self.publish_queue.put(AudioOutput([frame], ""))
                continue
            
            yield frame

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consume output frames and send them to the publish queue."""
        def enqueue_frame(frame: OutputFrame):
            self.publish_queue.put(frame)

        async for frame in output_frames:
            await asyncio.to_thread(enqueue_frame, frame)

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Emit monitoring events via the events publisher."""
        if not self.events_publisher:
            return
        try:
            event_json = json.dumps({"event": event, "queue_event_type": queue_event_type})
            async with await self.events_publisher.next() as segment:
                await segment.write(event_json.encode())
        except Exception as e:
            logger.error(f"Error reporting status: {e}")

    async def send_secondary_text(self, data: dict):
        """Send JSON text data to the secondary publish channel (text type only)."""
        if not self.secondary_publisher or self.secondary_publish_type != "text":
            return
        try:
            data_json = json.dumps(data)
            async with await self.secondary_publisher.next() as segment:
                await segment.write(data_json.encode())
            logger.debug(f"Sent secondary text data: {data}")
        except Exception as e:
            logger.error(f"Error sending secondary text data: {e}")

    async def send_secondary_frame(self, frame: OutputFrame):
        """Send video/audio frame to the secondary publish channel (video/audio type only)."""
        if not self.secondary_publish_queue or self.secondary_publish_type not in ["video", "audio"]:
            return
        try:
            self.secondary_publish_queue.put(frame)
            logger.debug(f"Queued secondary frame: {type(frame)}")
        except Exception as e:
            logger.error(f"Error sending secondary frame: {e}")

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """Generate control messages from the control stream."""
        if not self.control_subscriber:
            logger.warning("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logger.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        while not done.is_set():
            try:
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    return

                params = await segment.read()
                if not params:
                    continue
                data = json.loads(params)
                if data == keepalive_message:
                    # Ignore periodic keepalive messages
                    continue

                logger.info("Received control message with params: %s", data)
                yield data

            except Exception:
                logger.error(f"Error in control loop", exc_info=True)
                continue 