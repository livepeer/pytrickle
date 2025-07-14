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
from typing import AsyncGenerator, Optional, Callable

from .frames import InputFrame, OutputFrame, AudioFrame, AudioOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT
from .media import run_subscribe, run_publish
from .subscriber import TrickleSubscriber, create_subscriber
from .publisher import TricklePublisher, create_publisher
from .exceptions import ErrorPropagator

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
        width: Optional[int] = DEFAULT_WIDTH, 
        height: Optional[int] = DEFAULT_HEIGHT,
        max_retries: int = 3
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.width = width
        self.height = height
        self.max_retries = max_retries
        
        # Internal queues for frame processing
        self.subscribe_queue = queue.Queue()
        self.publish_queue = queue.Queue()
        
        # Error handling
        self.error_propagator = ErrorPropagator()
        
        # Control and events components
        self.control_subscriber: Optional[TrickleSubscriber] = None
        self.events_publisher: Optional[TricklePublisher] = None
        
        # Background tasks
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        
        # Shutdown handling
        self.shutdown_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the trickle protocol."""
        logger.info(f"Starting trickle protocol: subscribe={self.subscribe_url}, publish={self.publish_url}")
        
        # Initialize queues
        self.subscribe_queue = queue.Queue()
        self.publish_queue = queue.Queue()
        
        # Metadata cache to pass video metadata from decoder to encoder
        metadata_cache = LastValueCache()
        
        # Start shutdown monitor
        self.shutdown_task = asyncio.create_task(self._shutdown_monitor())
        
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
        self.control_subscriber = create_subscriber(
            self.control_url, 
            is_optional=True, 
            max_retries=self.max_retries,
            error_propagator=self.error_propagator
        )
        if self.control_subscriber:
            logger.info(f"Control subscriber initialized for: {self.control_url}")
        
        # Initialize events publisher if URL provided
        self.events_publisher = create_publisher(
            self.events_url, 
            mime_type="application/json", 
            is_optional=True,
            max_retries=self.max_retries,
            error_propagator=self.error_propagator
        )
        if self.events_publisher:
            await self.events_publisher.start()
            logger.info(f"Events publisher initialized for: {self.events_url}")
    
    async def _shutdown_monitor(self):
        """Monitor for shutdown signals and handle graceful shutdown."""
        try:
            await self.error_propagator.wait_for_shutdown()
            logger.info("Shutdown signal received, stopping trickle protocol")
            
            # Signal all components to stop
            self.subscribe_queue.put(None)
            self.publish_queue.put(None)
            
        except asyncio.CancelledError:
            logger.debug("Shutdown monitor cancelled")
        except Exception as e:
            logger.error(f"Error in shutdown monitor: {e}")

    async def stop(self):
        """Stop the trickle protocol."""
        logger.info("Stopping trickle protocol")
        
        # Cancel shutdown monitor
        if self.shutdown_task:
            self.shutdown_task.cancel()
            try:
                await self.shutdown_task
            except asyncio.CancelledError:
                pass
            self.shutdown_task = None
        
        # Send sentinel None values to stop the trickle tasks gracefully
        try:
            self.subscribe_queue.put(None)
            self.publish_queue.put(None)
        except Exception as e:
            logger.debug(f"Error sending sentinel values: {e}")

        # Close control and events components
        if self.control_subscriber:
            try:
                await self.control_subscriber.close()
            except Exception as e:
                logger.debug(f"Error closing control subscriber: {e}")
            finally:
                self.control_subscriber = None

        if self.events_publisher:
            try:
                await self.events_publisher.close()
            except Exception as e:
                logger.debug(f"Error closing events publisher: {e}")
            finally:
                self.events_publisher = None

        # Wait for tasks to complete with timeout
        if self.subscribe_task or self.publish_task:
            tasks = []
            if self.subscribe_task:
                tasks.append(self.subscribe_task)
            if self.publish_task:
                tasks.append(self.publish_task)
                
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
                logger.info("All tasks completed gracefully")
            except asyncio.TimeoutError:
                logger.warning("Tasks did not complete within timeout, canceling...")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Give a moment for cancellation to complete
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not respond to cancellation")

        self.subscribe_task = None
        self.publish_task = None

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        """Generate frames from the ingress stream."""
        def dequeue_frame():
            try:
                frame = self.subscribe_queue.get(timeout=0.1)
                return frame if frame else None
            except queue.Empty:
                return None

        while not done.is_set() and not self.error_propagator.is_shutdown_requested():
            frame = await asyncio.to_thread(dequeue_frame)
            if frame is None:
                # Check if we should continue waiting or if there's an error
                if self.error_propagator.is_shutdown_requested():
                    logger.info("Shutdown requested, stopping ingress loop")
                    break
                continue
            if frame is None:  # Sentinel value
                logger.info("Received sentinel value, stopping ingress loop")
                break
            
            # Handle audio frames (pass through for now)
            if isinstance(frame, AudioFrame):
                self.publish_queue.put(AudioOutput([frame], ""))
                continue
            
            yield frame

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consume output frames and send them to the publish queue."""
        def enqueue_frame(frame: OutputFrame):
            try:
                self.publish_queue.put(frame)
            except Exception as e:
                logger.error(f"Error enqueuing frame: {e}")

        async for frame in output_frames:
            if self.error_propagator.is_shutdown_requested():
                logger.info("Shutdown requested, stopping egress loop")
                break
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
            logger.debug(f"Error reporting status: {e}")

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """Generate control messages from the control stream."""
        if not self.control_subscriber:
            logger.info("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logger.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        try:
            while not done.is_set() and not self.error_propagator.is_shutdown_requested():
                try:
                    segment = await self.control_subscriber.next()
                    if not segment or segment.eos():
                        logger.info("Control stream ended")
                        return

                    params = await segment.read()
                    if not params:
                        continue
                        
                    try:
                        data = json.loads(params)
                        if data == keepalive_message:
                            # Ignore periodic keepalive messages
                            continue

                        logger.info("Received control message with params: %s", data)
                        yield data
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in control message: {e}")
                        continue

                except Exception as e:
                    logger.error(f"Error in control loop: {e}")
                    # For control loop errors, just continue unless shutdown is requested
                    if self.error_propagator.is_shutdown_requested():
                        break
                    await asyncio.sleep(0.1)  # Brief pause before retrying
                    continue
        except Exception as e:
            logger.error(f"Fatal error in control loop: {e}")
        finally:
            logger.info("Control loop ended") 