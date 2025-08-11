"""
Trickle Protocol implementation for video streaming.

Provides a high-level interface for trickle-based video streaming,
integrating subscription, publishing, and media processing.
"""

import asyncio
import json
import queue
import logging
import time
from typing import Optional, AsyncGenerator, Callable

from .base import TrickleComponent
from .subscriber import TrickleSubscriber
from .publisher import TricklePublisher
from .media import run_subscribe, run_publish
from .frames import InputFrame, OutputFrame, AudioFrame, AudioOutput
from .cache import LastValueCache
from . import ErrorCallback

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

class TrickleProtocol(TrickleComponent):
    """Trickle protocol coordinator that manages subscription, publishing, control, and events."""
    
    def __init__(
        self,
        subscribe_url: str,
        publish_url: str,
        control_url: str = "",
        events_url: str = "",
        data_url: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        error_callback: Optional[ErrorCallback] = None,
        heartbeat_interval: float = 10.0,
    ):
        super().__init__(error_callback)
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.data_url = data_url
        self.width = width
        self.height = height
        self.heartbeat_interval = heartbeat_interval
        
        # Tasks and components
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        self.control_subscriber: Optional[TrickleSubscriber] = None
        self.events_publisher: Optional[TricklePublisher] = None
        self.data_publisher: Optional[TricklePublisher] = None

        # Background tasks
        self.subscribe_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        
        # Coordination events
        self.subscription_ended = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def _on_component_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle errors from subscriber/publisher components."""
        logger.error(f"Component error: {error_type} - {exception}")
        await self._notify_error(error_type, exception)

    async def _monitor_subscription_end(self):
        """Monitor for subscription task completion and trigger immediate shutdown."""
        try:
            if self.subscribe_task:
                # Wait for subscription task to complete
                await self.subscribe_task
                logger.info("Subscription task completed, triggering immediate shutdown of events publisher")
                
                # Set shutdown and subscription ended events first
                self.shutdown_event.set()
                self.subscription_ended.set()
                
                # Immediately signal shutdown to events publisher to stop background tasks
                if self.events_publisher:
                    await self.events_publisher.shutdown()
                    logger.info("Events publisher shutdown signaled due to subscription end")
                # Stop heartbeat task on subscription end
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                
                # Also signal shutdown to control subscriber
                if self.control_subscriber:
                    await self.control_subscriber.shutdown()
                    logger.info("Control subscriber shutdown signaled due to subscription end")
                
        except Exception as e:
            logger.error(f"Error in subscription monitor: {e}")

    async def start(self):
        """Start the trickle protocol."""
        logger.info(f"Starting trickle protocol: subscribe={self.subscribe_url}, publish={self.publish_url}")
        
        # Initialize queues
        self.subscribe_queue = queue.Queue()
        self.publish_queue = queue.Queue()
        
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
            # Start protocol-level heartbeat loop if enabled
            if self.heartbeat_interval and self.heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
        # Initialize data publisher if URL provided
        if self.data_url and self.data_url.strip():
            self.data_publisher = TricklePublisher(self.data_url, "application/octet-stream", error_callback=self._on_component_error)
            await self.data_publisher.start()
        
        # Start monitoring subscription end for immediate cleanup
        self._monitor_task = asyncio.create_task(self._monitor_subscription_end())

    async def stop(self):
        """Stop the trickle protocol."""
        logger.info("Stopping trickle protocol")
        
        # Signal shutdown immediately to all components
        self.shutdown_event.set()
        
        # Stop heartbeat task first
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None

        # Stop monitoring task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Immediately signal shutdown to events publisher to stop background tasks
        if self.events_publisher:
            await self.events_publisher.shutdown()
        
        # Immediately signal shutdown to control subscriber
        if self.control_subscriber:
            await self.control_subscriber.shutdown()
        
        # Send sentinel None values to stop the trickle tasks gracefully if queues exist
        if self.subscribe_queue:
            self.subscribe_queue.put(None)
        if self.publish_queue:
            self.publish_queue.put(None)

        # Close control and events components
        if self.control_subscriber:
            await self.control_subscriber.close()
            self.control_subscriber = None

        if self.events_publisher:
            await self.events_publisher.close()
            self.events_publisher = None

        if self.data_publisher:
            await self.data_publisher.close()
            self.data_publisher = None

        # Wait for tasks to complete with timeout
        tasks = []
        if self.subscribe_task:
            tasks.append(self.subscribe_task)
        if self.publish_task:
            tasks.append(self.publish_task)
            
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Tasks did not complete within timeout, canceling...")
                for task in tasks:
                    if not task.done():
                        task.cancel()

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

        while not done.is_set() and not self.shutdown_event.is_set():
            frame = await asyncio.to_thread(dequeue_frame)
            if frame is None:
                continue
            if frame is None:  # Sentinel value
                break
            
            # Send all frames to frame processor
            yield frame

    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consume output frames and send them to the publish queue."""
        def enqueue_frame(frame: OutputFrame):
            self.publish_queue.put(frame)

        try:
            async for frame in output_frames:
                # Check if subscription has ended or shutdown is signaled
                if self.subscription_ended.is_set() or self.shutdown_event.is_set():
                    logger.info("Subscription ended or shutdown signaled, ending egress loop")
                    break
                    
                await asyncio.to_thread(enqueue_frame, frame)
        except Exception as e:
            logger.error(f"Error in egress loop: {e}")
            # Re-raise to trigger error handling in the client
            raise

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Emit monitoring events via the events publisher."""
        if not self.events_publisher or self.shutdown_event.is_set():
            return
        try:
            event_json = json.dumps({"event": event, "queue_event_type": queue_event_type})
            async with await self.events_publisher.next() as segment:
                await segment.write(event_json.encode())
        except Exception as e:
            logger.error(f"Error reporting status: {e}")

    async def _heartbeat_loop(self):
        """Emit minimal liveness events at a fixed cadence."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    heartbeat_event = {
                        "type": "heartbeat",
                        "timestamp": __import__("time").time(),
                    }
                    # Use milliseconds for timestamp consistency
                    heartbeat_event["timestamp"] = int(heartbeat_event["timestamp"] * 1000)
                    heartbeat_event.update({
                        "urls": {
                            "subscribe": self.subscribe_url,
                            "publish": self.publish_url,
                            "control": self.control_url or "",
                            "events": self.events_url or "",
                            "data": self.data_url or "",
                        },
                        "dimensions": {
                            "width": self.width or DEFAULT_WIDTH,
                            "height": self.height or DEFAULT_HEIGHT,
                        },
                    })
                    await self.emit_monitoring_event(heartbeat_event, queue_event_type="stream_heartbeat")
                except Exception as e:
                    logger.warning(f"Heartbeat emission error: {e}")
                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.heartbeat_interval)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """Generate control messages from the control stream."""
        if not self.control_subscriber:
            logger.warning("No control-url provided, inference won't get updates from the control trickle subscription")
            return

        logger.info("Starting Control subscriber at %s", self.control_url)
        keepalive_message = {"keep": "alive"}

        while not done.is_set() and not self.shutdown_event.is_set():
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

    async def publish_data(self, data: str):
        """Publish data via the data publisher."""
        if not self.data_publisher:
            return
        try:
            async with await self.data_publisher.next() as segment:
                await segment.write(data.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error publishing data: {e}")