"""
Trickle Publisher for sending streaming data over HTTP.

Handles the publishing of video/audio segments to trickle endpoints
with automatic preconnection and background task management.
"""

import asyncio
import aiohttp
import logging
from typing import Optional, List

from . import ErrorCallback
from .base import TrickleComponent

logger = logging.getLogger(__name__)

# Default timeouts (seconds)
CONNECT_TIMEOUT_SECONDS = 30
KEEPALIVE_TIMEOUT_SECONDS = 5

# Backoff for immediate retry when preconnect returns None
RETRY_DELAY_SECONDS = 0.05

class TricklePublisher(TrickleComponent):
    """Trickle publisher for sending data to a URL."""
    
    def __init__(self, url: str, mime_type: str = "video/mp4", error_callback: Optional[ErrorCallback] = None, component_name: Optional[str] = "publisher"):
        super().__init__(error_callback, component_name)
        self.url = url
        self.mime_type = mime_type
        self.idx = 0
        self.next_writer: Optional[asyncio.Queue] = None
        self.lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None
        self.connect_timeout_seconds = CONNECT_TIMEOUT_SECONDS
        self.keepalive_timeout_seconds = KEEPALIVE_TIMEOUT_SECONDS

    async def __aenter__(self):
        """Enter context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    async def start(self):
        """Start the publisher session."""
        # Always create a fresh session on start to avoid stale keep-alives across streams
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")
                pass
            finally:
                self.session = None
        connector = aiohttp.TCPConnector(
            verify_ssl=False,
            limit=0,
            keepalive_timeout=self.keepalive_timeout_seconds,
        )
        timeout = aiohttp.ClientTimeout(total=self.connect_timeout_seconds)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    def stream_idx(self):
        """Return the current stream index."""
        return f"{self.url}/{self.idx}"

    async def preconnect(self) -> Optional[asyncio.Queue]:
        """Preconnect to the server by initiating a POST request to the current index."""
        if not self.session:
            await self.start()
            
        url = self.stream_idx()
        logger.info(f"Preconnecting to URL: {url}")
        
        # Check shutdown status before expensive operations
        if self._should_stop():
            logger.info(f"Publisher shutting down, aborting preconnect to {url}")
            return None
            
        try:
            # Create a queue for streaming data incrementally
            queue = asyncio.Queue()
            self._track_background_task(asyncio.create_task(self._run_post(url, queue)))
            return queue
        except Exception as e:
            logger.error(f"Failed to complete POST for {url}: {e}")
            await self._notify_error("connection_failed", e)
            return None

    async def _run_post(self, url: str, queue: asyncio.Queue):
        """Run the POST request with streaming data."""
        try:
            if not self.session or self._should_stop():
                return
                
            resp = await self.session.post(
                url,
                headers={'Connection': 'close', 'Content-Type': self.mime_type},
                data=self._stream_data(queue)
            )
            
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Trickle POST failed {url}, status code: {resp.status}, msg: {body}")
                # Don't trigger error callback if we're shutting down
                if not self._should_stop():
                    await self._notify_error("post_failed", Exception(f"POST failed with status {resp.status}: {body}"))
        except aiohttp.ClientConnectionResetError as e:
            # Connection reset is usually due to server shutdown - treat as expected
            logger.debug(f"Connection reset for {url} (expected during shutdown): {e}")
        except asyncio.CancelledError:
            # Handle cancellation gracefully during shutdown
            logger.debug(f"POST request cancelled for {url} during shutdown")
            raise
        except Exception as e:
            # Don't trigger error callback if we're shutting down - this is expected
            if not self._should_stop():
                logger.error(f"Trickle POST exception {url} - {e}")
                await self._notify_error("post_exception", e)
            else:
                logger.debug(f"POST request failed for {url} during shutdown (expected): {e}")

    async def _run_delete(self):
        """Send DELETE request to clean up stream."""
        try:
            if self.session:
                resp = await self.session.delete(self.url)
                resp.release()
        except Exception:
            logger.error(f"Error sending trickle delete request", exc_info=True)

    async def _stream_data(self, queue: asyncio.Queue):
        """Stream data from the queue for the POST request."""
        while not self._should_stop():
            try:
                # Use a short timeout to check shutdown status frequently
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                if chunk is None:  # Stop signal
                    break
                yield chunk
            except asyncio.TimeoutError:
                # Check shutdown status and continue if still running
                continue
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                break

    async def next(self):
        """Start or retrieve a pending POST request and preconnect for the next segment."""
        async with self.lock:
            # Allow recovery from transient error state between encoder retries
            # Note: This allows for encoder restarts without losing data
            if self.error_event.is_set() and not self.shutdown_event.is_set():
                logger.info("Publisher recovering from error state for next segment")
                # Clear error and resume running state
                self.error_event.clear()
                self.component_state = self.component_state.RUNNING
                self.last_error = None

            if self._should_stop():
                logger.info(f"Publisher is in error or shutdown state, cannot get next segment")
                return SegmentWriter(None)
                
            if self.next_writer is None:
                logger.info(f"No pending connection, preconnecting {self.stream_idx()}...")
                self.next_writer = await self.preconnect()
                # If preconnect failed due to connection reuse issues, retry once immediately
                if self.next_writer is None and not self._should_stop():
                    logger.info("Preconnect returned None, retrying once immediately")
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    self.next_writer = await self.preconnect()

            writer = self.next_writer
            self.next_writer = None

            # Only create background task if not shutting down
            if not self._should_stop():
                self._track_background_task(asyncio.create_task(self._preconnect_next_segment()))

        return SegmentWriter(writer)

    async def _preconnect_next_segment(self):
        """Preconnect to the next POST in the background."""
        # Check if we should stop before doing expensive operations
        if self._should_stop():
            return
            
        logger.info(f"Setting up next connection for {self.stream_idx()}")
        async with self.lock:
            # Double-check shutdown status under lock
            if self._should_stop():
                return
                
            if self.next_writer is not None:
                return
                
            self.idx += 1  # Increment the index for the next POST
            next_writer = await self.preconnect()
            if next_writer and not self._should_stop():
                self.next_writer = next_writer

    async def close(self):
        """Close the session when done."""
        logger.info(f"Closing {self.url}")
        self.shutdown_event.set()  # Signal shutdown first
        
        # Cancel all background preconnect tasks immediately
        if self._background_tasks:
            logger.info(f"Cancelling {len(self._background_tasks)} background preconnect tasks")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Force immediate cancellation without waiting
            self._background_tasks.clear()
            logger.info("All background tasks cancelled immediately")
        
        async with self.lock:
            if self.next_writer:
                segment = SegmentWriter(self.next_writer)
                await segment.close()
                self.next_writer = None
            if self.session:
                try:
                    # Skip delete request during shutdown for faster cleanup
                    await self.session.close()
                    logger.info("Session closed immediately")
                except Exception as e:
                    logger.debug(f"Expected error during fast session close: {e}")
                finally:
                    self.session = None


class SegmentWriter:
    """Writer for individual trickle segments."""
    
    def __init__(self, queue: Optional[asyncio.Queue]):
        self.queue = queue

    async def write(self, data: bytes):
        """Write data to the current segment."""
        if self.queue:
            await self.queue.put(data)

    async def close(self):
        """Ensure the request is properly closed when done."""
        if self.queue:
            await self.queue.put(None)  # Send None to signal end of data

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the connection."""
        await self.close()