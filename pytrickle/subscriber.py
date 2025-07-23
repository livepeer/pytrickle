"""
Trickle Subscriber for receiving streaming data over HTTP.

Handles the subscription to video/audio segments from trickle endpoints
with automatic reconnection and error handling.
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Callable, Any, List

from . import ErrorCallback
from .base import TrickleComponent

logger = logging.getLogger(__name__)

class TrickleSubscriber(TrickleComponent):
    """Trickle subscriber for receiving data from a URL."""
    
    def __init__(self, url: str, start_seq: int = -2, max_retries: int = 5, error_callback: Optional[ErrorCallback] = None):
        super().__init__(error_callback)
        self.base_url = url
        self.idx = start_seq
        self.max_retries = max_retries
        self.pending_get: Optional[aiohttp.ClientResponse] = None
        self.lock = asyncio.Lock()
        self._background_tasks: List[asyncio.Task] = []  # Track background tasks
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Enter context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    async def start(self):
        """Start the subscriber session."""
        if not self.session:
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(connector=connector)

    async def preconnect(self) -> Optional[aiohttp.ClientResponse]:
        """
        Preconnect to the server by making a GET request to fetch the next segment.
        For any non-200 responses, retries up to max_retries unless a 404 is encountered.
        """
        if not self.session:
            await self.start()
            
        url = f"{self.base_url}/{self.idx}"
        for attempt in range(self.max_retries):
            logger.info(f"Trickle sub preconnecting attempt: {attempt} URL: {url}")
            try:
                if not self.session:
                    logger.error("Session is not initialized")
                    await self._notify_error("session_not_initialized", Exception("Session is not initialized"))
                    return None
                resp = await self.session.get(url, headers={'Connection': 'close'})

                if resp.status == 200:
                    # Return the response for later processing
                    return resp

                if resp.status == 404:
                    logger.info(f"Trickle sub got 404, terminating {url}")
                    resp.release()
                    await self._notify_error("stream_not_found", Exception(f"404 error for {url}"))
                    return None

                if resp.status == 470:
                    # Channel exists but no data at this index, so reset
                    idx = resp.headers.get('Lp-Trickle-Latest') or '-1'
                    url = f"{self.base_url}/{idx}"
                    logger.info(f"Trickle sub resetting index to leading edge {url}")
                    resp.release()
                    # Continue immediately
                    continue

                body = await resp.text()
                resp.release()
                logger.error(f"Trickle sub failed GET {url} status code: {resp.status}, msg: {body}")

            except Exception:
                logger.exception(f"Trickle sub failed to complete GET {url}", stack_info=True)

            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.5)

        # Max retries hit, so bail out
        logger.error(f"Trickle sub hit max retries, exiting {url}")
        await self._notify_error("max_retries_exceeded", Exception(f"Max retries exceeded for {url}"))
        return None

    async def next(self) -> Optional['Segment']:
        """Retrieve data from the current segment and set up the next segment concurrently."""
        async with self.lock:
            if self._should_stop():
                logger.info(f"Trickle subscription closed or errored for {self.base_url}")
                return None

            # If we don't have a pending GET request, preconnect
            if self.pending_get is None:
                logger.info("Trickle sub no pending connection, preconnecting...")
                self.pending_get = await self.preconnect()

            # Extract the current connection to use for reading
            resp = self.pending_get
            self.pending_get = None

            # Preconnect has failed, notify caller
            if resp is None:
                return None

            # Create segment from response
            segment = Segment(resp)

            if segment.eos():
                return None

            idx = segment.seq()
            if idx >= 0:
                self.idx = idx + 1

            # Set up the next connection in the background
            preconnect_task = asyncio.create_task(self._preconnect_next_segment())
            self._background_tasks.append(preconnect_task)
            logger.debug(f"Created background preconnect task for {self.base_url}, total tasks: {len(self._background_tasks)}")
            # Add callback to remove completed tasks from the list
            preconnect_task.add_done_callback(lambda t: self._background_tasks.remove(t) if t in self._background_tasks else None)

        return segment

    async def _preconnect_next_segment(self):
        """Preconnect to the next segment in the background."""
        # Check if we should stop before doing expensive operations
        if self._should_stop():
            logger.debug(f"Skipping preconnect for {self.base_url} - shutdown/error state")
            return
            
        logger.debug(f"Starting background preconnect for {self.base_url}")
        async with self.lock:
            if self.pending_get is not None:
                logger.debug(f"Pending connection already exists for {self.base_url}")
                return
            if self._should_stop():
                logger.debug(f"Shutdown detected during preconnect for {self.base_url}")
                return
            next_conn = await self.preconnect()
            if next_conn:
                self.pending_get = next_conn
                logger.debug(f"Preconnected successfully for {self.base_url}")
            else:
                logger.debug(f"Preconnect failed for {self.base_url}")

    async def close(self):
        """Close the session when done."""
        logger.info(f"Closing {self.base_url}")
        self.shutdown_event.set()  # Signal shutdown first
        
        # Cancel all background preconnect tasks
        if self._background_tasks:
            logger.info(f"Cancelling {len(self._background_tasks)} background preconnect tasks for {self.base_url}")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    logger.debug(f"Cancelled preconnect task for {self.base_url}: {task}")
            # Wait briefly for tasks to cancel
            try:
                await asyncio.wait_for(asyncio.gather(*self._background_tasks, return_exceptions=True), timeout=2.0)
                logger.info(f"All background tasks cancelled for {self.base_url}")
            except asyncio.TimeoutError:
                logger.warning(f"Some background tasks did not cancel within timeout for {self.base_url}")
            except Exception as e:
                logger.debug(f"Expected errors during task cancellation for {self.base_url}: {e}")
            self._background_tasks.clear()
        else:
            logger.info(f"No background tasks to cancel for {self.base_url}")
        
        async with self.lock:
            if self.pending_get:
                self.pending_get.close()
                self.pending_get = None
            if self.session:
                try:
                    await self.session.close()
                    logger.info(f"Session closed for {self.base_url}")
                except Exception:
                    logger.error(f"Error closing trickle subscriber session for {self.base_url}", exc_info=True)
                finally:
                    self.session = None

class Segment:
    """Represents a single trickle segment."""
    
    def __init__(self, response: aiohttp.ClientResponse):
        self.response = response

    def seq(self) -> int:
        """Extract the sequence number from the response headers."""
        seq_str = self.response.headers.get('Lp-Trickle-Seq')
        if seq_str is None:
            return -1
        try:
            seq = int(seq_str)
        except (TypeError, ValueError):
            return -1
        return seq

    def eos(self) -> bool:
        """Check if this is the end of stream."""
        return self.response.headers.get('Lp-Trickle-Closed') is not None

    async def read(self, chunk_size: int = 32 * 1024) -> Optional[bytes]:
        """Read the next chunk of the segment."""
        if not self.response:
            await self.close()
            return None
        chunk = await self.response.content.read(chunk_size)
        if not chunk:
            await self.close()
        return chunk

    async def close(self):
        """Ensure the response is properly closed when done."""
        if self.response is None:
            return
        if not self.response.closed:
            self.response.release()
            self.response.close() 