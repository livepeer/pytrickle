"""
Trickle Publisher for sending streaming data over HTTP.

Handles the publication of video/audio segments to trickle endpoints
with automatic reconnection and error handling.
"""

import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

class TricklePublisher:
    """Publisher for streaming data to trickle endpoints."""
    
    def __init__(self, url: str, mime_type: str):
        self.url = url
        self.mime_type = mime_type
        self.idx = 0  # Start index for POSTs
        self.next_writer: Optional[asyncio.Queue] = None
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Enter context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the session."""
        await self.close()

    async def start(self):
        """Start the publisher session."""
        if not self.session:
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(connector=connector)

    def stream_idx(self):
        """Get the current stream index URL."""
        return f"{self.url}/{self.idx}"

    async def preconnect(self) -> Optional[asyncio.Queue]:
        """Preconnect to the server by initiating a POST request to the current index."""
        if not self.session:
            await self.start()
            
        url = self.stream_idx()
        logger.info(f"Preconnecting to URL: {url}")
        try:
            # Create a queue for streaming data incrementally
            queue = asyncio.Queue()
            asyncio.create_task(self._run_post(url, queue))
            return queue
        except aiohttp.ClientError as e:
            logger.error(f"Failed to complete POST for {url}: {e}")
            return None

    async def _run_post(self, url: str, queue: asyncio.Queue):
        """Run the POST request with streaming data."""
        try:
            if not self.session:
                return
                
            resp = await self.session.post(
                url,
                headers={'Connection': 'close', 'Content-Type': self.mime_type},
                data=self._stream_data(queue)
            )
            
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Trickle POST failed {url}, status code: {resp.status}, msg: {body}")
        except Exception as e:
            logger.error(f"Trickle POST exception {url} - {e}")

    async def _run_delete(self):
        """Send DELETE request to cleanup the stream."""
        try:
            if self.session:
                await self.session.delete(self.url)
        except Exception:
            logger.error(f"Error sending trickle delete request", exc_info=True)

    async def _stream_data(self, queue: asyncio.Queue):
        """Stream data from the queue for the POST request."""
        while True:
            chunk = await queue.get()
            if chunk is None:  # Stop signal
                break
            yield chunk

    async def next(self):
        """Start or retrieve a pending POST request and preconnect for the next segment."""
        async with self.lock:
            if self.next_writer is None:
                logger.info(f"No pending connection, preconnecting {self.stream_idx()}...")
                self.next_writer = await self.preconnect()

            writer = self.next_writer
            self.next_writer = None

            # Set up the next POST in the background
            asyncio.create_task(self._preconnect_next_segment())

        return SegmentWriter(writer)

    async def _preconnect_next_segment(self):
        """Preconnect to the next POST in the background."""
        logger.info(f"Setting up next connection for {self.stream_idx()}")
        async with self.lock:
            if self.next_writer is not None:
                return
            self.idx += 1  # Increment the index for the next POST
            next_writer = await self.preconnect()
            if next_writer:
                self.next_writer = next_writer

    async def close(self):
        """Close the session when done."""
        logger.info(f"Closing {self.url}")
        async with self.lock:
            if self.next_writer:
                segment = SegmentWriter(self.next_writer)
                await segment.close()
                self.next_writer = None
            if self.session:
                try:
                    await self._run_delete()
                    await self.session.close()
                except Exception:
                    logger.error(f"Error closing trickle publisher", exc_info=True)
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