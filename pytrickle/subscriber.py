"""
Trickle Subscriber for receiving streaming data over HTTP.

Handles the subscription to video/audio segments from trickle endpoints
with automatic reconnection and error handling.
"""

import asyncio
import aiohttp
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TrickleSubscriber:
    """Subscriber for receiving streaming data from trickle endpoints."""
    
    def __init__(self, url: str, start_seq: int = -2, max_retries: int = 5):
        self.base_url = url
        self.idx = start_seq
        self.pending_get: Optional[aiohttp.ClientResponse] = None  # Pre-initialized GET request
        self.lock = asyncio.Lock()  # Lock to manage concurrent access
        self.session: Optional[aiohttp.ClientSession] = None
        self.errored = False
        self.max_retries = max_retries

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
                resp = await self.session.get(url, headers={'Connection': 'close'})

                if resp.status == 200:
                    # Return the response for later processing
                    return resp

                if resp.status == 404:
                    logger.info(f"Trickle sub got 404, terminating {url}")
                    resp.release()
                    self.errored = True
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
        self.errored = True
        return None

    async def next(self) -> Optional['Segment']:
        """Retrieve data from the current segment and set up the next segment concurrently."""
        async with self.lock:
            if self.errored:
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
            asyncio.create_task(self._preconnect_next_segment())

        return segment

    async def _preconnect_next_segment(self):
        """Preconnect to the next segment in the background."""
        async with self.lock:
            if self.pending_get is not None:
                return
            next_conn = await self.preconnect()
            if next_conn:
                self.pending_get = next_conn

    async def close(self):
        """Close the session when done."""
        logger.info(f"Closing {self.base_url}")
        async with self.lock:
            if self.pending_get:
                self.pending_get.close()
                self.pending_get = None
            if self.session:
                try:
                    await self.session.close()
                except Exception:
                    logger.error(f"Error closing trickle subscriber", exc_info=True)
                finally:
                    self.session = None

class Segment:
    """Represents a single trickle segment."""
    
    def __init__(self, response: aiohttp.ClientResponse):
        self.response = response

    def seq(self) -> int:
        """Extract the sequence number from the response headers."""
        seq_str = self.response.headers.get('Lp-Trickle-Seq')
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