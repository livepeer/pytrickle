"""
Trickle Publisher for sending streaming data over HTTP.

Handles the publication of video/audio segments to trickle endpoints
with automatic reconnection and error handling.
"""

import asyncio
import aiohttp
import logging
from typing import Optional

from .exceptions import (
    TrickleConnectionError, 
    TrickleStreamClosedError, 
    TrickleMaxRetriesError,
    ErrorPropagator
)
from .retry_utils import (
    RetryConfig, 
    retry_async, 
    REQUIRED_CHANNEL_RETRY_CONFIG,
    OPTIONAL_CHANNEL_RETRY_CONFIG
)

logger = logging.getLogger(__name__)

class TricklePublisher:
    """Publisher for streaming data to trickle endpoints."""
    
    def __init__(
        self, 
        url: str, 
        mime_type: str, 
        max_retries: int = 3,
        is_optional: bool = False,
        error_propagator: Optional[ErrorPropagator] = None
    ):
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
            
        self.url = url
        self.mime_type = mime_type
        self.idx = 0  # Start index for POSTs
        self.next_writer: Optional[asyncio.Queue] = None
        self.lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_optional = is_optional
        self.error_propagator = error_propagator
        self.errored = False
        
        # Configure retry behavior based on channel type
        if is_optional:
            self.retry_config = OPTIONAL_CHANNEL_RETRY_CONFIG
        else:
            self.retry_config = REQUIRED_CHANNEL_RETRY_CONFIG
            
        # Override max_retries if provided
        if max_retries != 3:
            self.retry_config = RetryConfig(
                max_retries=max_retries,
                initial_delay=self.retry_config.initial_delay,
                max_delay=self.retry_config.max_delay,
                backoff_factor=self.retry_config.backoff_factor,
                jitter=self.retry_config.jitter
            )

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
            timeout = aiohttp.ClientTimeout(total=30.0)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    def stream_idx(self):
        """Get the current stream index URL."""
        return f"{self.url}/{self.idx}"

    async def preconnect(self) -> Optional[asyncio.Queue]:
        """Preconnect to the server by initiating a POST request to the current index."""
        if not self.session:
            await self.start()
            
        url = self.stream_idx()
        
        async def _attempt_connection():
            try:
                if not self.session:
                    raise TrickleConnectionError("Session not initialized", url, retryable=True)
                    
                logger.debug(f"Trickle publisher preconnecting to: {url}")
                
                # Create a queue for streaming data incrementally
                queue = asyncio.Queue()
                asyncio.create_task(self._run_post(url, queue))
                return queue
                
            except aiohttp.ClientError as e:
                raise TrickleConnectionError(f"Connection failed: {str(e)}", url, retryable=True)
            except asyncio.TimeoutError as e:
                raise TrickleConnectionError(f"Connection timeout: {str(e)}", url, retryable=True)
            except Exception as e:
                # For unknown errors, make them retryable unless they're already TrickleExceptions
                if not isinstance(e, (TrickleConnectionError, TrickleStreamClosedError)):
                    raise TrickleConnectionError(f"Unexpected error: {str(e)}", url, retryable=True)
                raise
        
        try:
            logger.info(f"Trickle publisher connecting to: {url}")
            queue = await retry_async(_attempt_connection, self.retry_config)
            logger.debug(f"Trickle publisher successfully connected to: {url}")
            return queue
            
        except TrickleMaxRetriesError as e:
            logger.error(f"Trickle publisher max retries exceeded for {url}: {e}")
            self.errored = True
            
            # Propagate error for shutdown if this is a required channel
            if self.error_propagator and not self.is_optional:
                await self.error_propagator.propagate_error(e, f"publisher:{self.url}")
            elif self.is_optional:
                logger.warning(f"Optional publisher channel {self.url} failed, continuing without it")
                
            return None
            
        except Exception as e:
            logger.error(f"Trickle publisher unexpected error for {url}: {e}")
            self.errored = True
            
            # Propagate unexpected errors
            if self.error_propagator:
                await self.error_propagator.propagate_error(e, f"publisher:{self.url}")
                
            return None

    async def _run_post(self, url: str, queue: asyncio.Queue):
        """Run the POST request with streaming data."""
        try:
            if not self.session:
                logger.error(f"Session not available for POST {url}")
                return
                
            resp = await self.session.post(
                url,
                headers={'Connection': 'close', 'Content-Type': self.mime_type},
                data=self._stream_data(queue)
            )
            
            if resp.status == 200:
                logger.debug(f"Trickle POST successful: {url}")
            else:
                try:
                    body = await resp.text()
                    logger.error(f"Trickle POST failed {url}, status code: {resp.status}, msg: {body}")
                except Exception:
                    logger.error(f"Trickle POST failed {url}, status code: {resp.status}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Trickle POST client error {url}: {e}")
            if self.error_propagator:
                await self.error_propagator.propagate_error(
                    TrickleConnectionError(f"POST failed: {str(e)}", url, retryable=True),
                    f"publisher:{self.url}"
                )
        except asyncio.TimeoutError as e:
            logger.error(f"Trickle POST timeout {url}: {e}")
            if self.error_propagator:
                await self.error_propagator.propagate_error(
                    TrickleConnectionError(f"POST timeout: {str(e)}", url, retryable=True),
                    f"publisher:{self.url}"
                )
        except Exception as e:
            logger.error(f"Trickle POST exception {url}: {e}")
            if self.error_propagator:
                await self.error_propagator.propagate_error(e, f"publisher:{self.url}")

    async def _run_delete(self):
        """Send DELETE request to cleanup the stream."""
        if not self.session:
            return
            
        try:
            logger.debug(f"Sending DELETE request to {self.url}")
            await self.session.delete(self.url)
        except Exception as e:
            logger.debug(f"Error sending trickle delete request: {e}")

    async def _stream_data(self, queue: asyncio.Queue):
        """Stream data from the queue for the POST request."""
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:  # Stop signal
                    break
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
            raise

    async def next(self):
        """Start or retrieve a pending POST request and preconnect for the next segment."""
        if self.errored:
            logger.debug(f"Trickle publisher errored for {self.url}")
            return SegmentWriter(None)
            
        async with self.lock:
            if self.next_writer is None:
                logger.debug(f"Trickle publisher no pending connection, preconnecting {self.stream_idx()}...")
                self.next_writer = await self.preconnect()

            writer = self.next_writer
            self.next_writer = None

            # Set up the next POST in the background
            asyncio.create_task(self._preconnect_next_segment())

        return SegmentWriter(writer)

    async def _preconnect_next_segment(self):
        """Preconnect to the next POST in the background."""
        if self.errored:
            return
            
        async with self.lock:
            if self.next_writer is not None:
                return
            self.idx += 1  # Increment the index for the next POST
            logger.debug(f"Trickle publisher setting up next connection for {self.stream_idx()}")
            next_writer = await self.preconnect()
            if next_writer:
                self.next_writer = next_writer

    async def close(self):
        """Close the session when done."""
        logger.info(f"Closing trickle publisher for {self.url}")
        async with self.lock:
            if self.next_writer:
                segment = SegmentWriter(self.next_writer)
                await segment.close()
                self.next_writer = None
            if self.session:
                try:
                    await self._run_delete()
                    await self.session.close()
                except Exception as e:
                    logger.debug(f"Error closing trickle publisher session: {e}")
                finally:
                    self.session = None

class SegmentWriter:
    """Writer for individual trickle segments."""
    
    def __init__(self, queue: Optional[asyncio.Queue]):
        self.queue = queue

    async def write(self, data: bytes):
        """Write data to the current segment."""
        if self.queue:
            try:
                await self.queue.put(data)
            except Exception as e:
                logger.error(f"Error writing to segment: {e}")
                raise

    async def close(self):
        """Ensure the request is properly closed when done."""
        if self.queue:
            try:
                await self.queue.put(None)  # Send None to signal end of data
            except Exception as e:
                logger.debug(f"Error closing segment writer: {e}")

    async def __aenter__(self):
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager and close the connection."""
        await self.close()


def create_publisher(
    url: Optional[str], 
    mime_type: str = "application/octet-stream",
    max_retries: int = 3,
    is_optional: bool = False,
    error_propagator: Optional[ErrorPropagator] = None
) -> Optional[TricklePublisher]:
    """
    Create a TricklePublisher, handling the case where URL is not provided.
    
    Args:
        url: The URL to publish to (None or empty string for optional channels)
        mime_type: MIME type for the published data
        max_retries: Maximum retry attempts
        is_optional: Whether this is an optional channel
        error_propagator: Error propagator for shutdown signaling
        
    Returns:
        TricklePublisher instance or None if URL is not provided
    """
    if not url or not url.strip():
        if is_optional:
            logger.info("Optional publisher URL not provided, skipping publication")
            return None
        else:
            raise ValueError("Required publisher URL cannot be empty")
    
    return TricklePublisher(
        url=url,
        mime_type=mime_type,
        max_retries=max_retries,
        is_optional=is_optional,
        error_propagator=error_propagator
    ) 