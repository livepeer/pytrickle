"""
Trickle Subscriber for receiving streaming data over HTTP.

Handles the subscription to video/audio segments from trickle endpoints
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

class TrickleSubscriber:
    """Subscriber for receiving streaming data from trickle endpoints."""
    
    def __init__(
        self, 
        url: str, 
        start_seq: int = -2, 
        max_retries: int = 3,
        is_optional: bool = False,
        error_propagator: Optional[ErrorPropagator] = None
    ):
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
            
        self.base_url = url
        self.idx = start_seq
        self.pending_get: Optional[aiohttp.ClientResponse] = None
        self.lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None
        self.errored = False
        self.is_optional = is_optional
        self.error_propagator = error_propagator
        
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
        """Start the subscriber session."""
        if not self.session:
            connector = aiohttp.TCPConnector(verify_ssl=False)
            timeout = aiohttp.ClientTimeout(total=30.0)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def preconnect(self) -> Optional[aiohttp.ClientResponse]:
        """
        Preconnect to the server by making a GET request to fetch the next segment.
        Uses retry logic based on channel type (optional vs required).
        """
        if not self.session:
            await self.start()
            
        url = f"{self.base_url}/{self.idx}"
        
        async def _attempt_connection():
            nonlocal url  # Declare nonlocal at the top
            try:
                if not self.session:
                    raise TrickleConnectionError("Session not initialized", url, retryable=True)
                resp = await self.session.get(url, headers={'Connection': 'close'})

                if resp.status == 200:
                    return resp

                if resp.status == 404:
                    # Stream not found - this is typically a permanent error
                    await resp.release()
                    raise TrickleStreamClosedError(f"Stream not found (404): {url}", url)

                if resp.status == 470:
                    # Channel exists but no data at this index, reset to latest
                    idx = resp.headers.get('Lp-Trickle-Latest') or '-1'
                    new_url = f"{self.base_url}/{idx}"
                    logger.info(f"Trickle sub resetting index to leading edge: {new_url}")
                    await resp.release()
                    
                    # Update URL and try again immediately
                    url = new_url
                    if self.session:
                        resp = await self.session.get(url, headers={'Connection': 'close'})
                        if resp.status == 200:
                            return resp
                        await resp.release()

                # For other status codes, read the response body and raise error
                try:
                    body = await resp.text()
                    await resp.release()
                    raise TrickleConnectionError(
                        f"HTTP {resp.status}: {body}", 
                        url, 
                        retryable=True
                    )
                except Exception as e:
                    await resp.release()
                    raise TrickleConnectionError(
                        f"HTTP {resp.status}: {str(e)}", 
                        url, 
                        retryable=True
                    )

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
            logger.info(f"Trickle sub connecting to URL: {url}")
            resp = await retry_async(_attempt_connection, self.retry_config)
            logger.debug(f"Trickle sub successfully connected to: {url}")
            return resp
            
        except TrickleMaxRetriesError as e:
            logger.error(f"Trickle sub max retries exceeded for {url}: {e}")
            self.errored = True
            
            # Propagate error for shutdown if this is a required channel
            if self.error_propagator and not self.is_optional:
                await self.error_propagator.propagate_error(e, f"subscriber:{self.base_url}")
            elif self.is_optional:
                logger.warning(f"Optional channel {self.base_url} failed, continuing without it")
                
            return None
            
        except TrickleStreamClosedError as e:
            logger.info(f"Trickle sub stream closed for {url}: {e}")
            self.errored = True
            
            # Even for optional channels, stream closed is a signal that we're done
            if self.error_propagator:
                await self.error_propagator.propagate_error(e, f"subscriber:{self.base_url}")
                
            return None
            
        except Exception as e:
            logger.error(f"Trickle sub unexpected error for {url}: {e}")
            self.errored = True
            
            # Propagate unexpected errors
            if self.error_propagator:
                await self.error_propagator.propagate_error(e, f"subscriber:{self.base_url}")
                
            return None

    async def next(self) -> Optional['Segment']:
        """Retrieve data from the current segment and set up the next segment concurrently."""
        async with self.lock:
            if self.errored:
                logger.debug(f"Trickle subscription errored for {self.base_url}")
                return None

            # If we don't have a pending GET request, preconnect
            if self.pending_get is None:
                logger.debug("Trickle sub no pending connection, preconnecting...")
                self.pending_get = await self.preconnect()

            # Extract the current connection to use for reading
            resp = self.pending_get
            self.pending_get = None

            # Preconnect failed
            if resp is None:
                return None

            # Create segment from response
            segment = Segment(resp)

            if segment.eos():
                logger.info(f"Trickle subscriber reached end of stream for {self.base_url}")
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
        logger.info(f"Closing trickle subscriber for {self.base_url}")
        async with self.lock:
            if self.pending_get:
                if not self.pending_get.closed:
                    self.pending_get.close()
                self.pending_get = None
            if self.session:
                try:
                    await self.session.close()
                except Exception as e:
                    logger.debug(f"Error closing trickle subscriber session: {e}")
                finally:
                    self.session = None

class Segment:
    """Represents a single trickle segment."""
    
    def __init__(self, response: aiohttp.ClientResponse):
        self.response = response

    def seq(self) -> int:
        """Extract the sequence number from the response headers."""
        if not self.response:
            return -1
        seq_str = self.response.headers.get('Lp-Trickle-Seq')
        try:
            seq = int(seq_str) if seq_str else -1
        except (TypeError, ValueError):
            return -1
        return seq

    def eos(self) -> bool:
        """Check if this is the end of stream."""
        if not self.response:
            return True
        return self.response.headers.get('Lp-Trickle-Closed') is not None

    async def read(self, chunk_size: int = 32 * 1024) -> Optional[bytes]:
        """Read the next chunk of the segment."""
        if not self.response or self.response.closed:
            await self.close()
            return None
            
        try:
            chunk = await self.response.content.read(chunk_size)
            if not chunk:
                await self.close()
            return chunk
        except Exception as e:
            logger.debug(f"Error reading segment data: {e}")
            await self.close()
            return None

    async def close(self):
        """Ensure the response is properly closed when done."""
        if self.response is None:
            return
        try:
            if not self.response.closed:
                self.response.close()
        except Exception as e:
            logger.debug(f"Error closing segment response: {e}")
        finally:
            self.response = None


def create_subscriber(
    url: Optional[str], 
    start_seq: int = -2, 
    max_retries: int = 3,
    is_optional: bool = False,
    error_propagator: Optional[ErrorPropagator] = None
) -> Optional[TrickleSubscriber]:
    """
    Create a TrickleSubscriber, handling the case where URL is not provided.
    
    Args:
        url: The URL to subscribe to (None or empty string for optional channels)
        start_seq: Starting sequence number
        max_retries: Maximum retry attempts
        is_optional: Whether this is an optional channel
        error_propagator: Error propagator for shutdown signaling
        
    Returns:
        TrickleSubscriber instance or None if URL is not provided
    """
    if not url or not url.strip():
        if is_optional:
            logger.info("Optional channel URL not provided, skipping subscription")
            return None
        else:
            raise ValueError("Required channel URL cannot be empty")
    
    return TrickleSubscriber(
        url=url,
        start_seq=start_seq,
        max_retries=max_retries,
        is_optional=is_optional,
        error_propagator=error_propagator
    ) 