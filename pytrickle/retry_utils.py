"""
Retry utilities for trickle streaming.

Provides exponential backoff, jitter, and retry mechanisms for robust
error handling in trickle streaming applications.
"""

import asyncio
import random
import time
import logging
from typing import Optional, Callable, Any, Union
from .exceptions import TrickleException, TrickleMaxRetriesError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on_exceptions: tuple = (Exception,),
        stop_on_exceptions: tuple = ()
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions
        self.stop_on_exceptions = stop_on_exceptions


class RetryState:
    """State tracking for retry operations."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempt_count = 0
        self.last_error: Optional[Exception] = None
        self.start_time = time.time()
        self.total_delay = 0.0
        
    def should_retry(self, error: Exception) -> bool:
        """Check if we should retry based on the error and attempt count."""
        # Check if we've exceeded max retries
        if self.attempt_count >= self.config.max_retries:
            return False
            
        # Check if this is a stop-on exception
        if isinstance(error, self.config.stop_on_exceptions):
            return False
            
        # Check if this is a TrickleException and not retryable
        if isinstance(error, TrickleException) and not error.retryable:
            return False
            
        # Check if this is a retryable exception
        if isinstance(error, self.config.retry_on_exceptions):
            return True
            
        return False
        
    def get_delay(self) -> float:
        """Calculate the delay for the next retry with exponential backoff and jitter."""
        if self.attempt_count == 0:
            return 0.0
            
        # Calculate exponential backoff
        delay = self.config.initial_delay * (self.config.backoff_factor ** (self.attempt_count - 1))
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
            
        return delay
        
    def next_attempt(self, error: Exception):
        """Move to the next retry attempt."""
        self.attempt_count += 1
        self.last_error = error
        
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since first attempt."""
        return time.time() - self.start_time


async def retry_async(
    func: Callable, 
    config: RetryConfig, 
    *args, 
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: The async function to retry
        config: Retry configuration
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the successful function call
        
    Raises:
        TrickleMaxRetriesError: If max retries are exceeded
        Exception: The last exception if it's not retryable
    """
    state = RetryState(config)
    
    while True:
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Log successful retry if this isn't the first attempt
            if state.attempt_count > 0:
                logger.info(f"Function {func.__name__} succeeded after {state.attempt_count} retries")
                
            return result
            
        except Exception as e:
            state.next_attempt(e)
            
            # Check if we should retry
            if not state.should_retry(e):
                if state.attempt_count >= config.max_retries:
                    logger.error(f"Function {func.__name__} failed after {state.attempt_count} attempts")
                    raise TrickleMaxRetriesError(
                        f"Max retries ({config.max_retries}) exceeded for {func.__name__}",
                        config.max_retries,
                        e
                    )
                else:
                    # Non-retryable error
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                    raise
            
            # Calculate delay and wait
            delay = state.get_delay()
            if delay > 0:
                logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {state.attempt_count}/{config.max_retries})")
                await asyncio.sleep(delay)
                state.total_delay += delay


def retry_sync(
    func: Callable, 
    config: RetryConfig, 
    *args, 
    **kwargs
) -> Any:
    """
    Retry a sync function with exponential backoff.
    
    Args:
        func: The sync function to retry
        config: Retry configuration
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the successful function call
        
    Raises:
        TrickleMaxRetriesError: If max retries are exceeded
        Exception: The last exception if it's not retryable
    """
    state = RetryState(config)
    
    while True:
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful retry if this isn't the first attempt
            if state.attempt_count > 0:
                logger.info(f"Function {func.__name__} succeeded after {state.attempt_count} retries")
                
            return result
            
        except Exception as e:
            state.next_attempt(e)
            
            # Check if we should retry
            if not state.should_retry(e):
                if state.attempt_count >= config.max_retries:
                    logger.error(f"Function {func.__name__} failed after {state.attempt_count} attempts")
                    raise TrickleMaxRetriesError(
                        f"Max retries ({config.max_retries}) exceeded for {func.__name__}",
                        config.max_retries,
                        e
                    )
                else:
                    # Non-retryable error
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                    raise
            
            # Calculate delay and wait
            delay = state.get_delay()
            if delay > 0:
                logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {state.attempt_count}/{config.max_retries})")
                time.sleep(delay)
                state.total_delay += delay


# Pre-configured retry configs for common scenarios
NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=0.5,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True
)

OPTIONAL_CHANNEL_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    initial_delay=1.0,
    max_delay=5.0,
    backoff_factor=1.5,
    jitter=True
)

REQUIRED_CHANNEL_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True
) 