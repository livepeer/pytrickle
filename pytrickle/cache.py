"""
Thread-safe cache for storing values between trickle components.
"""

import threading
from typing import Any, Optional


class LastValueCache:
    """Thread-safe cache for storing the last value."""
    
    def __init__(self):
        self._value: Optional[Any] = None
        self._lock = threading.Lock()
    
    def put(self, value: Any) -> None:
        """Store a value in the cache."""
        with self._lock:
            self._value = value
    
    def get(self) -> Optional[Any]:
        """Retrieve the last stored value."""
        with self._lock:
            return self._value 