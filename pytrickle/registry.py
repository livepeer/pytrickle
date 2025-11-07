"""Typed handler registry and metadata utilities.

This module centralizes handler typing and registration to avoid circular
imports between decorators and stream processing components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Protocol, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .frames import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)

# Common callable type for raw handler functions
HandlerFn = Callable[..., Any]

# Supported handler kind literals
HandlerKind = Literal[
    "video",
    "audio",
    "model_loader",
    "param_updater",
    "stream_start",
    "stream_stop",
    "warmup",
]

# Protocol definitions for decorated handler signatures
class VideoHandlerProtocol(Protocol):
    """Protocol for decorated video handler functions."""
    async def __call__(self, *args: VideoFrame) -> Optional[VideoFrame]: ...

class AudioHandlerProtocol(Protocol):
    """Protocol for decorated audio handler functions."""
    async def __call__(self, *args: AudioFrame) -> Optional[List[AudioFrame]]: ...

class ModelLoaderProtocol(Protocol):
    """Protocol for decorated model loader functions."""
    async def __call__(self, *args: Any, **kwargs: Any) -> None: ...

class ParamUpdaterProtocol(Protocol):
    """Protocol for decorated parameter updater functions."""
    async def __call__(self, params: Dict[str, Any]) -> None: ...

class LifecycleProtocol(Protocol):
    """Protocol for decorated lifecycle functions (start/stop)."""
    async def __call__(self) -> None: ...    

class WarmupProtocol(Protocol):
    """Protocol for decorated warmup functions."""
    async def __call__(self, **kwargs: Any) -> None: ...

@dataclass
class HandlerInfo:
    """Metadata captured for each registered handler."""

    handler_type: HandlerKind
    description: Optional[str] = None
    signature: Optional[str] = None


class HandlerRegistry:
    """Registry that stores the active handlers for a component.

    Provides typed lookups via overloads so call sites get precise types.
    """

    def __init__(self) -> None:
        self._handlers: Dict[HandlerKind, HandlerFn] = {}
        self._info: Dict[HandlerKind, HandlerInfo] = {}

    def register(self, handler: HandlerFn, info: HandlerInfo) -> None:
        """Register a handler, warning if another handler is replaced."""

        if info.handler_type in self._handlers:
            previous = self._info[info.handler_type]
            logger.warning(
                "Overwriting handler '%s': %s -> %s",
                info.handler_type,
                previous.description,
                info.description,
            )
        self._handlers[info.handler_type] = handler
        self._info[info.handler_type] = info

    # Overloads for typed lookups
    @overload
    def get(self, handler_type: Literal["video"]) -> Optional[VideoHandlerProtocol]: ...

    @overload
    def get(self, handler_type: Literal["audio"]) -> Optional[AudioHandlerProtocol]: ...

    @overload
    def get(self, handler_type: Literal["model_loader"]) -> Optional[ModelLoaderProtocol]: ...

    @overload
    def get(self, handler_type: Literal["param_updater"]) -> Optional[ParamUpdaterProtocol]: ...

    @overload
    def get(
        self, handler_type: Literal["stream_start", "stream_stop"]
    ) -> Optional[LifecycleProtocol]: ...

    def get(self, handler_type: HandlerKind) -> Optional[HandlerFn]:
        return self._handlers.get(handler_type)

