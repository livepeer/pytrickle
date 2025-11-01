"""Typed handler registry and metadata utilities.

This module centralizes handler typing and registration to avoid circular
imports between decorators and stream processing components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload

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
]


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
    def get(self, handler_type: Literal["video"]) -> Optional[HandlerFn]: ...

    @overload
    def get(self, handler_type: Literal["audio"]) -> Optional[HandlerFn]: ...

    @overload
    def get(self, handler_type: Literal["model_loader"]) -> Optional[HandlerFn]: ...

    @overload
    def get(self, handler_type: Literal["param_updater"]) -> Optional[HandlerFn]: ...

    @overload
    def get(
        self, handler_type: Literal["stream_start", "stream_stop"]
    ) -> Optional[HandlerFn]: ...

    def get(self, handler_type: HandlerKind) -> Optional[HandlerFn]:
        return self._handlers.get(handler_type)
