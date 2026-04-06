"""openclaw-gateway: Agent-to-agent message translation gateway."""

from .deduplication import DeduplicationTracker
from .errors import (
    OpenclawError,
    AdapterError,
    RoutingError,
    CycleDetectedError,
    TimeoutError,
    ValidationError,
    DuplicateMessageError,
)
from .canonical_message import CanonicalMessage

__version__ = "0.1.0"

__all__ = [
    "DeduplicationTracker",
    "OpenclawError",
    "AdapterError",
    "RoutingError",
    "CycleDetectedError",
    "TimeoutError",
    "ValidationError",
    "DuplicateMessageError",
    "CanonicalMessage",
]
