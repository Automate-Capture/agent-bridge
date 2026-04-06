"""Error exception hierarchy for openclaw-gateway system.

Provides a complete error exception hierarchy for consistent error handling,
proper exception propagation, and exception-specific recovery routing
throughout the openclaw-gateway system.
"""


class OpenclawError(Exception):
    """Base exception for all openclaw errors."""

    pass


class AdapterError(OpenclawError):
    """Protocol adapter transformation error.

    Raised when a protocol adapter fails to transform a message
    between agent-specific format and canonical format.
    """

    pass


class RoutingError(OpenclawError):
    """Message routing error.

    Raised when message routing fails due to invalid routes,
    missing destinations, or routing configuration issues.
    """

    pass


class CycleDetectedError(RoutingError):
    """Circular delegation detected.

    Raised when a cycle is detected in the conversation delegation graph,
    indicating that an agent would be asked to process a request
    that traces back to itself.
    """

    pass


class TimeoutError(OpenclawError):
    """Agent request timeout.

    Raised when an agent request exceeds its configured timeout threshold
    and does not complete within the expected time window.
    """

    pass


class ValidationError(OpenclawError):
    """Semantic validation error.

    Raised when a message fails semantic validation checks,
    such as missing critical fields or inconsistent intent preservation.
    """

    pass


class DuplicateMessageError(OpenclawError):
    """Duplicate message detected.

    Raised when a duplicate message is detected by the deduplication
    tracker, indicating the message has already been processed.
    """

    pass
