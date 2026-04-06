"""Error exception hierarchy for openclaw-gateway system.

This module defines all error types used throughout the openclaw-gateway system,
enabling consistent error handling, proper exception propagation, and
exception-specific recovery routing.
"""


class OpenclawError(Exception):
    """Base exception for all openclaw errors.

    All specialized error types inherit from this class, allowing callers to catch
    any openclaw-specific error using `except OpenclawError`.
    """
    pass


class AdapterError(OpenclawError):
    """Protocol adapter transformation error.

    Raised when a protocol adapter fails to transform messages between the
    canonical format and agent-specific formats (ingest or egress operations).
    """
    pass


class RoutingError(OpenclawError):
    """Message routing error.

    Raised when a message cannot be routed to its intended destination.
    This is the parent class for routing-specific errors.
    """
    pass


class CycleDetectedError(RoutingError):
    """Circular delegation detected.

    Raised when the router detects a circular delegation pattern in the
    conversation graph (e.g., A→B→C→A). This is a specialized RoutingError.
    """
    pass


class TimeoutError(OpenclawError):
    """Agent request timeout.

    Raised when an agent fails to respond within the configured timeout period.
    This triggers rollback and reroute recovery mechanisms.
    """
    pass


class ValidationError(OpenclawError):
    """Semantic validation error.

    Raised when a message fails semantic validation, such as missing critical
    fields or intent preservation checks.
    """
    pass


class DuplicateMessageError(OpenclawError):
    """Duplicate message detected.

    Raised when a duplicate message is detected by the deduplication tracker,
    indicating the same message was processed multiple times.
    """
    pass
