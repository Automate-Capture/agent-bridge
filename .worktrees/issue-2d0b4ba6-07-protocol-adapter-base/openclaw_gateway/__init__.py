"""
openclaw-gateway: Lightweight agent-to-agent message translation gateway.

This module provides the core classes and CLI entry point for the openclaw-gateway package.
It enables seamless communication between heterogeneous autonomous agents by normalizing
messages to a canonical format and handling protocol translation.

Public API:
- Gateway: Main gateway orchestrator class
- LangChainAdapter: Adapter for LangChain agent protocol
- AutoGPTAdapter: Adapter for AutoGPT agent protocol
- EventStreamAdapter: Adapter for event-streaming agents
- CanonicalMessage: Standard message format for inter-agent communication
- Error classes: OpenclawError, AdapterError, RoutingError, CycleDetectedError, TimeoutError, ValidationError, DuplicateMessageError
"""

__version__ = "0.1.0"

# Import error classes (always available - defined in this package)
from openclaw_gateway.errors import (
    AdapterError,
    CycleDetectedError,
    DuplicateMessageError,
    OpenclawError,
    RoutingError,
    TimeoutError,
    ValidationError,
)

# Lazy imports - these will be available once sibling issues are implemented
try:
    from openclaw_gateway.gateway import Gateway
except ImportError:
    Gateway = None  # type: ignore

try:
    from openclaw_gateway.adapters.langchain import LangChainAdapter
except ImportError:
    LangChainAdapter = None  # type: ignore

try:
    from openclaw_gateway.adapters.autogpt import AutoGPTAdapter
except ImportError:
    AutoGPTAdapter = None  # type: ignore

try:
    from openclaw_gateway.adapters.event_stream import EventStreamAdapter
except ImportError:
    EventStreamAdapter = None  # type: ignore

try:
    from openclaw_gateway.canonical_message import CanonicalMessage
except ImportError:
    CanonicalMessage = None  # type: ignore

__all__ = [
    "__version__",
    "OpenclawError",
    "AdapterError",
    "RoutingError",
    "CycleDetectedError",
    "TimeoutError",
    "ValidationError",
    "DuplicateMessageError",
    "Gateway",
    "LangChainAdapter",
    "AutoGPTAdapter",
    "EventStreamAdapter",
    "CanonicalMessage",
]
