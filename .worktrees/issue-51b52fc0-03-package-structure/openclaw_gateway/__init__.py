"""openclaw-gateway: Agent-to-agent message translation gateway.

This package provides a lightweight message translation gateway for coordinating
autonomous agents with incompatible protocols. The gateway translates messages
between agent-specific schemas and a canonical intermediate format, preserving
context through multi-hop delegations and automatically recovering from failures.

Public API exports:
- CanonicalMessage: Standard message format for all inter-agent communication
- OpenclawError, AdapterError, RoutingError, etc.: Exception hierarchy
- Gateway: Main orchestrator for message routing and state management
- ProtocolAdapter: Abstract base class for protocol adapters
- LangChainAdapter, AutoGPTAdapter, EventStreamAdapter: Built-in adapters
"""

__version__ = "0.1.0"

# Import public types from existing modules
from openclaw_gateway.canonical_message import (
    CanonicalMessage,
    CRITICAL_FIELDS_BY_INTENT,
)
from openclaw_gateway.errors import (
    OpenclawError,
    AdapterError,
    RoutingError,
    CycleDetectedError,
    TimeoutError,
    ValidationError,
    DuplicateMessageError,
)

# Import classes from local modules (stubs will be replaced by sibling issues)
from openclaw_gateway.gateway import Gateway
from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.adapters.langchain import LangChainAdapter
from openclaw_gateway.adapters.autogpt import AutoGPTAdapter
from openclaw_gateway.adapters.event_stream import EventStreamAdapter


__all__ = [
    "__version__",
    # From errors.py
    "OpenclawError",
    "AdapterError",
    "RoutingError",
    "CycleDetectedError",
    "TimeoutError",
    "ValidationError",
    "DuplicateMessageError",
    # From canonical_message.py
    "CanonicalMessage",
    "CRITICAL_FIELDS_BY_INTENT",
    # From gateway.py
    "Gateway",
    # From adapters
    "ProtocolAdapter",
    "LangChainAdapter",
    "AutoGPTAdapter",
    "EventStreamAdapter",
]
