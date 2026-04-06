"""
openclaw_gateway.adapters: Protocol adapter implementations for various agent types.

This package contains protocol adapters that translate between agent-specific message
formats and the canonical message format used internally by the gateway.
"""

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.adapters.event_stream import EventStreamAdapter

__all__ = ["ProtocolAdapter", "EventStreamAdapter"]
