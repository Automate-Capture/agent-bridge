"""Protocol adapters module for openclaw-gateway.

This module provides the ProtocolAdapter abstract base class that all adapters
must implement, along with concrete adapter implementations for specific agent
protocols (LangChain, AutoGPT, EventStream, etc.).

Public API:
    ProtocolAdapter: Abstract base class for all protocol adapters
"""

from .base import ProtocolAdapter

__all__ = ["ProtocolAdapter"]
