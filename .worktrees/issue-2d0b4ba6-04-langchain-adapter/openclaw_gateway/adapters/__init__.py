"""
openclaw_gateway.adapters: Protocol adapter implementations for various agent types.

This package contains protocol adapters that translate between agent-specific message
formats and the canonical message format used internally by the gateway.
"""

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.adapters.langchain import LangChainAdapter

__all__ = ["ProtocolAdapter", "LangChainAdapter"]
