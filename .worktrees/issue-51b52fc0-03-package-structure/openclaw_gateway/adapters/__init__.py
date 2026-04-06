"""Protocol adapters package for openclaw-gateway.

This package contains implementations of protocol adapters that translate
between agent-specific message formats and the canonical intermediate format.

Adapters:
- ProtocolAdapter: Abstract base class for all adapters
- LangChainAdapter: Adapter for LangChain agents
- AutoGPTAdapter: Adapter for AutoGPT agents
- EventStreamAdapter: Adapter for event stream-based agents
"""

from openclaw_gateway.adapters.base import ProtocolAdapter

__all__ = ["ProtocolAdapter"]
