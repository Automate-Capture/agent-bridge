"""openclaw-gateway: Agent-to-agent message translation gateway."""

from .negotiation import ProtocolNegotiationManager, AgentCapabilities

__version__ = "0.1.0"

__all__ = [
    "ProtocolNegotiationManager",
    "AgentCapabilities",
]
