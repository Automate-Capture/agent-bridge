"""
Protocol Adapter Base Class

Abstract base class defining the interface contract for all protocol adapters.
All adapters must implement ingest() and egress() methods to translate between
agent-specific formats and the canonical message format.

This module establishes the protocol adapter boundary and enables polymorphic
adapter composition across heterogeneous agents (LangChain, AutoGPT, EventStream, custom).
"""

from abc import ABC, abstractmethod

from openclaw_gateway.canonical_message import CanonicalMessage


class ProtocolAdapter(ABC):
    """
    Abstract base class for protocol adapters.

    All concrete adapters must implement ingest() and egress() to translate
    between agent-specific message formats and the canonical format.

    Attributes:
        _agent_id: Unique identifier for this adapter instance
        _protocol_name: Protocol this adapter handles (langchain, autogpt, event_stream)
    """

    def __init__(self, agent_id: str, protocol_name: str) -> None:
        """
        Initialize the protocol adapter.

        Args:
            agent_id: Unique identifier for this adapter instance
            protocol_name: Protocol this adapter handles (langchain, autogpt, event_stream)
        """
        self._agent_id = agent_id
        self._protocol_name = protocol_name

    @property
    def agent_id(self) -> str:
        """Unique identifier for this adapter instance."""
        return self._agent_id

    @property
    def protocol_name(self) -> str:
        """Protocol this adapter handles."""
        return self._protocol_name

    @abstractmethod
    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """
        Parse agent-specific format → CanonicalMessage.

        Args:
            raw_message: Raw bytes from agent (assumed UTF-8 JSON)

        Returns:
            CanonicalMessage (validated via Pydantic)

        Raises:
            ValueError: If raw_message cannot be parsed or validated
            KeyError: If required fields missing in parsed JSON

        Performance requirement: < 100ms for initialization/parsing
        """
        pass

    @abstractmethod
    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """
        Transform CanonicalMessage → agent-specific format.

        Args:
            canonical: Valid CanonicalMessage instance

        Returns:
            bytes (UTF-8 JSON) suitable for agent consumption

        Raises:
            ValueError: If transformation would lose critical fields

        Performance requirement: < 100ms for transformation
        """
        pass
