"""Base protocol adapter abstract class.

Defines the interface that all protocol adapters must implement to translate
between agent-specific message formats and the canonical intermediate format.
"""

from abc import ABC, abstractmethod
from typing import Optional

from openclaw_gateway.canonical_message import CanonicalMessage


class ProtocolAdapter(ABC):
    """
    Base class for all protocol adapters.

    Adapters translate between agent-specific formats and CanonicalMessage.
    Each adapter must implement ingest() and egress() with lossless round-trip guarantee.
    """

    def __init__(self, agent_id: str, protocol_name: str):
        """
        Initialize adapter.

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
