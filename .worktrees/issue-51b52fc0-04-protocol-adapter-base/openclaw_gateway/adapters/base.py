"""Protocol adapter base class for translating agent message formats.

This module defines the ProtocolAdapter abstract base class that all protocol
adapters must implement. Each adapter translates between an agent-specific
format and the canonical message format.

The ProtocolAdapter establishes the interface contract that enables:
- O(1) new adapter registration without combinatorial translation code
- Polymorphic dispatch across heterogeneous agent protocols
- Standard error handling and validation across all adapters
"""

from abc import ABC, abstractmethod
from ..canonical_message import CanonicalMessage


class ProtocolAdapter(ABC):
    """Abstract base class for all protocol adapters.

    Adapters translate between agent-specific formats and the canonical message
    format. Each adapter must implement ingest() and egress() methods to handle
    bidirectional translation with lossless round-trip guarantees.

    Properties:
        agent_id: Unique identifier for this adapter instance (read-only)
        protocol_name: Protocol name this adapter handles (read-only)
    """

    def __init__(self, agent_id: str, protocol_name: str):
        """Initialize the protocol adapter.

        Args:
            agent_id: Unique identifier for this adapter instance
            protocol_name: Protocol name this adapter handles (e.g., 'langchain', 'autogpt', 'event_stream')
        """
        self._agent_id = agent_id
        self._protocol_name = protocol_name

    @property
    def agent_id(self) -> str:
        """Unique identifier for this adapter instance.

        Returns:
            The agent_id string provided at initialization.
        """
        return self._agent_id

    @property
    def protocol_name(self) -> str:
        """Protocol name this adapter handles.

        Returns:
            The protocol_name string provided at initialization.
        """
        return self._protocol_name

    @abstractmethod
    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """Parse agent-specific format into canonical message format.

        Translates raw message bytes from an agent (assumed UTF-8 JSON) into
        a validated CanonicalMessage instance. This method is called when
        receiving messages from agents to normalize them for routing.

        Args:
            raw_message: Raw bytes from agent (assumed UTF-8 JSON format)

        Returns:
            CanonicalMessage instance (validated via Pydantic)

        Raises:
            ValueError: If raw_message cannot be parsed or validated
            KeyError: If required fields are missing in parsed JSON
            UnicodeDecodeError: If raw_message is not valid UTF-8

        Performance Requirement:
            Must complete in < 100ms per message (includes parsing and validation)
        """
        pass

    @abstractmethod
    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """Transform canonical message into agent-specific format.

        Translates a CanonicalMessage into the agent-specific format as raw bytes
        (UTF-8 JSON). This method is called when sending messages to agents to
        convert from the canonical format back to agent-native format.

        Args:
            canonical: Valid CanonicalMessage instance to transform

        Returns:
            bytes (UTF-8 JSON) suitable for agent consumption

        Raises:
            ValueError: If transformation would lose critical fields
            TypeError: If canonical is not a CanonicalMessage instance

        Performance Requirement:
            Must complete in < 100ms per message (includes serialization)
        """
        pass
