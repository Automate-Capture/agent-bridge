"""
ProtocolAdapter: Abstract base class defining the interface contract for protocol adapters.

This module defines the ProtocolAdapter ABC that all protocol adapters must inherit from.
Each adapter implementation (LangChain, AutoGPT, EventStream, custom) normalizes agent-specific
message formats to the canonical message format, enabling O(1) integration of new agent types
without combinatorial translation code.

Key responsibilities:
- ingest(raw_message: bytes) -> CanonicalMessage: Parse agent-specific format to canonical
- egress(canonical: CanonicalMessage) -> bytes: Canonical format to agent-specific output
- agent_id: Property returning the unique adapter instance identifier
- protocol_name: Property returning the protocol name (set at initialization)

Performance requirement: < 100ms per ingest/egress operation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openclaw_gateway.canonical_message import CanonicalMessage


class ProtocolAdapter(ABC):
    """
    Abstract base class defining the interface contract for protocol adapters.

    All protocol adapters must inherit from this class and implement the abstract
    methods and properties defined here. This ensures that all adapters follow
    the same interface contract for translating between agent-specific formats
    and the canonical message format.

    Attributes:
        protocol_name (str): The protocol name (e.g., "langchain", "autogpt")
                            Set during initialization and immutable.
    """

    def __init__(self, agent_id: str, protocol_name: str) -> None:
        """
        Initialize the protocol adapter.

        Args:
            agent_id: Unique identifier for this adapter instance. Used to track
                     which agent this adapter is translating for.
            protocol_name: Name of the protocol this adapter handles (e.g., "langchain",
                          "autogpt", "event_stream"). Used for routing and negotiation.

        Raises:
            ValueError: If agent_id or protocol_name are empty strings.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not protocol_name:
            raise ValueError("protocol_name cannot be empty")

        self._agent_id = agent_id
        self._protocol_name = protocol_name

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """
        Unique identifier for this adapter instance.

        Returns:
            str: The agent_id passed to __init__. This is a read-only property
                 that identifies which agent this adapter is translating for.
        """
        pass

    @property
    def protocol_name(self) -> str:
        """
        Protocol name this adapter handles.

        Returns:
            str: The protocol_name passed to __init__ (e.g., "langchain", "autogpt").
                 This is immutable after initialization.
        """
        return self._protocol_name

    @abstractmethod
    async def ingest(self, raw_message: bytes) -> "CanonicalMessage":
        """
        Parse agent-specific message format to canonical representation.

        This method handles the translation from the agent's native protocol format
        (which may be JSON, protobuf, or any binary format) to the standardized
        CanonicalMessage format used internally by the gateway.

        Performance requirement: Must complete in < 100ms for typical messages.

        Args:
            raw_message: Raw bytes from the agent in the agent's native format.
                        The format depends on the specific protocol adapter implementation
                        (e.g., JSON for LangChain, JSON for AutoGPT, event objects for streams).

        Returns:
            CanonicalMessage: Standardized message format with all required fields:
                            - message_id: UUID-v4
                            - source_agent_id: Agent originating the message
                            - conversation_id: Hierarchical conversation ID
                            - message_type: "request", "response", or "state_update"
                            - intent: "analyze", "delegate", or "stream_result"
                            - payload: Message-specific data
                            - metadata: Protocol metadata (source, timestamps, chain)

        Raises:
            ValueError: If raw_message cannot be parsed or is malformed.
            TypeError: If raw_message is not bytes.
        """
        pass

    @abstractmethod
    async def egress(self, canonical: "CanonicalMessage") -> bytes:
        """
        Translate canonical message format to agent-specific output format.

        This method handles the reverse translation from the standardized CanonicalMessage
        format back to the agent's native protocol format for delivery.

        Performance requirement: Must complete in < 100ms for typical messages.

        Args:
            canonical: CanonicalMessage instance containing:
                      - message_id: UUID-v4
                      - source_agent_id: Agent originating the message
                      - conversation_id: Hierarchical conversation ID
                      - message_type: "request", "response", or "state_update"
                      - intent: "analyze", "delegate", or "stream_result"
                      - payload: Message-specific data
                      - metadata: Protocol metadata (source, timestamps, chain)

        Returns:
            bytes: Raw bytes in the agent's native format, ready for transmission to the agent.

        Raises:
            ValueError: If canonical cannot be serialized to the agent's format.
            TypeError: If canonical is not a CanonicalMessage instance.
        """
        pass
