"""Base protocol adapter abstract class.

This module defines the ProtocolAdapter abstract base class that all protocol
adapters must implement. Concrete adapters will inherit from this class and
provide ingest and egress methods for their specific protocol.

Implementation details are provided in issue-04-protocol-adapter-base.
"""

from abc import ABC, abstractmethod
from openclaw_gateway.canonical_message import CanonicalMessage


class ProtocolAdapter(ABC):
    """Abstract base class for protocol adapters.

    Protocol adapters translate between agent-specific message formats and
    the canonical intermediate format. Each adapter handles ingest (agent-specific
    format → canonical) and egress (canonical → agent-specific format) operations.

    This is a stub that will be fully implemented in issue-04-protocol-adapter-base.
    """

    @abstractmethod
    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """Parse agent-specific format into canonical representation.

        Args:
            raw_message: Raw message bytes in the agent-specific format

        Returns:
            CanonicalMessage instance

        Raises:
            AdapterError: If parsing or validation fails
        """
        pass

    @abstractmethod
    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """Translate canonical representation into agent-specific format.

        Args:
            canonical: CanonicalMessage instance to translate

        Returns:
            Raw message bytes in the agent-specific format

        Raises:
            AdapterError: If translation fails
        """
        pass

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Return the unique identifier for this adapter instance.

        Returns:
            Unique adapter instance identifier
        """
        pass
