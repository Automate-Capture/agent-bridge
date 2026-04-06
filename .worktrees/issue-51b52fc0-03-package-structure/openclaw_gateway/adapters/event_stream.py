"""Event stream protocol adapter.

Implementation details are provided in issue-12-event-stream-adapter.
This is a stub that will be replaced with the actual implementation.
"""

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class EventStreamAdapter(ProtocolAdapter):
    """Adapter for event stream-based agents.

    Translates between event stream format and canonical format.

    This is a stub implementation that will be completed in issue-12-event-stream-adapter.
    """

    def __init__(self, agent_id: str = "event_stream_agent"):
        """Initialize the event stream adapter.

        Args:
            agent_id: Unique identifier for this adapter instance
        """
        self._agent_id = agent_id

    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """Parse event stream format into canonical representation.

        Args:
            raw_message: Event stream message bytes

        Returns:
            CanonicalMessage instance

        Raises:
            NotImplementedError: Implementation pending in issue-12
        """
        raise NotImplementedError("EventStreamAdapter.ingest() to be implemented in issue-12")

    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """Translate canonical representation into event stream format.

        Args:
            canonical: CanonicalMessage instance

        Returns:
            Event stream format bytes

        Raises:
            NotImplementedError: Implementation pending in issue-12
        """
        raise NotImplementedError("EventStreamAdapter.egress() to be implemented in issue-12")

    @property
    def agent_id(self) -> str:
        """Return the adapter instance identifier."""
        return self._agent_id
