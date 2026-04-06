"""AutoGPT protocol adapter.

Implementation details are provided in issue-11-autogpt-adapter.
This is a stub that will be replaced with the actual implementation.
"""

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class AutoGPTAdapter(ProtocolAdapter):
    """Adapter for AutoGPT agents.

    Translates between AutoGPT hierarchical task schema and canonical format.

    This is a stub implementation that will be completed in issue-11-autogpt-adapter.
    """

    def __init__(self, agent_id: str = "autogpt_agent"):
        """Initialize the AutoGPT adapter.

        Args:
            agent_id: Unique identifier for this adapter instance
        """
        self._agent_id = agent_id

    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """Parse AutoGPT format into canonical representation.

        Args:
            raw_message: AutoGPT message bytes

        Returns:
            CanonicalMessage instance

        Raises:
            NotImplementedError: Implementation pending in issue-11
        """
        raise NotImplementedError("AutoGPTAdapter.ingest() to be implemented in issue-11")

    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """Translate canonical representation into AutoGPT format.

        Args:
            canonical: CanonicalMessage instance

        Returns:
            AutoGPT format bytes

        Raises:
            NotImplementedError: Implementation pending in issue-11
        """
        raise NotImplementedError("AutoGPTAdapter.egress() to be implemented in issue-11")

    @property
    def agent_id(self) -> str:
        """Return the adapter instance identifier."""
        return self._agent_id
