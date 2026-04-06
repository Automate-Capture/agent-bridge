"""LangChain protocol adapter.

Implementation details are provided in issue-10-langchain-adapter.
This is a stub that will be replaced with the actual implementation.
"""

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class LangChainAdapter(ProtocolAdapter):
    """Adapter for LangChain agents.

    Translates between LangChain tool-calling schema and canonical format.

    This is a stub implementation that will be completed in issue-10-langchain-adapter.
    """

    def __init__(self, agent_id: str = "langchain_agent"):
        """Initialize the LangChain adapter.

        Args:
            agent_id: Unique identifier for this adapter instance
        """
        self._agent_id = agent_id

    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """Parse LangChain format into canonical representation.

        Args:
            raw_message: LangChain message bytes

        Returns:
            CanonicalMessage instance

        Raises:
            NotImplementedError: Implementation pending in issue-10
        """
        raise NotImplementedError("LangChainAdapter.ingest() to be implemented in issue-10")

    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """Translate canonical representation into LangChain format.

        Args:
            canonical: CanonicalMessage instance

        Returns:
            LangChain format bytes

        Raises:
            NotImplementedError: Implementation pending in issue-10
        """
        raise NotImplementedError("LangChainAdapter.egress() to be implemented in issue-10")

    @property
    def agent_id(self) -> str:
        """Return the adapter instance identifier."""
        return self._agent_id
