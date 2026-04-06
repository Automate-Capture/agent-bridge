"""Protocol negotiation manager for agent-to-agent communication.

Implements automatic protocol selection between heterogeneous agents based on
declared capabilities. Uses a capability matrix (agent_id × protocol → score)
to select optimal communication protocols with intelligent fallback.

Classes:
    AgentCapabilities: Dataclass for agent protocol capabilities
    ProtocolNegotiationManager: Manager for protocol negotiation and selection
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentCapabilities:
    """Agent capability advertisement.

    Tracks which communication protocols an agent supports.

    MVP: Only supported_protocols tracked.
    Phase 2: Add features field for capability-based routing.

    Attributes:
        agent_id: Unique identifier for the agent
        supported_protocols: List of protocol names the agent supports
                            (e.g., ["langchain", "event_stream"])
    """

    agent_id: str
    supported_protocols: List[str]


class ProtocolNegotiationManager:
    """Negotiates optimal protocol between agents based on capabilities.

    Responsibilities:
    - Store agent capabilities
    - Build capability matrix (agent × protocol → compatibility score)
    - Select optimal protocol using greedy matching
    - Fallback to target's first advertised protocol if no match

    Algorithm for select_protocol():
    1. If preferred_protocol specified and supported by target, use it
    2. Else, find highest-scored common protocol between source and target
    3. Else, fallback to target's first advertised protocol (best-effort)
    4. Raise ValueError if target agent not registered

    Performance: < 50ms per protocol selection
    """

    def __init__(self):
        """Initialize negotiation manager.

        Sets up empty capability tracking structures.
        """
        self._capabilities: Dict[str, AgentCapabilities] = {}
        # agent_id -> protocol -> score (binary: 1.0 if supported, 0.0 otherwise)
        self._capability_matrix: Dict[str, Dict[str, float]] = {}

    def register_capabilities(self, capabilities: AgentCapabilities) -> None:
        """Register agent capabilities.

        Updates both the capabilities store and the capability matrix.
        For MVP: binary scoring (1.0 if supported, 0.0 otherwise).

        Args:
            capabilities: AgentCapabilities instance with agent_id and
                         supported_protocols list

        Side effects:
            - Stores capabilities in _capabilities dict
            - Updates _capability_matrix with binary scores (1.0 per protocol)
        """
        agent_id = capabilities.agent_id
        self._capabilities[agent_id] = capabilities

        # Update capability matrix: 1.0 for each supported protocol
        self._capability_matrix[agent_id] = {}
        for protocol in capabilities.supported_protocols:
            self._capability_matrix[agent_id][protocol] = 1.0

        logger.info(
            f"Registered capabilities for {agent_id}: {capabilities.supported_protocols}"
        )

    def select_protocol(
        self,
        source_agent_id: str,
        target_agent_id: str,
        preferred_protocol: Optional[str] = None,
    ) -> str:
        """Select optimal protocol for communication between agents.

        Algorithm:
        1. If preferred_protocol specified and supported by target, use it
        2. Else, find highest-scored common protocol between source and target
        3. Else, fallback to target's first advertised protocol (best-effort)
        4. Log warning if fallback used

        Args:
            source_agent_id: Requesting/source agent ID
            target_agent_id: Target/destination agent ID
            preferred_protocol: Optional preferred protocol name

        Returns:
            Selected protocol name (str)

        Raises:
            ValueError: If target_agent_id has no registered capabilities

        Performance requirement: < 50ms for protocol selection
        """
        # Validate target agent is registered
        if target_agent_id not in self._capabilities:
            raise ValueError(
                f"No capabilities registered for target agent {target_agent_id}"
            )

        target_caps = self._capabilities[target_agent_id]

        # Step 1: Try preferred protocol
        if (
            preferred_protocol
            and preferred_protocol in target_caps.supported_protocols
        ):
            logger.info(
                f"Using preferred protocol {preferred_protocol} for "
                f"{source_agent_id} → {target_agent_id}"
            )
            return preferred_protocol

        # Step 2: Select highest-scored common protocol
        if source_agent_id in self._capabilities:
            source_caps = self._capabilities[source_agent_id]
            common = set(source_caps.supported_protocols) & set(
                target_caps.supported_protocols
            )

            if common:
                # Select protocol with highest score in target's capability matrix
                selected = max(
                    common,
                    key=lambda p: self._capability_matrix[target_agent_id].get(p, 0.0),
                )
                logger.info(
                    f"Selected common protocol {selected} for "
                    f"{source_agent_id} → {target_agent_id}"
                )
                return selected

        # Step 3: Fallback to target's first advertised protocol
        fallback = target_caps.supported_protocols[0]
        logger.warning(
            f"No common protocol for {source_agent_id} → {target_agent_id}, "
            f"using fallback: {fallback}"
        )
        return fallback

    def get_compatibility_score(self, agent_id: str, protocol: str) -> float:
        """Get compatibility score for an agent-protocol pair.

        Returns the score from the capability matrix, indicating how well
        the agent supports the specified protocol.

        Args:
            agent_id: Agent identifier
            protocol: Protocol name

        Returns:
            Compatibility score (float). For MVP, returns 1.0 if supported,
            0.0 otherwise.
        """
        if agent_id not in self._capability_matrix:
            return 0.0
        return self._capability_matrix[agent_id].get(protocol, 0.0)
