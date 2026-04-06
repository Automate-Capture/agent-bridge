"""Protocol negotiation manager for agent-to-agent communication.

This module implements protocol negotiation between agents based on their
capabilities, enabling heterogeneous agents to negotiate compatible
communication channels during handshake.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentCapabilities:
    """Agent capability advertisement.

    MVP: Only supported_protocols tracked.
    Phase 2: Add features field for capability-based routing.

    Attributes:
        agent_id: Unique identifier for the agent
        supported_protocols: List of protocols the agent supports (e.g., ["langchain", "event_stream"])
    """

    agent_id: str
    supported_protocols: List[str]
    # features: Set[str]  # Phase 2: ["streaming", "task_decomposition", "batch_processing"]


class ProtocolNegotiationManager:
    """Negotiates protocol between agents based on capabilities.

    Responsibilities:
    - Store agent capabilities
    - Build capability matrix (agent × protocol → compatibility score)
    - Select optimal protocol using greedy matching
    - Fallback to target's first advertised protocol if no match

    Performance: < 50ms for protocol selection
    """

    def __init__(self):
        """Initialize negotiation manager."""
        self._capabilities: Dict[str, AgentCapabilities] = {}
        self._capability_matrix: Dict[str, Dict[str, float]] = {}  # agent_id -> protocol -> score

    def register_capabilities(self, capabilities: AgentCapabilities) -> None:
        """Register agent capabilities.

        Args:
            capabilities: AgentCapabilities instance

        Side effects:
            - Stores capabilities
            - Updates capability matrix
        """
        agent_id = capabilities.agent_id
        self._capabilities[agent_id] = capabilities

        # Update capability matrix
        # For MVP: binary scoring (1.0 if supported, 0.0 otherwise)
        self._capability_matrix[agent_id] = {}
        for protocol in capabilities.supported_protocols:
            self._capability_matrix[agent_id][protocol] = 1.0

        logger.info(f"Registered capabilities for {agent_id}: {capabilities.supported_protocols}")

    def select_protocol(
        self,
        source_agent_id: str,
        target_agent_id: str,
        preferred_protocol: Optional[str] = None
    ) -> str:
        """Select optimal protocol for communication.

        Algorithm:
        1. If preferred_protocol specified and supported by target, use it
        2. Else, find highest-scored common protocol between source and target
        3. Else, fallback to target's first advertised protocol (best-effort)
        4. Log warning if fallback used

        Args:
            source_agent_id: Requesting agent
            target_agent_id: Target agent
            preferred_protocol: Optional preferred protocol

        Returns:
            Selected protocol name

        Raises:
            ValueError: If target agent has no capabilities registered

        Performance: < 50ms for protocol selection
        """
        if target_agent_id not in self._capabilities:
            raise ValueError(f"No capabilities registered for target agent {target_agent_id}")

        target_caps = self._capabilities[target_agent_id]

        # Step 1: Try preferred protocol
        if preferred_protocol and preferred_protocol in target_caps.supported_protocols:
            logger.info(
                f"Using preferred protocol {preferred_protocol} for "
                f"{source_agent_id} → {target_agent_id}"
            )
            return preferred_protocol

        # Step 2: Select highest-scored common protocol
        if source_agent_id in self._capabilities:
            source_caps = self._capabilities[source_agent_id]
            common = set(source_caps.supported_protocols) & set(target_caps.supported_protocols)

            if common:
                # Select protocol with highest score in target's capability matrix
                selected = max(
                    common,
                    key=lambda p: self._capability_matrix[target_agent_id].get(p, 0.0)
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
