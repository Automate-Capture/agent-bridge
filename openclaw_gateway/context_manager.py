"""
Conversation Context Manager for persistent state tracking.

This module manages conversation metadata, delegation chains, and state snapshots.
Features:
- Store conversation contexts with LRU eviction (max 1000 conversations)
- Track delegation chains across multi-hop agent delegations
- Persist all changes to Write-Ahead Log (WAL)
- Recover from WAL on startup with zero data loss
- O(1) context lookup latency via OrderedDict
"""

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional

from openclaw_gateway.wal import WAL

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """
    Metadata for a single conversation.

    Tracks delegation chain, state snapshots, and conversation status.
    """

    conversation_id: str
    created_at: float  # unix timestamp
    agents_in_chain: List[str] = field(
        default_factory=list
    )  # [original, intermediate_1, ..., final]
    state_snapshots: Dict[str, Any] = field(
        default_factory=dict
    )  # per-agent state
    status: Literal["active", "completed", "failed", "partially_recovered"] = "active"
    wal_entry_id: int = 0  # sequence number for recovery

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for WAL persistence."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "agents_in_chain": self.agents_in_chain,
            "state_snapshots": self.state_snapshots,
            "status": self.status,
            "wal_entry_id": self.wal_entry_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Deserialize from dict during WAL recovery."""
        return cls(**data)


class ConversationContextManager:
    """
    Manages conversation contexts with LRU eviction and WAL persistence.

    Responsibilities:
    - Create and retrieve conversation contexts
    - Track delegation chains (agents_in_chain)
    - LRU eviction at 1000 conversations
    - WAL persistence and recovery
    """

    def __init__(
        self,
        max_conversations: int = 1000,
        wal_path: str = "/var/lib/openclaw/gateway.wal",
    ):
        """
        Initialize context manager.

        Args:
            max_conversations: LRU capacity (default 1000)
            wal_path: Path to WAL file
        """
        self._contexts: OrderedDict[str, ConversationContext] = OrderedDict()
        self._max_conversations = max_conversations
        self._wal = WAL(wal_path)
        self._next_wal_id = 1

    def create_context(
        self, conversation_id: str, initial_agent: str, created_at: float
    ) -> ConversationContext:
        """
        Create new conversation context.

        Args:
            conversation_id: Conversation identifier
            initial_agent: Initial agent in chain
            created_at: Creation timestamp

        Returns:
            ConversationContext instance

        Side effects:
            - Adds to _contexts dict
            - Triggers LRU eviction if needed
            - Writes to WAL
        """
        context = ConversationContext(
            conversation_id=conversation_id,
            created_at=created_at,
            agents_in_chain=[initial_agent],
            wal_entry_id=self._next_wal_id,
        )

        # Write to WAL before adding to memory
        wal_entry_id = self._wal.write(
            component="context_manager",
            conversation_id=conversation_id,
            operation_type="create_context",
            data=context.to_dict(),
        )
        context.wal_entry_id = wal_entry_id
        self._next_wal_id = wal_entry_id + 1

        self._contexts[conversation_id] = context

        # LRU eviction if over capacity
        if len(self._contexts) > self._max_conversations:
            evicted_id, _ = self._contexts.popitem(last=False)
            logger.info(f"LRU evicted conversation {evicted_id}")

        return context

    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationContext or None if not found

        Side effects:
            - Moves to end of OrderedDict (LRU)

        Performance: O(1) dict lookup + O(1) LRU update
        """
        if conversation_id not in self._contexts:
            return None

        # Move to end (most recently used)
        self._contexts.move_to_end(conversation_id)
        return self._contexts[conversation_id]

    def add_to_chain(self, conversation_id: str, agent_id: str) -> None:
        """
        Add agent to conversation delegation chain.

        Args:
            conversation_id: Conversation identifier
            agent_id: Agent to add

        Side effects:
            - Updates agents_in_chain list
            - Writes to WAL
        """
        if conversation_id not in self._contexts:
            return

        context = self._contexts[conversation_id]
        if agent_id not in context.agents_in_chain:
            context.agents_in_chain.append(agent_id)

            # Write update to WAL
            wal_entry_id = self._wal.write(
                component="context_manager",
                conversation_id=conversation_id,
                operation_type="add_to_chain",
                data=context.to_dict(),
            )
            context.wal_entry_id = wal_entry_id
            self._next_wal_id = wal_entry_id + 1

    def remove_from_chain(self, conversation_id: str, agent_id: str) -> None:
        """
        Remove agent from conversation delegation chain.

        Args:
            conversation_id: Conversation identifier
            agent_id: Agent to remove

        Side effects:
            - Updates agents_in_chain list
            - Writes to WAL
        """
        if conversation_id not in self._contexts:
            return

        context = self._contexts[conversation_id]
        if agent_id in context.agents_in_chain:
            context.agents_in_chain.remove(agent_id)

            # Write update to WAL
            wal_entry_id = self._wal.write(
                component="context_manager",
                conversation_id=conversation_id,
                operation_type="remove_from_chain",
                data=context.to_dict(),
            )
            context.wal_entry_id = wal_entry_id
            self._next_wal_id = wal_entry_id + 1

    def update_status(
        self,
        conversation_id: str,
        status: Literal["active", "completed", "failed", "partially_recovered"],
    ) -> None:
        """
        Update conversation status.

        Args:
            conversation_id: Conversation identifier
            status: New status

        Side effects:
            - Updates status field
            - Writes to WAL
        """
        if conversation_id not in self._contexts:
            return

        context = self._contexts[conversation_id]
        context.status = status

        # Write update to WAL
        wal_entry_id = self._wal.write(
            component="context_manager",
            conversation_id=conversation_id,
            operation_type="update_status",
            data=context.to_dict(),
        )
        context.wal_entry_id = wal_entry_id
        self._next_wal_id = wal_entry_id + 1

    def update_state_snapshot(
        self, conversation_id: str, agent_id: str, snapshot: Dict[str, Any]
    ) -> None:
        """
        Update state snapshot for an agent in a conversation.

        Args:
            conversation_id: Conversation identifier
            agent_id: Agent identifier
            snapshot: State snapshot data

        Side effects:
            - Updates state_snapshots dict
            - Writes to WAL
        """
        if conversation_id not in self._contexts:
            return

        context = self._contexts[conversation_id]
        context.state_snapshots[agent_id] = snapshot

        # Write update to WAL
        wal_entry_id = self._wal.write(
            component="context_manager",
            conversation_id=conversation_id,
            operation_type="update_state_snapshot",
            data=context.to_dict(),
        )
        context.wal_entry_id = wal_entry_id
        self._next_wal_id = wal_entry_id + 1

    async def recover_from_wal(self, wal_entries: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Recover context manager state from WAL.

        Called during gateway startup (after state_coordinator recovery).

        Args:
            wal_entries: List of all WAL entries from gateway.wal file.
                        If None, reads from WAL file directly.

        Recovery logic:
        - Read all context_manager entries from WAL
        - Reconstruct contexts in memory
        - Preserve LRU order based on wal_entry_id
        """
        try:
            if wal_entries is None:
                wal_entries = self._wal.read_all()

            for entry in wal_entries:
                if entry.get("component") != "context_manager":
                    continue

                conversation_id = entry["conversation_id"]
                operation_type = entry["operation_type"]
                data = entry["data"]

                if operation_type == "create_context":
                    context = ConversationContext.from_dict(data)
                    self._contexts[conversation_id] = context
                elif operation_type in (
                    "add_to_chain",
                    "remove_from_chain",
                    "update_status",
                    "update_state_snapshot",
                ):
                    # Update existing context
                    if conversation_id in self._contexts:
                        self._contexts[conversation_id] = ConversationContext.from_dict(
                            data
                        )

                self._next_wal_id = max(self._next_wal_id, entry.get("entry_id", 0) + 1)

        except FileNotFoundError:
            logger.info(f"WAL file not found, starting fresh")

        logger.info(f"Recovered {len(self._contexts)} conversations from WAL")

    def get_all_contexts(self) -> Dict[str, ConversationContext]:
        """
        Get all conversation contexts.

        Returns:
            Dictionary of all conversation contexts
        """
        return dict(self._contexts)

    def get_active_contexts_count(self) -> int:
        """
        Get count of active conversations in memory.

        Returns:
            Number of conversations currently stored
        """
        return len(self._contexts)
