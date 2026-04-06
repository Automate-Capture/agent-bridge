"""Conversation context manager with LRU eviction and write-ahead logging.

This module implements the ConversationContextManager class for storing conversation
metadata, tracking delegation chains, managing LRU eviction of conversation contexts,
and recovering from write-ahead log (WAL) entries.

Key features:
- ConversationContext dataclass with 6 fields for complete conversation state
- OrderedDict-based LRU cache with 1000 conversation limit
- Automatic eviction of oldest contexts when capacity exceeded
- O(1) context lookup with LRU move_to_end() updates
- WAL recovery for crash resilience with zero data loss
- Agent chain tracking through agents_in_chain list
- State snapshot isolation per agent
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Metadata for a single conversation.

    Tracks delegation chain, state snapshots, and conversation status.

    Attributes:
        conversation_id: Unique identifier for this conversation
        created_at: Unix timestamp (float) of conversation creation
        agents_in_chain: List of agent IDs in delegation order: [original, intermediate_1, ..., final]
        state_snapshots: Dictionary mapping agent IDs to their state snapshots
        status: Current conversation status (active, completed, failed, partially_recovered)
        wal_entry_id: Sequence number for recovery and debugging
    """

    conversation_id: str
    created_at: float  # unix timestamp
    agents_in_chain: List[str]  # [original, intermediate_1, ..., final]
    state_snapshots: Dict[str, Any] = field(default_factory=dict)  # per-agent state
    status: Literal["active", "completed", "failed", "partially_recovered"] = "active"
    wal_entry_id: int = 0  # sequence number for recovery

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for WAL persistence.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "agents_in_chain": self.agents_in_chain,
            "state_snapshots": self.state_snapshots,
            "status": self.status,
            "wal_entry_id": self.wal_entry_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Deserialize from dict during WAL recovery.

        Args:
            data: Dictionary with ConversationContext fields

        Returns:
            Reconstructed ConversationContext instance
        """
        return cls(**data)


class ConversationContextManager:
    """Manages conversation contexts with LRU eviction and WAL persistence.

    This manager maintains an in-memory cache of conversation contexts using an
    OrderedDict for O(1) LRU operations. When capacity is exceeded (> 1000
    conversations), the oldest conversation is automatically evicted.

    Key responsibilities:
    - Create and retrieve conversation contexts
    - Track delegation chains (agents_in_chain)
    - Enforce LRU eviction at 1000 conversations
    - Persist and recover context state from WAL
    - Move accessed contexts to end of OrderedDict (LRU update)

    Attributes:
        _contexts: OrderedDict mapping conversation_id → ConversationContext
        _max_conversations: Maximum number of conversations to keep in memory (default 1000)
        _wal_path: Path to write-ahead log file
        _next_wal_id: Next WAL entry ID for sequence tracking
    """

    def __init__(self, max_conversations: int = 1000, wal_path: str = "/var/lib/openclaw/gateway.wal"):
        """Initialize context manager.

        Args:
            max_conversations: LRU capacity (default 1000)
            wal_path: Path to WAL file for persistence
        """
        self._contexts: OrderedDict[str, ConversationContext] = OrderedDict()
        self._max_conversations = max_conversations
        self._wal_path = wal_path
        self._next_wal_id = 1

    def store_context(
        self,
        conversation_id: str,
        initial_agent: str,
        created_at: float
    ) -> ConversationContext:
        """Store new conversation context.

        Stores a new context with the initial agent in the delegation chain.
        Automatically evicts the oldest context if capacity is exceeded.

        Args:
            conversation_id: Conversation identifier
            initial_agent: Initial agent in the delegation chain
            created_at: Creation timestamp (unix float)

        Returns:
            ConversationContext instance

        Side effects:
            - Adds context to _contexts OrderedDict
            - Evicts oldest context if len(_contexts) > max_conversations
            - Increments _next_wal_id

        Note:
            WAL entry written by caller (Gateway._write_wal_entry)
        """
        context = ConversationContext(
            conversation_id=conversation_id,
            created_at=created_at,
            agents_in_chain=[initial_agent],
            wal_entry_id=self._next_wal_id
        )
        self._next_wal_id += 1

        self._contexts[conversation_id] = context

        # LRU eviction: if over capacity, remove oldest
        if len(self._contexts) > self._max_conversations:
            evicted_id, _ = self._contexts.popitem(last=False)
            logger.info(f"LRU evicted conversation {evicted_id}")

        return context

    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context by ID.

        Retrieves a context and updates LRU state by moving it to the end
        of the OrderedDict (mark as most recently used).

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationContext if found, None otherwise

        Side effects:
            - Moves conversation to end of OrderedDict (LRU update)

        Performance:
            - O(1) dict lookup + O(1) move_to_end() operation
        """
        if conversation_id not in self._contexts:
            return None

        # Move to end (most recently used) for LRU tracking
        self._contexts.move_to_end(conversation_id)
        return self._contexts[conversation_id]

    def add_agent_to_chain(self, conversation_id: str, agent_id: str) -> None:
        """Add agent to conversation delegation chain.

        Appends an agent to the agents_in_chain list if not already present.
        Increments wal_entry_id for change tracking.

        Args:
            conversation_id: Conversation identifier
            agent_id: Agent to add to the chain

        Side effects:
            - Appends agent_id to agents_in_chain if not already present
            - Increments wal_entry_id and _next_wal_id

        Note:
            WAL entry written by caller (Gateway._write_wal_entry)
        """
        if conversation_id not in self._contexts:
            return

        context = self._contexts[conversation_id]
        if agent_id not in context.agents_in_chain:
            context.agents_in_chain.append(agent_id)
            context.wal_entry_id = self._next_wal_id
            self._next_wal_id += 1

    def update_status(
        self,
        conversation_id: str,
        status: Literal["active", "completed", "failed", "partially_recovered"]
    ) -> None:
        """Update conversation status.

        Changes the status field of an existing conversation context.
        Increments wal_entry_id for change tracking.

        Args:
            conversation_id: Conversation identifier
            status: New status value

        Side effects:
            - Updates context.status
            - Increments wal_entry_id and _next_wal_id

        Note:
            WAL entry written by caller (Gateway._write_wal_entry)
        """
        if conversation_id not in self._contexts:
            return

        context = self._contexts[conversation_id]
        context.status = status
        context.wal_entry_id = self._next_wal_id
        self._next_wal_id += 1

    async def recover_from_wal(self, wal_entries: list) -> None:
        """Recover context manager state from WAL.

        Reconstructs all conversation contexts from WAL entries.
        Called during gateway startup as part of sequential WAL recovery.

        Args:
            wal_entries: List of all WAL entries from gateway.wal file

        Recovery algorithm:
        1. Filter entries with component == "context_manager"
        2. For each create_context entry, add context to _contexts dict
        3. For each add_to_chain or update_status entry, update existing context
        4. Track maximum wal_entry_id to restore _next_wal_id sequence

        Side effects:
            - Rebuilds _contexts OrderedDict from WAL
            - Restores _next_wal_id from maximum entry_id

        Note:
            Called by Gateway.start() after state_coordinator recovery.
            Preserves all 6 fields: conversation_id, created_at, agents_in_chain,
            state_snapshots, status, wal_entry_id
        """
        try:
            for entry in wal_entries:
                if entry.get("component") != "context_manager":
                    continue

                conversation_id = entry.get("conversation_id")
                operation_type = entry.get("operation_type")
                data = entry.get("data", {})

                if operation_type == "create_context":
                    # Create new context from WAL data
                    context = ConversationContext.from_dict(data)
                    self._contexts[conversation_id] = context
                elif operation_type in ("add_to_chain", "update_status"):
                    # Update existing context from WAL data
                    if conversation_id in self._contexts:
                        context = ConversationContext.from_dict(data)
                        self._contexts[conversation_id] = context

                # Track maximum wal_entry_id for sequence restoration
                entry_id = entry.get("entry_id", 0)
                self._next_wal_id = max(self._next_wal_id, entry_id + 1)

        except FileNotFoundError:
            logger.info(f"WAL file {self._wal_path} not found, starting fresh")

        logger.info(f"Recovered {len(self._contexts)} conversations from WAL")
