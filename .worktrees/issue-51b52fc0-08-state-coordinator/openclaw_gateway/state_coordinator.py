"""Distributed state coordinator with vector clocks and Last-Write-Wins conflict resolution.

This module provides the StateCoordinator class for managing per-conversation state with:
- Vector clock tracking for causal ordering
- Last-Write-Wins (LWW) conflict resolution based on timestamp
- State subscription callbacks for reactive updates
- Write-Ahead Log (WAL) based persistence and recovery

The StateCoordinator maintains a separate state store per conversation, with each
state entry containing:
- The actual value
- A vector clock (dict mapping agent_id to logical timestamps)
- A UTC timestamp for LWW conflict resolution
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StateEntry:
    """Single state entry with vector clock metadata.

    Attributes:
        value: The actual state value
        vector_clock: Dict mapping agent_id to logical timestamp (for causal ordering)
        timestamp_utc: Unix timestamp (float) for Last-Write-Wins conflict resolution
    """
    value: Any
    vector_clock: Dict[str, int]  # agent_id -> logical timestamp
    timestamp_utc: float  # unix timestamp for LWW


class StateCoordinator:
    """
    Manages distributed state with vector clocks and LWW conflict resolution.

    Responsibilities:
    - Per-conversation key-value state store (in-memory dict-based)
    - Vector clock tracking for causal ordering and concurrent update detection
    - Last-Write-Wins conflict resolution using timestamp comparison
    - State subscriptions and callbacks for reactive updates
    - WAL persistence and recovery

    Performance characteristics:
    - O(1) get() and set() operations (dict lookup)
    - State lookup latency: < 5ms per read from in-memory store
    - Subscription callback execution: O(n) where n = number of subscribers

    Example:
        >>> coordinator = StateCoordinator()
        >>> coordinator.set("conv_1", "key_1", "value_1", "agent_a", {"agent_a": 1})
        >>> value = coordinator.get("conv_1", "key_1")
        >>> assert value == "value_1"
        >>> coordinator.subscribe("conv_1", lambda k, v: print(f"{k}={v}"))
        >>> coordinator.set("conv_1", "key_2", "value_2", "agent_b", {"agent_a": 1, "agent_b": 1})
        # Output: key_2=value_2
    """

    def __init__(self, wal_path: str = "/var/lib/openclaw/gateway.wal"):
        """
        Initialize state coordinator.

        Args:
            wal_path: Path to WAL file (used for recovery, not for direct writes)
        """
        # State store: conversation_id -> {key -> StateEntry}
        self._state: Dict[str, Dict[str, StateEntry]] = defaultdict(dict)

        # Subscriptions: conversation_id -> list(callbacks)
        # Each callback is Callable[[str, Any], None] - (key, value)
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)

        self._wal_path = wal_path
        self._next_wal_id = 1

    def set(
        self,
        conversation_id: str,
        key: str,
        value: Any,
        agent_id: str,
        vector_clock: Dict[str, int]
    ) -> None:
        """
        Set state value with vector clock and timestamp.

        This method:
        1. Increments the agent's logical timestamp in the vector clock
        2. Creates a StateEntry with the new value and updated clock
        3. Performs Last-Write-Wins conflict detection
        4. Updates the in-memory state store
        5. Fires all subscription callbacks

        Args:
            conversation_id: Conversation ID (unique key for state isolation)
            key: State key (within the conversation)
            value: State value (can be any JSON-serializable type)
            agent_id: Agent making update (used for vector clock increment)
            vector_clock: Current vector clock {agent_id -> logical_count}

        Side effects:
            - Updates in-memory state store (O(1) dict operation)
            - Fires subscription callbacks (if any registered for this conversation)
            - WAL entry should be written by caller before acknowledging to client

        Performance: O(n) where n = number of subscribers for this conversation
        """
        # Increment this agent's logical timestamp
        # Initialize to 0 if agent_id not yet in clock
        if agent_id not in vector_clock:
            vector_clock[agent_id] = 0
        vector_clock[agent_id] += 1

        # Create new entry with current timestamp
        entry = StateEntry(
            value=value,
            vector_clock=vector_clock.copy(),  # Copy to avoid external mutations
            timestamp_utc=datetime.now(timezone.utc).timestamp()
        )

        # Check if we're overwriting an existing entry (for LWW conflict detection)
        old_entry = self._state[conversation_id].get(key)
        if old_entry is not None:
            # LWW: compare timestamps, higher wins
            if entry.timestamp_utc < old_entry.timestamp_utc:
                # Our write lost the conflict - don't overwrite
                logger.debug(
                    f"LWW conflict: incoming timestamp {entry.timestamp_utc} < "
                    f"existing {old_entry.timestamp_utc} for {conversation_id}:{key}"
                )
                return

        # Store the new entry
        self._state[conversation_id][key] = entry

        # Fire subscriptions
        for callback in self._subscriptions.get(conversation_id, []):
            try:
                callback(key, value)
            except Exception as e:
                logger.error(
                    f"Subscription callback failed for {conversation_id}: {e}",
                    exc_info=True
                )

    def get(self, conversation_id: str, key: str) -> Optional[Any]:
        """
        Get state value (O(1) dict lookup).

        Args:
            conversation_id: Conversation ID
            key: State key

        Returns:
            State value, or None if not found

        Performance: O(1) dict lookup - < 5ms latency
        """
        if conversation_id not in self._state or key not in self._state[conversation_id]:
            return None
        return self._state[conversation_id][key].value

    def get_entry(self, conversation_id: str, key: str) -> Optional[StateEntry]:
        """
        Get full state entry (value + metadata).

        Internal method for testing and debugging.

        Args:
            conversation_id: Conversation ID
            key: State key

        Returns:
            StateEntry object, or None if not found
        """
        if conversation_id not in self._state or key not in self._state[conversation_id]:
            return None
        return self._state[conversation_id][key]

    def _is_concurrent(
        self,
        vc1: Dict[str, int],
        vc2: Dict[str, int]
    ) -> bool:
        """
        Check if two vector clocks are concurrent (neither happened-before the other).

        Two vector clocks are concurrent if neither is a causal predecessor of the other.
        vc1 < vc2 (vc1 happened-before vc2) iff:
        - vc1[agent] <= vc2[agent] for all agents, AND
        - there exists at least one agent where vc1[agent] < vc2[agent]

        Args:
            vc1: First vector clock
            vc2: Second vector clock

        Returns:
            True if concurrent, False if one happened-before the other

        Example:
            >>> vc1 = {"agent_a": 1, "agent_b": 0}
            >>> vc2 = {"agent_a": 1, "agent_b": 1}
            >>> coordinator._is_concurrent(vc1, vc2)  # vc1 < vc2
            False
            >>> vc1 = {"agent_a": 1, "agent_b": 2}
            >>> vc2 = {"agent_a": 2, "agent_b": 1}
            >>> coordinator._is_concurrent(vc1, vc2)  # concurrent
            True
        """
        # Check if vc1 <= vc2 (all components of vc1 <= vc2)
        less_or_equal = all(vc1.get(agent, 0) <= vc2.get(agent, 0) for agent in set(vc1.keys()) | set(vc2.keys()))

        # Check if vc1 >= vc2 (all components of vc1 >= vc2)
        greater_or_equal = all(vc1.get(agent, 0) >= vc2.get(agent, 0) for agent in set(vc1.keys()) | set(vc2.keys()))

        # Concurrent if neither is a predecessor of the other
        return not (less_or_equal or greater_or_equal)

    def subscribe(self, conversation_id: str, callback: Callable) -> None:
        """
        Subscribe to state updates for a conversation.

        When state is updated via set(), all registered callbacks are invoked
        with (key, value) parameters.

        Args:
            conversation_id: Conversation ID to subscribe to
            callback: Function called with (key, value) signature on each update

        Example:
            >>> def on_update(key: str, value: Any) -> None:
            ...     print(f"State updated: {key}={value}")
            >>> coordinator.subscribe("conv_1", on_update)
        """
        self._subscriptions[conversation_id].append(callback)

    async def recover_from_wal(self, wal_entries: list) -> None:
        """
        Recover state coordinator from WAL entries.

        Called during Gateway.start() as part of sequential WAL recovery sequence.
        Filters for "state_coordinator" component entries and replays set() operations.

        Args:
            wal_entries: List of all WAL entries from gateway.wal file,
                        each entry is a dict with keys:
                        - entry_id: int (for tracking replay position)
                        - component: str (component name, e.g., "state_coordinator")
                        - conversation_id: str
                        - operation_type: str (e.g., "set")
                        - data: dict with operation-specific fields
                        - timestamp: float (operation timestamp)

        Side effects:
            - Replays all "set" operations for state_coordinator component
            - Updates in-memory state store
            - Updates _next_wal_id to prevent re-processing

        Example WAL entry:
            {
                "entry_id": 42,
                "component": "state_coordinator",
                "conversation_id": "conv_123",
                "operation_type": "set",
                "data": {
                    "key": "document_id",
                    "value": "doc_abc123",
                    "agent_id": "langchain_analyzer",
                    "vector_clock": {"langchain_analyzer": 1}
                },
                "timestamp": 1743948896.789012
            }
        """
        for entry in wal_entries:
            # Only process state_coordinator entries
            if entry.get("component") != "state_coordinator":
                continue

            conversation_id = entry.get("conversation_id")
            operation_type = entry.get("operation_type")
            data = entry.get("data", {})

            if operation_type == "set":
                key = data.get("key")
                value = data.get("value")
                vector_clock = data.get("vector_clock", {})
                agent_id = data.get("agent_id", "unknown")

                self.set(conversation_id, key, value, agent_id, vector_clock)

            # Update WAL ID tracking to prevent re-processing
            self._next_wal_id = max(self._next_wal_id, entry.get("entry_id", 0) + 1)

        recovered_keys = sum(len(v) for v in self._state.values())
        logger.info(f"Recovered state coordinator: {recovered_keys} keys from WAL")

    def get_state_snapshot(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get complete state snapshot for a conversation.

        Internal method for testing and debugging.

        Args:
            conversation_id: Conversation ID

        Returns:
            Dict mapping keys to values (StateEntry metadata excluded)
        """
        if conversation_id not in self._state:
            return {}
        return {
            key: entry.value
            for key, entry in self._state[conversation_id].items()
        }

    def get_vector_clock(self, conversation_id: str, key: str) -> Optional[Dict[str, int]]:
        """
        Get vector clock for a specific state entry.

        Internal method for testing.

        Args:
            conversation_id: Conversation ID
            key: State key

        Returns:
            Vector clock dict, or None if entry not found
        """
        entry = self.get_entry(conversation_id, key)
        if entry is None:
            return None
        return entry.vector_clock.copy()
