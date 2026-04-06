"""
Distributed state coordinator with vector clocks and LWW conflict resolution.

This module maintains per-conversation state with vector clocks for causal
ordering, detects concurrent updates, resolves conflicts using Last-Write-Wins
(LWW), persists to WAL, and provides subscription callbacks for state updates.

Responsibilities:
- Per-conversation key-value state store
- Vector clock tracking for causal ordering
- LWW conflict resolution (higher timestamp wins)
- State subscriptions and callbacks
- WAL persistence and recovery
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StateEntry:
    """Single state entry with vector clock metadata."""

    value: Any
    vector_clock: Dict[str, int]  # agent_id -> logical timestamp
    timestamp_utc: float  # unix timestamp for LWW


class StateCoordinator:
    """
    Manages distributed state with vector clocks and LWW conflict resolution.

    Responsibilities:
    - Per-conversation key-value state store
    - Vector clock tracking for causal ordering
    - LWW conflict resolution (higher timestamp wins)
    - State subscriptions and callbacks
    - WAL persistence and recovery
    """

    def __init__(self, wal_path: str = "/var/lib/openclaw/gateway.wal"):
        """
        Initialize state coordinator.

        Args:
            wal_path: Path to WAL file
        """
        # State store: conversation_id -> {key -> StateEntry}
        self._state: Dict[str, Dict[str, StateEntry]] = defaultdict(dict)

        # Subscriptions: conversation_id -> list(callbacks)
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)

        self._wal_path = wal_path
        self._next_wal_id = 1

    def set(
        self,
        conversation_id: str,
        key: str,
        value: Any,
        agent_id: str,
        vector_clock: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Set state value with vector clock.

        Increments the agent's logical clock, applies LWW if concurrent update
        exists, stores entry, and fires subscriptions.

        Args:
            conversation_id: Conversation ID
            key: State key
            value: State value
            agent_id: Agent making update
            vector_clock: Current vector clock {agent_id -> logical_count}.
                         If None, initialized as empty dict.

        Side effects:
            - Updates state store (LWW applied if concurrent update exists)
            - Increments vector_clock[agent_id]
            - Triggers subscriptions
            - WAL entry written by caller
        """
        if vector_clock is None:
            vector_clock = {}

        # Increment this agent's clock
        if agent_id not in vector_clock:
            vector_clock[agent_id] = 0
        vector_clock[agent_id] += 1

        timestamp_utc = datetime.now(timezone.utc).timestamp()

        # Check for concurrent update (LWW resolution)
        if conversation_id in self._state and key in self._state[conversation_id]:
            existing = self._state[conversation_id][key]
            # If new write has higher timestamp, it wins
            # If same timestamp or lower, keep existing (deterministic choice)
            if timestamp_utc <= existing.timestamp_utc:
                logger.debug(
                    f"LWW: Ignoring write ({timestamp_utc}) for {key}, "
                    f"keeping existing ({existing.timestamp_utc})"
                )
                return

        entry = StateEntry(
            value=value,
            vector_clock=vector_clock.copy(),
            timestamp_utc=timestamp_utc,
        )

        self._state[conversation_id][key] = entry

        # Fire subscriptions
        for callback in self._subscriptions.get(conversation_id, []):
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Subscription callback failed: {e}")

    def get(self, conversation_id: str, key: str) -> Optional[Any]:
        """
        Get state value.

        Args:
            conversation_id: Conversation ID
            key: State key

        Returns:
            State value, or None if not found

        Performance: O(1) dict lookup
        """
        if conversation_id not in self._state or key not in self._state[conversation_id]:
            return None
        return self._state[conversation_id][key].value

    def get_vector_clock(self, conversation_id: str) -> Dict[str, int]:
        """
        Get the aggregated vector clock for a conversation.

        Returns the "maximum" vector clock across all state entries
        for this conversation (per-agent max).

        Args:
            conversation_id: Conversation ID

        Returns:
            Vector clock dict {agent_id -> logical_count}, or empty dict
            if conversation not found
        """
        if conversation_id not in self._state:
            return {}

        # Aggregate vector clocks: take max for each agent
        aggregated: Dict[str, int] = {}
        for key_data in self._state[conversation_id].values():
            for agent_id, clock_value in key_data.vector_clock.items():
                aggregated[agent_id] = max(aggregated.get(agent_id, 0), clock_value)

        return aggregated

    def _is_concurrent(self, vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """
        Check if two vector clocks are concurrent (neither happened-before the other).

        Args:
            vc1: First vector clock
            vc2: Second vector clock

        Returns:
            True if concurrent, False if one happened-before the other

        Algorithm:
        - vc1 < vc2 if vc1[agent] <= vc2[agent] for all agents and vc1 != vc2
        - vc1 and vc2 concurrent if neither <
        """
        less_or_equal = all(vc1.get(agent, 0) <= vc2.get(agent, 0) for agent in vc1)
        greater_or_equal = all(vc1.get(agent, 0) >= vc2.get(agent, 0) for agent in vc1)

        # Concurrent if neither is a predecessor of the other
        return not (less_or_equal or greater_or_equal)

    def subscribe(self, conversation_id: str, callback: Callable) -> None:
        """
        Subscribe to state updates for a conversation.

        Args:
            conversation_id: Conversation ID
            callback: Function called with (key, value) on each update
        """
        self._subscriptions[conversation_id].append(callback)

    async def recover_from_wal(self, wal_entries: list) -> None:
        """
        Recover state coordinator from WAL.

        Called during Gateway.start() as part of sequential WAL recovery.

        During recovery, restores state and vector clocks exactly as they were,
        without triggering callbacks or incrementing clocks.

        Args:
            wal_entries: List of all WAL entries from gateway.wal file
        """
        for entry in wal_entries:
            if entry.get("component") != "state_coordinator":
                continue

            conversation_id = entry["conversation_id"]
            operation_type = entry["operation_type"]
            data = entry["data"]

            if operation_type == "set":
                key = data.get("key")
                value = data.get("value")
                vector_clock = data.get("vector_clock", {})
                timestamp_utc = data.get("timestamp_utc", 0.0)

                # Restore state entry directly without calling set()
                # (which would increment vector clock and trigger callbacks)
                state_entry = StateEntry(
                    value=value,
                    vector_clock=vector_clock.copy(),
                    timestamp_utc=timestamp_utc,
                )
                self._state[conversation_id][key] = state_entry

            self._next_wal_id = max(self._next_wal_id, entry["entry_id"] + 1)

        logger.info(f"Recovered state: {sum(len(v) for v in self._state.values())} keys")
