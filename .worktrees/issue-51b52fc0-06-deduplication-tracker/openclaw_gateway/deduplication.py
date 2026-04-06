"""Deduplication tracker for preventing double-processing of messages.

This module provides the DeduplicationTracker class for maintaining idempotence
across rollback/retry scenarios. The tracker maintains per-conversation sets of
processed message IDs and supports WAL-based recovery on startup.
"""

from typing import Set, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DeduplicationTracker:
    """Tracks processed message IDs to prevent double-processing.

    Used during rollback/reroute scenarios to ensure idempotence.
    Persisted to WAL for recovery via recover_from_wal().

    Attributes:
        _processed: Dict mapping conversation_id -> set(message_ids)
    """

    def __init__(self) -> None:
        """Initialize empty deduplication tracker."""
        self._processed: Dict[str, Set[str]] = {}  # conversation_id -> set(message_ids)

    def is_duplicate(self, conversation_id: str, message_id: str) -> bool:
        """Check if message has already been processed.

        Args:
            conversation_id: Conversation this message belongs to
            message_id: UUID-v4 message identifier

        Returns:
            True if message was previously processed, False otherwise

        Performance: O(1) lookup
        """
        if conversation_id not in self._processed:
            return False
        return message_id in self._processed[conversation_id]

    def mark_processed(self, conversation_id: str, message_id: str) -> None:
        """Mark message as processed.

        Args:
            conversation_id: Conversation this message belongs to
            message_id: UUID-v4 message identifier

        Side effects:
            - Adds message_id to in-memory set
            - Caller responsible for WAL write (via Gateway._write_wal_entry())

        Performance: O(1) insertion
        """
        if conversation_id not in self._processed:
            self._processed[conversation_id] = set()

        if message_id in self._processed[conversation_id]:
            logger.warning(
                f"Message {message_id} in conversation {conversation_id} "
                f"already marked as processed"
            )

        self._processed[conversation_id].add(message_id)

    def get_processed_count(self, conversation_id: str) -> int:
        """Get count of processed messages for a conversation.

        Args:
            conversation_id: Conversation ID to check

        Returns:
            Number of processed messages in this conversation
        """
        return len(self._processed.get(conversation_id, set()))

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear processed messages for a conversation (e.g., after completion).

        Args:
            conversation_id: Conversation ID to clear
        """
        if conversation_id in self._processed:
            del self._processed[conversation_id]

    async def recover_from_wal(self, wal_entries: List[Dict[str, Any]]) -> None:
        """Recover deduplication state from WAL entries.

        Called during Gateway.start() as part of sequential WAL recovery.

        Args:
            wal_entries: List of all WAL entries from gateway.wal file

        Algorithm:
        1. Filter entries with component == "deduplication"
        2. For each mark_processed entry, restore message_id to processed set
        3. Rebuild in-memory _processed dict to match persisted state

        Performance: O(n) where n = number of deduplication entries in WAL
        """
        for entry in wal_entries:
            if entry.get("component") != "deduplication":
                continue

            operation_type = entry.get("operation_type")
            conversation_id = entry.get("conversation_id")
            data = entry.get("data", {})

            if operation_type == "mark_processed":
                message_id = data.get("message_id")
                if conversation_id and message_id:
                    if conversation_id not in self._processed:
                        self._processed[conversation_id] = set()
                    self._processed[conversation_id].add(message_id)

            elif operation_type == "clear_conversation":
                if conversation_id in self._processed:
                    del self._processed[conversation_id]

        logger.info(
            f"Recovered deduplication state: "
            f"{sum(len(v) for v in self._processed.values())} messages "
            f"across {len(self._processed)} conversations"
        )
