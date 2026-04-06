"""Deduplication tracker for preventing duplicate message processing.

This module provides the DeduplicationTracker class which maintains per-conversation
sets of processed message IDs to prevent idempotent message processing after rollback
or network retries.

Key features:
- O(1) duplicate detection via set-based membership testing
- Per-conversation message ID isolation
- WAL-based persistence and recovery
- Zero data loss across gateway restart
"""

from typing import Dict, List, Set, Any


class DeduplicationTracker:
    """Tracks processed messages per conversation to detect duplicates.

    Maintains an in-memory store of message IDs that have been processed,
    organized by conversation_id. Supports WAL-based persistence and recovery
    to ensure no duplicate processing even after failures.

    Attributes:
        _processed: Dict mapping conversation_id → Set of message_ids that have
                   been marked as processed in that conversation.
    """

    def __init__(self) -> None:
        """Initialize empty deduplication tracker.

        Creates an empty dictionary that will map conversation IDs to sets
        of processed message IDs.
        """
        self._processed: Dict[str, Set[str]] = {}

    def is_duplicate(self, conversation_id: str, message_id: str) -> bool:
        """Check if a message has already been processed in a conversation.

        Performs O(1) lookup in the per-conversation set to determine if the
        message_id has been previously marked as processed.

        Args:
            conversation_id: The conversation context for this message.
            message_id: The unique identifier of the message to check.

        Returns:
            True if the message has been processed before in this conversation,
            False otherwise (including for unknown conversations).
        """
        if conversation_id not in self._processed:
            return False
        return message_id in self._processed[conversation_id]

    def mark_processed(self, conversation_id: str, message_id: str) -> None:
        """Mark a message as processed in a conversation.

        Adds the message_id to the set of processed messages for the given
        conversation_id. If this is the first message in the conversation,
        creates a new set.

        Idempotent operation: marking the same message multiple times has no
        additional effect (set semantics).

        Args:
            conversation_id: The conversation context for this message.
            message_id: The unique identifier of the message to mark as processed.
        """
        if conversation_id not in self._processed:
            self._processed[conversation_id] = set()
        self._processed[conversation_id].add(message_id)

    async def recover_from_wal(self, wal_entries: List[Dict[str, Any]]) -> None:
        """Reconstruct deduplication state from WAL entries.

        Filters the WAL entries to those with component == "deduplication" and
        reconstructs the _processed dictionary by replaying mark_processed calls
        for each entry.

        This method is called during Gateway startup to restore the deduplication
        state after a crash or restart.

        Args:
            wal_entries: List of WAL entry dictionaries. Each entry should have:
                - component: str indicating the component that created the entry
                - data: dict with 'conversation_id' and 'message_id' keys
        """
        # Filter to only deduplication entries
        dedup_entries = [
            entry for entry in wal_entries
            if entry.get("component") == "deduplication"
        ]

        # Reconstruct state by replaying each entry
        for entry in dedup_entries:
            data = entry.get("data", {})
            conversation_id = data.get("conversation_id")
            message_id = data.get("message_id")

            # Skip entries with missing data
            if conversation_id is not None and message_id is not None:
                self.mark_processed(conversation_id, message_id)

    def get_processed_count(self, conversation_id: str) -> int:
        """Get the number of processed messages in a conversation.

        Returns the count of unique message IDs that have been marked as
        processed in the given conversation.

        Args:
            conversation_id: The conversation to query.

        Returns:
            The number of processed messages in this conversation. Returns 0
            for unknown conversations.
        """
        if conversation_id not in self._processed:
            return 0
        return len(self._processed[conversation_id])

    def clear_conversation(self, conversation_id: str) -> None:
        """Remove all tracking for a conversation.

        Deletes the set of processed message IDs for the given conversation_id.
        Useful for cleanup when a conversation is completed or archived.

        Safe operation: clearing a non-existent conversation does nothing.

        Args:
            conversation_id: The conversation to clear.
        """
        if conversation_id in self._processed:
            del self._processed[conversation_id]
