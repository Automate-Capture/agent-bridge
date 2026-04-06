"""Tests for WAL recovery functionality."""

import pytest
import asyncio
import uuid
from openclaw_gateway import DeduplicationTracker


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


class TestWALRecovery:
    """Test WAL-based recovery of deduplication state."""

    def test_recover_from_empty_wal(self):
        """Test recovery from empty WAL entries list."""
        tracker = DeduplicationTracker()
        wal_entries = []

        run_async(tracker.recover_from_wal(wal_entries))

        # Should have no processed messages
        assert tracker.get_processed_count("any_conv") == 0

    def test_recover_filters_by_component(self):
        """Test recover_from_wal filters entries by component == 'deduplication'."""
        tracker = DeduplicationTracker()
        message_id = str(uuid.uuid4())
        conversation_id = "conv_123"

        # Create WAL entries with mixed components
        wal_entries = [
            {
                "component": "context_manager",
                "operation_type": "create_context",
                "conversation_id": conversation_id,
                "data": {"agent": "test"},
            },
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": conversation_id,
                "data": {"message_id": message_id},
            },
            {
                "component": "state_coordinator",
                "operation_type": "update_state",
                "conversation_id": conversation_id,
                "data": {"key": "value"},
            },
        ]

        run_async(tracker.recover_from_wal(wal_entries))

        # Only deduplication entries should be processed
        assert tracker.is_duplicate(conversation_id, message_id) is True

    def test_recover_single_mark_processed_entry(self):
        """Test recovery of single mark_processed entry."""
        tracker = DeduplicationTracker()
        message_id = str(uuid.uuid4())
        conversation_id = "conv_123"

        wal_entries = [
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": conversation_id,
                "data": {"message_id": message_id},
            }
        ]

        run_async(tracker.recover_from_wal(wal_entries))

        assert tracker.is_duplicate(conversation_id, message_id) is True
        assert tracker.get_processed_count(conversation_id) == 1

    def test_recover_multiple_messages_single_conversation(self):
        """Test recovery of multiple messages in single conversation."""
        tracker = DeduplicationTracker()
        message_ids = [str(uuid.uuid4()) for _ in range(5)]
        conversation_id = "conv_123"

        wal_entries = [
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": conversation_id,
                "data": {"message_id": msg_id},
            }
            for msg_id in message_ids
        ]

        run_async(tracker.recover_from_wal(wal_entries))

        assert tracker.get_processed_count(conversation_id) == 5
        for msg_id in message_ids:
            assert tracker.is_duplicate(conversation_id, msg_id) is True

    def test_recover_multiple_conversations(self):
        """Test recovery of messages across multiple conversations."""
        tracker = DeduplicationTracker()
        message_id_a = str(uuid.uuid4())
        message_id_b = str(uuid.uuid4())
        message_id_c = str(uuid.uuid4())

        wal_entries = [
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_a",
                "data": {"message_id": message_id_a},
            },
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_b",
                "data": {"message_id": message_id_b},
            },
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_b",
                "data": {"message_id": message_id_c},
            },
        ]

        run_async(tracker.recover_from_wal(wal_entries))

        assert tracker.get_processed_count("conv_a") == 1
        assert tracker.get_processed_count("conv_b") == 2
        assert tracker.is_duplicate("conv_a", message_id_a) is True
        assert tracker.is_duplicate("conv_b", message_id_b) is True
        assert tracker.is_duplicate("conv_b", message_id_c) is True

    def test_recover_with_clear_conversation(self):
        """Test recovery handles clear_conversation operations."""
        tracker = DeduplicationTracker()
        message_id_a = str(uuid.uuid4())
        message_id_b = str(uuid.uuid4())
        message_id_c = str(uuid.uuid4())

        wal_entries = [
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_a",
                "data": {"message_id": message_id_a},
            },
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_b",
                "data": {"message_id": message_id_b},
            },
            {
                "component": "deduplication",
                "operation_type": "clear_conversation",
                "conversation_id": "conv_a",
                "data": {},
            },
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_c",
                "data": {"message_id": message_id_c},
            },
        ]

        run_async(tracker.recover_from_wal(wal_entries))

        # conv_a was cleared
        assert tracker.get_processed_count("conv_a") == 0

        # conv_b was not cleared
        assert tracker.get_processed_count("conv_b") == 1
        assert tracker.is_duplicate("conv_b", message_id_b) is True

        # conv_c was added after
        assert tracker.get_processed_count("conv_c") == 1
        assert tracker.is_duplicate("conv_c", message_id_c) is True

    def test_restart_scenario_simulation(self):
        """Simulate restart: create tracker, mark messages, create new tracker, recover."""
        # Simulate original tracker state
        original_tracker = DeduplicationTracker()
        message_id_1 = str(uuid.uuid4())
        message_id_2 = str(uuid.uuid4())
        conversation_id = "conv_123"

        original_tracker.mark_processed(conversation_id, message_id_1)
        original_tracker.mark_processed(conversation_id, message_id_2)

        # Simulate WAL entries from original state
        wal_entries = [
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": conversation_id,
                "data": {"message_id": message_id_1},
            },
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": conversation_id,
                "data": {"message_id": message_id_2},
            },
        ]

        # Create new tracker after "restart"
        recovered_tracker = DeduplicationTracker()
        run_async(recovered_tracker.recover_from_wal(wal_entries))

        # Verify recovered state matches original
        assert recovered_tracker.get_processed_count(conversation_id) == 2
        assert recovered_tracker.is_duplicate(conversation_id, message_id_1) is True
        assert recovered_tracker.is_duplicate(conversation_id, message_id_2) is True

    def test_recovery_zero_loss_verification(self):
        """Test recovery preserves all messages (zero data loss)."""
        tracker = DeduplicationTracker()

        # Create a large batch of messages across multiple conversations
        messages_per_conv = 100
        num_conversations = 10
        all_entries = []

        for conv_idx in range(num_conversations):
            conversation_id = f"conv_{conv_idx}"
            for msg_idx in range(messages_per_conv):
                message_id = f"msg_{conv_idx}_{msg_idx}"
                all_entries.append(
                    {
                        "component": "deduplication",
                        "operation_type": "mark_processed",
                        "conversation_id": conversation_id,
                        "data": {"message_id": message_id},
                    }
                )

        # Recover all entries
        run_async(tracker.recover_from_wal(all_entries))

        # Verify all recovered
        total_count = sum(
            tracker.get_processed_count(f"conv_{i}") for i in range(num_conversations)
        )
        assert total_count == messages_per_conv * num_conversations

        # Spot check a few messages
        assert tracker.is_duplicate("conv_0", "msg_0_0") is True
        assert tracker.is_duplicate("conv_5", "msg_5_50") is True
        assert tracker.is_duplicate("conv_9", "msg_9_99") is True

    def test_recover_handles_missing_fields_gracefully(self):
        """Test recovery handles malformed entries gracefully."""
        tracker = DeduplicationTracker()

        wal_entries = [
            # Valid entry
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_1",
                "data": {"message_id": "msg_1"},
            },
            # Missing conversation_id
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": None,
                "data": {"message_id": "msg_2"},
            },
            # Missing message_id
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_3",
                "data": {"message_id": None},
            },
            # Missing data entirely
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_4",
            },
            # Valid entry at end
            {
                "component": "deduplication",
                "operation_type": "mark_processed",
                "conversation_id": "conv_5",
                "data": {"message_id": "msg_5"},
            },
        ]

        run_async(tracker.recover_from_wal(wal_entries))

        # Only valid entries should be recovered
        assert tracker.get_processed_count("conv_1") == 1
        assert tracker.get_processed_count("conv_3") == 0
        assert tracker.get_processed_count("conv_4") == 0
        assert tracker.get_processed_count("conv_5") == 1
