"""
Unit tests for DeduplicationTracker.

Tests cover:
- Basic mark/detect functionality (new messages return False, marked messages return True)
- Per-conversation isolation (same message_id in different conversations allowed)
- WAL recovery (simulate restart and verify state reconstructed)
- Edge cases (empty strings, special characters, boundary conditions)
- No false positives (different message_ids are tracked independently)
"""

import pytest
import uuid
from openclaw_gateway.deduplication import DeduplicationTracker


class TestMarkAndDetect:
    """Test basic mark_processed and is_duplicate functionality."""

    def test_new_message_returns_false(self):
        """A message that hasn't been marked should return False for is_duplicate."""
        tracker = DeduplicationTracker()
        msg_id = str(uuid.uuid4())
        conv_id = "conv_123"

        assert tracker.is_duplicate(conv_id, msg_id) is False

    def test_marked_message_returns_true(self):
        """A message that has been marked should return True for is_duplicate."""
        tracker = DeduplicationTracker()
        msg_id = str(uuid.uuid4())
        conv_id = "conv_123"

        tracker.mark_processed(conv_id, msg_id)
        assert tracker.is_duplicate(conv_id, msg_id) is True

    def test_mark_processed_returns_none(self):
        """mark_processed should return None."""
        tracker = DeduplicationTracker()
        msg_id = str(uuid.uuid4())
        conv_id = "conv_123"

        result = tracker.mark_processed(conv_id, msg_id)
        assert result is None

    def test_different_messages_tracked_separately(self):
        """Different message_ids in the same conversation are tracked independently."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"
        msg_id_1 = str(uuid.uuid4())
        msg_id_2 = str(uuid.uuid4())

        tracker.mark_processed(conv_id, msg_id_1)

        assert tracker.is_duplicate(conv_id, msg_id_1) is True
        assert tracker.is_duplicate(conv_id, msg_id_2) is False

    def test_marking_same_message_twice_is_idempotent(self):
        """Marking the same message twice should have no additional effect."""
        tracker = DeduplicationTracker()
        msg_id = str(uuid.uuid4())
        conv_id = "conv_123"

        tracker.mark_processed(conv_id, msg_id)
        tracker.mark_processed(conv_id, msg_id)

        # Should still report as duplicate
        assert tracker.is_duplicate(conv_id, msg_id) is True
        # Should have count of 1, not 2
        assert tracker.get_processed_count(conv_id) == 1

    def test_multiple_messages_in_conversation(self):
        """Multiple different messages can be marked in the same conversation."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"
        msg_ids = [str(uuid.uuid4()) for _ in range(5)]

        for msg_id in msg_ids:
            tracker.mark_processed(conv_id, msg_id)

        # All should be detected as duplicates
        for msg_id in msg_ids:
            assert tracker.is_duplicate(conv_id, msg_id) is True

    def test_unknown_conversation_returns_false(self):
        """Checking a message in an unknown conversation should return False."""
        tracker = DeduplicationTracker()
        unknown_conv_id = "unknown_conv"
        msg_id = str(uuid.uuid4())

        # Even if the message_id was marked in another conversation,
        # it's unknown in this one
        tracker.mark_processed("other_conv", msg_id)
        assert tracker.is_duplicate(unknown_conv_id, msg_id) is False


class TestPerConversationIsolation:
    """Test that message_ids are isolated per conversation."""

    @pytest.mark.parametrize("num_conversations", [2, 3, 5])
    def test_same_message_id_different_conversations(self, num_conversations):
        """Same message_id can be independently marked in different conversations."""
        tracker = DeduplicationTracker()
        msg_id = str(uuid.uuid4())
        conv_ids = [f"conv_{i}" for i in range(num_conversations)]

        # Mark the same message_id in different conversations
        for conv_id in conv_ids:
            # Should not be duplicate yet
            assert tracker.is_duplicate(conv_id, msg_id) is False
            # Mark it
            tracker.mark_processed(conv_id, msg_id)

        # Should be duplicate in all conversations now
        for conv_id in conv_ids:
            assert tracker.is_duplicate(conv_id, msg_id) is True

    def test_concurrent_conversations_dont_interfere(self):
        """Messages in different conversations shouldn't affect each other."""
        tracker = DeduplicationTracker()
        conv_id_1 = "conv_1"
        conv_id_2 = "conv_2"

        msg_id_1a = str(uuid.uuid4())
        msg_id_1b = str(uuid.uuid4())
        msg_id_2a = str(uuid.uuid4())
        msg_id_2b = str(uuid.uuid4())

        # Mark messages in conversation 1
        tracker.mark_processed(conv_id_1, msg_id_1a)
        tracker.mark_processed(conv_id_1, msg_id_1b)

        # Mark messages in conversation 2
        tracker.mark_processed(conv_id_2, msg_id_2a)
        tracker.mark_processed(conv_id_2, msg_id_2b)

        # Verify isolation
        assert tracker.is_duplicate(conv_id_1, msg_id_1a) is True
        assert tracker.is_duplicate(conv_id_1, msg_id_1b) is True
        assert tracker.is_duplicate(conv_id_1, msg_id_2a) is False
        assert tracker.is_duplicate(conv_id_1, msg_id_2b) is False

        assert tracker.is_duplicate(conv_id_2, msg_id_2a) is True
        assert tracker.is_duplicate(conv_id_2, msg_id_2b) is True
        assert tracker.is_duplicate(conv_id_2, msg_id_1a) is False
        assert tracker.is_duplicate(conv_id_2, msg_id_1b) is False


class TestWALRecovery:
    """Test WAL-based persistence and recovery."""

    @pytest.mark.asyncio
    async def test_recover_from_wal_filters_by_component(self):
        """Recovery should filter WAL entries by component == 'deduplication'."""
        tracker = DeduplicationTracker()

        wal_entries = [
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_1", "message_id": "msg_1"},
            },
            {
                "component": "state_coordinator",
                "data": {"conversation_id": "conv_1", "message_id": "msg_2"},
            },
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_1", "message_id": "msg_2"},
            },
        ]

        await tracker.recover_from_wal(wal_entries)

        # Only deduplication entries should be recovered
        assert tracker.is_duplicate("conv_1", "msg_1") is True
        assert tracker.is_duplicate("conv_1", "msg_2") is True

    @pytest.mark.asyncio
    async def test_recover_from_wal_reconstructs_state(self):
        """Recovery should reconstruct the exact state from WAL entries."""
        tracker = DeduplicationTracker()

        wal_entries = [
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_1", "message_id": "msg_1"},
            },
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_1", "message_id": "msg_2"},
            },
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_2", "message_id": "msg_3"},
            },
        ]

        await tracker.recover_from_wal(wal_entries)

        # Verify state matches
        assert tracker.get_processed_count("conv_1") == 2
        assert tracker.get_processed_count("conv_2") == 1
        assert tracker.is_duplicate("conv_1", "msg_1") is True
        assert tracker.is_duplicate("conv_1", "msg_2") is True
        assert tracker.is_duplicate("conv_2", "msg_3") is True

    @pytest.mark.asyncio
    async def test_wal_recovery_with_restart_simulation(self):
        """Test restart scenario: create tracker, add messages, recover in new instance."""
        # Phase 1: Original tracker marks messages
        tracker_1 = DeduplicationTracker()
        conv_id = "conv_test"
        msg_ids = [f"msg_{i}" for i in range(5)]

        for msg_id in msg_ids:
            tracker_1.mark_processed(conv_id, msg_id)

        # Simulate WAL entries that would be created
        wal_entries = [
            {
                "component": "deduplication",
                "data": {"conversation_id": conv_id, "message_id": msg_id},
            }
            for msg_id in msg_ids
        ]

        # Phase 2: Restart - new tracker recovers from WAL
        tracker_2 = DeduplicationTracker()
        await tracker_2.recover_from_wal(wal_entries)

        # Verify state matches original
        assert tracker_2.get_processed_count(conv_id) == 5
        for msg_id in msg_ids:
            assert tracker_2.is_duplicate(conv_id, msg_id) is True

    @pytest.mark.asyncio
    async def test_wal_recovery_with_empty_wal(self):
        """Recovery with no deduplication entries should result in empty state."""
        tracker = DeduplicationTracker()

        wal_entries = [
            {"component": "state_coordinator", "data": {}},
            {"component": "context_manager", "data": {}},
        ]

        await tracker.recover_from_wal(wal_entries)

        # Tracker should be empty
        assert tracker.get_processed_count("any_conv") == 0
        assert tracker.is_duplicate("any_conv", "any_msg") is False

    @pytest.mark.asyncio
    async def test_wal_recovery_skips_malformed_entries(self):
        """Recovery should skip entries with missing conversation_id or message_id."""
        tracker = DeduplicationTracker()

        wal_entries = [
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_1", "message_id": "msg_1"},
            },
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_2"},  # Missing message_id
            },
            {
                "component": "deduplication",
                "data": {"message_id": "msg_3"},  # Missing conversation_id
            },
            {
                "component": "deduplication",
                "data": {"conversation_id": "conv_3", "message_id": "msg_4"},
            },
        ]

        await tracker.recover_from_wal(wal_entries)

        # Only valid entries should be recovered
        assert tracker.is_duplicate("conv_1", "msg_1") is True
        assert tracker.get_processed_count("conv_2") == 0
        assert tracker.is_duplicate("conv_3", "msg_4") is True


class TestGetProcessedCount:
    """Test get_processed_count functionality."""

    def test_get_processed_count_new_conversation(self):
        """Count for new conversation should be 0."""
        tracker = DeduplicationTracker()
        assert tracker.get_processed_count("new_conv") == 0

    def test_get_processed_count_after_marking(self):
        """Count should reflect number of unique messages marked."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"

        for i in range(10):
            tracker.mark_processed(conv_id, f"msg_{i}")

        assert tracker.get_processed_count(conv_id) == 10

    def test_get_processed_count_with_duplicate_marks(self):
        """Count should reflect unique message_ids, not total marks."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"
        msg_id = "msg_123"

        # Mark the same message multiple times
        for _ in range(5):
            tracker.mark_processed(conv_id, msg_id)

        # Should count as 1, not 5
        assert tracker.get_processed_count(conv_id) == 1


class TestClearConversation:
    """Test clear_conversation functionality."""

    def test_clear_conversation_removes_state(self):
        """Clearing a conversation should remove all its tracked messages."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"

        # Mark some messages
        for i in range(5):
            tracker.mark_processed(conv_id, f"msg_{i}")

        assert tracker.get_processed_count(conv_id) == 5

        # Clear the conversation
        tracker.clear_conversation(conv_id)

        # Should now be empty
        assert tracker.get_processed_count(conv_id) == 0
        assert tracker.is_duplicate(conv_id, "msg_0") is False

    def test_clear_conversation_is_safe_for_unknown_conversations(self):
        """Clearing a non-existent conversation should not raise an error."""
        tracker = DeduplicationTracker()

        # Should not raise
        tracker.clear_conversation("unknown_conv")

        # Tracker should still work normally
        tracker.mark_processed("conv_123", "msg_1")
        assert tracker.is_duplicate("conv_123", "msg_1") is True

    def test_clear_conversation_doesnt_affect_other_conversations(self):
        """Clearing one conversation should not affect others."""
        tracker = DeduplicationTracker()

        # Mark messages in two conversations
        tracker.mark_processed("conv_1", "msg_1")
        tracker.mark_processed("conv_2", "msg_2")

        # Clear one
        tracker.clear_conversation("conv_1")

        # First should be clear, second should remain
        assert tracker.get_processed_count("conv_1") == 0
        assert tracker.get_processed_count("conv_2") == 1
        assert tracker.is_duplicate("conv_2", "msg_2") is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_conversation_id(self):
        """Empty string conversation_id should be handled like any other."""
        tracker = DeduplicationTracker()
        msg_id = str(uuid.uuid4())

        tracker.mark_processed("", msg_id)
        assert tracker.is_duplicate("", msg_id) is True

    def test_empty_string_message_id(self):
        """Empty string message_id should be tracked like any other."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"

        tracker.mark_processed(conv_id, "")
        assert tracker.is_duplicate(conv_id, "") is True

    def test_special_characters_in_message_id(self):
        """Message IDs with special characters should be handled correctly."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"
        msg_id = "msg-123_abc.def/ghi~jkl!@#$%"

        tracker.mark_processed(conv_id, msg_id)
        assert tracker.is_duplicate(conv_id, msg_id) is True

    def test_uuid_format_message_id(self):
        """UUID-format message IDs should work correctly."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"
        msg_id = str(uuid.uuid4())

        tracker.mark_processed(conv_id, msg_id)
        assert tracker.is_duplicate(conv_id, msg_id) is True

    def test_hierarchical_conversation_id(self):
        """Hierarchical conversation IDs should work correctly."""
        tracker = DeduplicationTracker()
        conv_id = "root_123.subtask_1.subsubtask_2"
        msg_id = str(uuid.uuid4())

        tracker.mark_processed(conv_id, msg_id)
        assert tracker.is_duplicate(conv_id, msg_id) is True

    def test_large_number_of_messages(self):
        """Tracker should handle 1000+ messages without false positives."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"
        num_messages = 1000

        msg_ids = [f"msg_{i}" for i in range(num_messages)]

        # Mark all messages
        for msg_id in msg_ids:
            tracker.mark_processed(conv_id, msg_id)

        # Verify all are detected as duplicates (no false negatives)
        for msg_id in msg_ids:
            assert tracker.is_duplicate(conv_id, msg_id) is True

        # Verify non-existent messages are not detected (no false positives)
        for i in range(num_messages, num_messages + 100):
            assert tracker.is_duplicate(conv_id, f"msg_{i}") is False

        assert tracker.get_processed_count(conv_id) == num_messages

    def test_many_conversations(self):
        """Tracker should handle many conversations without interference."""
        tracker = DeduplicationTracker()
        num_conversations = 100
        messages_per_conv = 10

        msg_id_template = "msg_{}"

        # Mark messages in many conversations
        for conv_idx in range(num_conversations):
            conv_id = f"conv_{conv_idx}"
            for msg_idx in range(messages_per_conv):
                msg_id = msg_id_template.format(msg_idx)
                tracker.mark_processed(conv_id, msg_id)

        # Verify each conversation has correct counts
        for conv_idx in range(num_conversations):
            conv_id = f"conv_{conv_idx}"
            assert tracker.get_processed_count(conv_id) == messages_per_conv

        # Verify isolation: same message_id in different convs
        msg_id = msg_id_template.format(0)
        for conv_idx in range(num_conversations):
            assert tracker.is_duplicate(f"conv_{conv_idx}", msg_id) is True

    @pytest.mark.asyncio
    async def test_wal_recovery_with_large_dataset(self):
        """WAL recovery should handle 1000+ messages efficiently."""
        tracker = DeduplicationTracker()
        num_messages = 1000
        conv_id = "conv_123"

        # Create WAL entries for many messages
        wal_entries = [
            {
                "component": "deduplication",
                "data": {"conversation_id": conv_id, "message_id": f"msg_{i}"},
            }
            for i in range(num_messages)
        ]

        await tracker.recover_from_wal(wal_entries)

        # Verify all recovered
        assert tracker.get_processed_count(conv_id) == num_messages

        # Verify random samples are duplicates
        assert tracker.is_duplicate(conv_id, "msg_0") is True
        assert tracker.is_duplicate(conv_id, "msg_500") is True
        assert tracker.is_duplicate(conv_id, "msg_999") is True

        # Verify non-existent is not duplicate
        assert tracker.is_duplicate(conv_id, "msg_1000") is False


class TestNoFalsePositives:
    """Verify that different message_ids are tracked independently."""

    @pytest.mark.parametrize(
        "msg_id_1,msg_id_2",
        [
            ("msg_1", "msg_2"),
            ("a", "b"),
            ("conv_123", "conv_124"),
            (str(uuid.uuid4()), str(uuid.uuid4())),
        ],
    )
    def test_different_message_ids_not_conflated(self, msg_id_1, msg_id_2):
        """Different message_ids should never be reported as the same."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"

        tracker.mark_processed(conv_id, msg_id_1)

        # msg_id_2 should not be reported as duplicate
        assert tracker.is_duplicate(conv_id, msg_id_2) is False

    def test_message_id_prefix_not_conflated(self):
        """Message IDs that are prefixes of others should not be conflated."""
        tracker = DeduplicationTracker()
        conv_id = "conv_123"

        msg_id_short = "msg"
        msg_id_long = "msg_123"

        tracker.mark_processed(conv_id, msg_id_short)

        # Longer ID should not be reported as duplicate
        assert tracker.is_duplicate(conv_id, msg_id_long) is False
