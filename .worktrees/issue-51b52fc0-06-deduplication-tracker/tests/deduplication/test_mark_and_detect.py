"""Unit tests for mark_processed and is_duplicate functionality."""

import pytest
import uuid
from openclaw_gateway import DeduplicationTracker


class TestMarkAndDetect:
    """Test basic mark/detect functionality."""

    def test_mark_processed_returns_none(self):
        """Test that mark_processed returns None."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id = str(uuid.uuid4())

        result = tracker.mark_processed(conversation_id, message_id)

        assert result is None

    def test_is_duplicate_returns_false_for_new_message(self):
        """Test is_duplicate returns False for unmarked messages."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id = str(uuid.uuid4())

        is_dup = tracker.is_duplicate(conversation_id, message_id)

        assert is_dup is False

    def test_is_duplicate_returns_true_after_mark(self):
        """Test is_duplicate returns True after mark_processed."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id = str(uuid.uuid4())

        tracker.mark_processed(conversation_id, message_id)
        is_dup = tracker.is_duplicate(conversation_id, message_id)

        assert is_dup is True

    def test_different_messages_tracked_separately(self):
        """Test that different messages in same conversation are tracked independently."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id_1 = str(uuid.uuid4())
        message_id_2 = str(uuid.uuid4())

        tracker.mark_processed(conversation_id, message_id_1)

        assert tracker.is_duplicate(conversation_id, message_id_1) is True
        assert tracker.is_duplicate(conversation_id, message_id_2) is False

    def test_get_processed_count_zero_initially(self):
        """Test get_processed_count returns 0 for new conversation."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"

        count = tracker.get_processed_count(conversation_id)

        assert count == 0

    def test_get_processed_count_increments(self):
        """Test get_processed_count increments with mark_processed."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id_1 = str(uuid.uuid4())
        message_id_2 = str(uuid.uuid4())

        assert tracker.get_processed_count(conversation_id) == 0

        tracker.mark_processed(conversation_id, message_id_1)
        assert tracker.get_processed_count(conversation_id) == 1

        tracker.mark_processed(conversation_id, message_id_2)
        assert tracker.get_processed_count(conversation_id) == 2

    def test_get_processed_count_accurate(self):
        """Test get_processed_count returns accurate count per conversation."""
        tracker = DeduplicationTracker()
        conv_a = "conv_a"
        conv_b = "conv_b"

        # Add 3 messages to conv_a
        for _ in range(3):
            tracker.mark_processed(conv_a, str(uuid.uuid4()))

        # Add 2 messages to conv_b
        for _ in range(2):
            tracker.mark_processed(conv_b, str(uuid.uuid4()))

        assert tracker.get_processed_count(conv_a) == 3
        assert tracker.get_processed_count(conv_b) == 2

    def test_clear_conversation_removes_state(self):
        """Test clear_conversation removes conversation from tracking."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id = str(uuid.uuid4())

        tracker.mark_processed(conversation_id, message_id)
        assert tracker.get_processed_count(conversation_id) == 1

        tracker.clear_conversation(conversation_id)

        assert tracker.get_processed_count(conversation_id) == 0
        assert tracker.is_duplicate(conversation_id, message_id) is False

    def test_clear_conversation_idempotent(self):
        """Test clear_conversation is idempotent."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"

        # Should not raise even if conversation doesn't exist
        tracker.clear_conversation(conversation_id)
        tracker.clear_conversation(conversation_id)

    def test_mark_same_message_twice_idempotent(self):
        """Test marking same message twice doesn't change state."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id = str(uuid.uuid4())

        tracker.mark_processed(conversation_id, message_id)
        assert tracker.get_processed_count(conversation_id) == 1

        tracker.mark_processed(conversation_id, message_id)
        assert tracker.get_processed_count(conversation_id) == 1
        assert tracker.is_duplicate(conversation_id, message_id) is True
