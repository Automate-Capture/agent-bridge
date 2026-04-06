"""Tests for edge cases."""

import pytest
import uuid
from openclaw_gateway import DeduplicationTracker


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_conversation_id(self):
        """Test handling of empty conversation_id."""
        tracker = DeduplicationTracker()
        message_id = str(uuid.uuid4())

        tracker.mark_processed("", message_id)

        assert tracker.is_duplicate("", message_id) is True
        assert tracker.get_processed_count("") == 1

    def test_special_characters_in_message_id(self):
        """Test message_id with UUID hyphens is handled correctly."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        # Standard UUID v4 format with hyphens
        message_id = str(uuid.uuid4())

        tracker.mark_processed(conversation_id, message_id)

        assert tracker.is_duplicate(conversation_id, message_id) is True

    def test_special_characters_in_conversation_id(self):
        """Test conversation_id with special characters."""
        tracker = DeduplicationTracker()
        special_chars = [
            "conv_123-456",
            "conv/subtask/sub",
            "conv.subtask.sub",
            "conv@agent:test",
            "conv_123_!@#$%",
        ]
        message_id = str(uuid.uuid4())

        for conv_id in special_chars:
            tracker.mark_processed(conv_id, message_id)
            assert tracker.is_duplicate(conv_id, message_id) is True

    def test_very_long_message_id(self):
        """Test handling of very long message_id string."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        # Create a very long message_id
        message_id = "x" * 10000

        tracker.mark_processed(conversation_id, message_id)

        assert tracker.is_duplicate(conversation_id, message_id) is True
        assert tracker.get_processed_count(conversation_id) == 1

    def test_very_long_conversation_id(self):
        """Test handling of very long conversation_id string."""
        tracker = DeduplicationTracker()
        # Create a very long conversation_id
        conversation_id = "x" * 10000
        message_id = str(uuid.uuid4())

        tracker.mark_processed(conversation_id, message_id)

        assert tracker.is_duplicate(conversation_id, message_id) is True
        assert tracker.get_processed_count(conversation_id) == 1

    def test_unicode_in_conversation_id(self):
        """Test unicode characters in conversation_id."""
        tracker = DeduplicationTracker()
        unicode_convs = [
            "conversation_日本語_123",
            "conv_中文_456",
            "conv_😀_emoji",
            "conv_ñ_spanish",
        ]
        message_id = str(uuid.uuid4())

        for conv_id in unicode_convs:
            tracker.mark_processed(conv_id, message_id)
            assert tracker.is_duplicate(conv_id, message_id) is True

    def test_unicode_in_message_id(self):
        """Test unicode characters in message_id."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        unicode_msg_ids = [
            "msg_日本語_123",
            "msg_😀_emoji",
            "msg_ñ_spanish",
        ]

        for msg_id in unicode_msg_ids:
            tracker.mark_processed(conversation_id, msg_id)
            assert tracker.is_duplicate(conversation_id, msg_id) is True

        assert tracker.get_processed_count(conversation_id) == len(unicode_msg_ids)

    def test_whitespace_in_ids(self):
        """Test whitespace in conversation_id and message_id."""
        tracker = DeduplicationTracker()

        # Conversation ID with spaces
        conv_with_spaces = "conversation with spaces"
        msg_with_spaces = "message with spaces"

        tracker.mark_processed(conv_with_spaces, msg_with_spaces)

        assert tracker.is_duplicate(conv_with_spaces, msg_with_spaces) is True

    def test_newlines_in_ids(self):
        """Test newlines in conversation_id and message_id."""
        tracker = DeduplicationTracker()
        conversation_id = "conv\n123"
        message_id = "msg\nid"

        tracker.mark_processed(conversation_id, message_id)

        assert tracker.is_duplicate(conversation_id, message_id) is True

    def test_null_character_in_ids(self):
        """Test null characters in IDs."""
        tracker = DeduplicationTracker()
        conversation_id = "conv\x00null"
        message_id = "msg\x00null"

        tracker.mark_processed(conversation_id, message_id)

        assert tracker.is_duplicate(conversation_id, message_id) is True

    def test_duplicate_detection_after_many_marks(self):
        """Test duplicate detection works after marking many messages."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"

        # Mark 1000 messages
        message_ids = [str(uuid.uuid4()) for _ in range(1000)]
        for msg_id in message_ids:
            tracker.mark_processed(conversation_id, msg_id)

        # Verify all are detected as duplicates
        for msg_id in message_ids:
            assert tracker.is_duplicate(conversation_id, msg_id) is True

        assert tracker.get_processed_count(conversation_id) == 1000

    def test_is_duplicate_for_unknown_conversation(self):
        """Test is_duplicate returns False for completely unknown conversation."""
        tracker = DeduplicationTracker()

        assert tracker.is_duplicate("unknown_conv", str(uuid.uuid4())) is False
        assert tracker.get_processed_count("unknown_conv") == 0

    def test_clearing_unknown_conversation(self):
        """Test clearing a conversation that was never created."""
        tracker = DeduplicationTracker()

        # Should not raise
        tracker.clear_conversation("never_created_conv")

    def test_hierarchical_conversation_ids(self):
        """Test hierarchical conversation IDs like root.subtask.subsubtask."""
        tracker = DeduplicationTracker()
        hierarchical_ids = [
            "root_123",
            "root_123.subtask_1",
            "root_123.subtask_1.subsubtask_1",
            "root.sub.subsub.subsubsub",
        ]
        message_id = str(uuid.uuid4())

        for conv_id in hierarchical_ids:
            tracker.mark_processed(conv_id, message_id)
            assert tracker.is_duplicate(conv_id, message_id) is True

    def test_identical_conversation_and_message_id(self):
        """Test when conversation_id and message_id are the same string."""
        tracker = DeduplicationTracker()
        same_id = str(uuid.uuid4())

        tracker.mark_processed(same_id, same_id)

        assert tracker.is_duplicate(same_id, same_id) is True
        assert tracker.get_processed_count(same_id) == 1

    def test_zero_message_tracking(self):
        """Test behavior with no messages tracked."""
        tracker = DeduplicationTracker()

        # All should work with empty state
        assert tracker.get_processed_count("any_conv") == 0
        assert tracker.is_duplicate("any_conv", "any_msg") is False
        tracker.clear_conversation("any_conv")  # Should not raise

    def test_is_duplicate_is_case_sensitive(self):
        """Test that duplicate detection is case-sensitive."""
        tracker = DeduplicationTracker()
        conversation_id = "conv_123"
        message_id = "MESSAGE_ID"

        tracker.mark_processed(conversation_id, message_id)

        # Different cases should not be duplicates
        assert tracker.is_duplicate(conversation_id, message_id) is True
        assert tracker.is_duplicate(conversation_id, "message_id") is False
        assert tracker.is_duplicate(conversation_id, "Message_Id") is False

    def test_is_duplicate_is_conversation_case_sensitive(self):
        """Test that duplicate detection is case-sensitive for conversation."""
        tracker = DeduplicationTracker()
        message_id = "msg_123"

        tracker.mark_processed("CONV_123", message_id)

        # Different cases should be different conversations
        assert tracker.is_duplicate("CONV_123", message_id) is True
        assert tracker.is_duplicate("conv_123", message_id) is False
        assert tracker.is_duplicate("Conv_123", message_id) is False
