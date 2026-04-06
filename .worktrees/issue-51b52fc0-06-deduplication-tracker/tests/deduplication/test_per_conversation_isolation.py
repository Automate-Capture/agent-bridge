"""Tests for per-conversation isolation."""

import pytest
import uuid
from openclaw_gateway import DeduplicationTracker


class TestPerConversationIsolation:
    """Test that same message_id in different conversations are allowed."""

    def test_same_message_id_different_conversations_allowed(self):
        """Test same message_id can exist in different conversations."""
        tracker = DeduplicationTracker()
        message_id = str(uuid.uuid4())
        conv_a = "conv_a"
        conv_b = "conv_b"

        # Mark same message_id in different conversations
        tracker.mark_processed(conv_a, message_id)
        tracker.mark_processed(conv_b, message_id)

        # Both should be marked
        assert tracker.is_duplicate(conv_a, message_id) is True
        assert tracker.is_duplicate(conv_b, message_id) is True

    @pytest.mark.parametrize("conv_id", ["conv_1", "conv_2", "conv_3"])
    def test_same_message_parametrized_conversations(self, conv_id):
        """Test same message_id across parametrized conversations."""
        tracker = DeduplicationTracker()
        message_id = str(uuid.uuid4())

        tracker.mark_processed(conv_id, message_id)

        assert tracker.is_duplicate(conv_id, message_id) is True

    def test_different_messages_different_conversations(self):
        """Test different messages in different conversations don't interfere."""
        tracker = DeduplicationTracker()
        message_id_a = str(uuid.uuid4())
        message_id_b = str(uuid.uuid4())
        conv_a = "conv_a"
        conv_b = "conv_b"

        tracker.mark_processed(conv_a, message_id_a)
        tracker.mark_processed(conv_b, message_id_b)

        # Cross-conversation should be False
        assert tracker.is_duplicate(conv_a, message_id_b) is False
        assert tracker.is_duplicate(conv_b, message_id_a) is False

    def test_concurrent_conversations_dont_interfere(self):
        """Test concurrent conversations don't interfere with each other."""
        tracker = DeduplicationTracker()

        # Simulate concurrent conversations
        conversations = {
            "conv_1": [str(uuid.uuid4()) for _ in range(5)],
            "conv_2": [str(uuid.uuid4()) for _ in range(5)],
            "conv_3": [str(uuid.uuid4()) for _ in range(5)],
        }

        # Mark all messages
        for conv_id, message_ids in conversations.items():
            for msg_id in message_ids:
                tracker.mark_processed(conv_id, msg_id)

        # Verify each conversation tracks only its own messages
        for conv_id, message_ids in conversations.items():
            assert tracker.get_processed_count(conv_id) == len(message_ids)

            for msg_id in message_ids:
                assert tracker.is_duplicate(conv_id, msg_id) is True

            # Verify messages from other conversations are not marked
            for other_conv_id, other_message_ids in conversations.items():
                if other_conv_id != conv_id:
                    for msg_id in other_message_ids:
                        assert tracker.is_duplicate(conv_id, msg_id) is False

    def test_three_conversations_isolation_explicit(self):
        """Explicit test: three conversations with shared message IDs."""
        tracker = DeduplicationTracker()
        # Use same message_id for all three conversations
        shared_message_id = str(uuid.uuid4())

        tracker.mark_processed("conv_1", shared_message_id)
        tracker.mark_processed("conv_2", shared_message_id)
        tracker.mark_processed("conv_3", shared_message_id)

        # All three should be marked in their respective conversations
        assert tracker.is_duplicate("conv_1", shared_message_id) is True
        assert tracker.is_duplicate("conv_2", shared_message_id) is True
        assert tracker.is_duplicate("conv_3", shared_message_id) is True

        # Each has exactly one processed message
        assert tracker.get_processed_count("conv_1") == 1
        assert tracker.get_processed_count("conv_2") == 1
        assert tracker.get_processed_count("conv_3") == 1

    def test_clear_conversation_isolates_from_others(self):
        """Test clearing one conversation doesn't affect others."""
        tracker = DeduplicationTracker()
        message_id_a = str(uuid.uuid4())
        message_id_b = str(uuid.uuid4())
        conv_a = "conv_a"
        conv_b = "conv_b"

        tracker.mark_processed(conv_a, message_id_a)
        tracker.mark_processed(conv_b, message_id_b)

        # Clear conv_a
        tracker.clear_conversation(conv_a)

        # conv_a should be cleared
        assert tracker.is_duplicate(conv_a, message_id_a) is False
        assert tracker.get_processed_count(conv_a) == 0

        # conv_b should be unaffected
        assert tracker.is_duplicate(conv_b, message_id_b) is True
        assert tracker.get_processed_count(conv_b) == 1
