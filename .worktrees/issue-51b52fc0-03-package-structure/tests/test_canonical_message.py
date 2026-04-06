"""Unit tests for CanonicalMessage Pydantic model.

Tests cover:
1. Valid message creation with all fields
2. UUID v4 validation with invalid inputs
3. Intent literal validation
4. Payload field validation per intent type (parametrized)
5. JSON round-trip with timestamp conversion
6. Metadata defaults auto-population
7. Edge cases (empty payload, special characters, boundary timestamps)
"""

import pytest
import json
import uuid
from datetime import datetime, timezone

from openclaw_gateway.canonical_message import (
    CanonicalMessage,
    CRITICAL_FIELDS_BY_INTENT
)


class TestCanonicalMessageCreation:
    """Test valid message creation and field access."""

    def test_create_analyze_message(self):
        """Test creating a valid analyze intent message."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_agent",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_abc123",
                "format": "pdf",
                "size": 1024
            },
            metadata={
                "protocol_source": "langchain",
                "conversation_chain": ["langchain_agent"],
                "vector_clock": {"langchain_agent": 1}
            },
            timestamp_utc=1743948896.789012
        )

        assert msg.message_id == msg.message_id
        assert msg.source_agent_id == "langchain_agent"
        assert msg.conversation_id == "conv_123"
        assert msg.message_type == "request"
        assert msg.intent == "analyze"
        assert msg.payload["document_id"] == "doc_abc123"
        assert msg.timestamp_utc == 1743948896.789012

    def test_create_delegate_message(self):
        """Test creating a valid delegate intent message."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="autogpt_agent",
            conversation_id="conv_456.subtask_1",
            message_type="request",
            intent="delegate",
            payload={
                "target_agent": "spacy_agent",
                "task_description": "Extract entities",
                "deadline": 1743948900.0
            }
        )

        assert msg.intent == "delegate"
        assert msg.payload["target_agent"] == "spacy_agent"

    def test_create_stream_result_message(self):
        """Test creating a valid stream_result intent message."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="spacy_agent",
            conversation_id="conv_789",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 5,
                "total_chunks": 10,
                "data": b"chunk_data_here"
            }
        )

        assert msg.intent == "stream_result"
        assert msg.payload["chunk_index"] == 5

    def test_all_eight_fields_present(self):
        """Test all 8 required fields are accessible."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_1",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            metadata={"protocol_source": "test"},
            timestamp_utc=123456.789
        )

        # Verify all 8 fields exist and are accessible
        assert hasattr(msg, 'message_id')
        assert hasattr(msg, 'source_agent_id')
        assert hasattr(msg, 'conversation_id')
        assert hasattr(msg, 'message_type')
        assert hasattr(msg, 'intent')
        assert hasattr(msg, 'payload')
        assert hasattr(msg, 'metadata')
        assert hasattr(msg, 'timestamp_utc')

    def test_metadata_default_population(self):
        """Test metadata fields are auto-populated with defaults when omitted."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            metadata={}  # Empty metadata
        )

        # Verify defaults are populated
        assert msg.metadata["protocol_source"] == "unknown"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}

    def test_metadata_none_default_population(self):
        """Test metadata defaults are populated even when metadata is not provided."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
            # metadata not provided
        )

        # Verify defaults are populated
        assert msg.metadata["protocol_source"] == "unknown"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}


class TestUUIDv4Validation:
    """Test UUID v4 validation on message_id."""

    def test_valid_uuidv4(self):
        """Test that valid UUID v4 is accepted."""
        valid_uuid = str(uuid.uuid4())
        msg = CanonicalMessage(
            message_id=valid_uuid,
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        assert msg.message_id == valid_uuid

    def test_invalid_uuid_not_uuidv4(self):
        """Test that UUID v3 is rejected."""
        # Create a UUID v1
        v1_uuid = str(uuid.uuid1())
        with pytest.raises(ValueError, match="UUID version"):
            CanonicalMessage(
                message_id=v1_uuid,
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    def test_invalid_uuid_malformed(self):
        """Test that malformed UUID string is rejected."""
        with pytest.raises(ValueError, match="Invalid UUID-v4"):
            CanonicalMessage(
                message_id="invalid-uuid",
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    def test_invalid_uuid_empty(self):
        """Test that empty message_id is rejected."""
        with pytest.raises(ValueError, match="empty"):
            CanonicalMessage(
                message_id="",
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    def test_invalid_uuid_random_string(self):
        """Test that random string is rejected as message_id."""
        with pytest.raises(ValueError, match="Invalid UUID-v4"):
            CanonicalMessage(
                message_id="not-a-uuid-at-all",
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    @pytest.mark.parametrize("invalid_uuid", [
        "123e4567-e89b-11d3-a456-426614174000",  # v1
        "123e4567-e89b-21d3-a456-426614174000",  # v2
        "123e4567-e89b-31d3-a456-426614174000",  # v3
        "123e4567-e89b-51d3-a456-426614174000",  # v5
    ])
    def test_invalid_uuid_versions(self, invalid_uuid):
        """Test that non-v4 UUIDs are rejected (parametrized)."""
        with pytest.raises(ValueError):
            CanonicalMessage(
                message_id=invalid_uuid,
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )


class TestIntentValidation:
    """Test intent literal validation."""

    @pytest.mark.parametrize("valid_intent", ["analyze", "delegate", "stream_result"])
    def test_valid_intents(self, valid_intent):
        """Test that all valid intents are accepted (parametrized)."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent=valid_intent,
            payload=self._get_payload_for_intent(valid_intent)
        )
        assert msg.intent == valid_intent

    def test_invalid_intent(self):
        """Test that invalid intent is rejected."""
        with pytest.raises(ValueError):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="invalid_intent",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    def test_empty_intent(self):
        """Test that empty intent is rejected."""
        with pytest.raises(ValueError):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="",  # type: ignore
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    @staticmethod
    def _get_payload_for_intent(intent: str) -> dict:
        """Helper to get valid payload for given intent."""
        if intent == "analyze":
            return {"document_id": "doc", "format": "pdf", "size": 100}
        elif intent == "delegate":
            return {"target_agent": "agent", "task_description": "task", "deadline": 123.0}
        elif intent == "stream_result":
            return {"chunk_index": 0, "total_chunks": 1, "data": "data"}
        return {}


class TestPayloadValidation:
    """Test payload field validation per intent type."""

    @pytest.mark.parametrize("intent,required_fields,payload", [
        (
            "analyze",
            {"document_id", "format", "size"},
            {"document_id": "doc_123", "format": "pdf", "size": 1024}
        ),
        (
            "delegate",
            {"target_agent", "task_description", "deadline"},
            {"target_agent": "spacy_agent", "task_description": "Extract entities", "deadline": 1743948900.0}
        ),
        (
            "stream_result",
            {"chunk_index", "total_chunks", "data"},
            {"chunk_index": 0, "total_chunks": 10, "data": "chunk_data"}
        ),
    ])
    def test_valid_payload_per_intent(self, intent, required_fields, payload):
        """Test valid payload for each intent type (parametrized)."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent=intent,
            payload=payload
        )
        assert msg.intent == intent
        assert all(field in msg.payload for field in required_fields)

    @pytest.mark.parametrize("intent,payload,missing_field", [
        ("analyze", {"document_id": "doc", "format": "pdf"}, "size"),
        ("analyze", {"document_id": "doc", "size": 100}, "format"),
        ("analyze", {"format": "pdf", "size": 100}, "document_id"),
        ("delegate", {"target_agent": "agent", "task_description": "task"}, "deadline"),
        ("delegate", {"target_agent": "agent", "deadline": 123.0}, "task_description"),
        ("delegate", {"task_description": "task", "deadline": 123.0}, "target_agent"),
        ("stream_result", {"chunk_index": 0, "total_chunks": 10}, "data"),
        ("stream_result", {"chunk_index": 0, "data": "x"}, "total_chunks"),
        ("stream_result", {"total_chunks": 10, "data": "x"}, "chunk_index"),
    ])
    def test_missing_critical_fields(self, intent, payload, missing_field):
        """Test that missing critical fields raise error (parametrized)."""
        with pytest.raises(ValueError, match="Missing critical fields"):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent=intent,
                payload=payload
            )

    def test_payload_with_extra_fields(self):
        """Test that payload can have extra fields beyond critical ones."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc",
                "format": "pdf",
                "size": 100,
                "extra_field": "extra_value",
                "another_field": 123
            }
        )
        assert msg.payload["document_id"] == "doc"
        assert msg.payload["extra_field"] == "extra_value"

    def test_empty_payload_for_analyze(self):
        """Test that analyze intent with empty payload fails."""
        with pytest.raises(ValueError, match="Missing critical fields"):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent",
                conversation_id="conv",
                message_type="request",
                intent="analyze",
                payload={}
            )


class TestJSONSerialization:
    """Test JSON serialization with timestamp conversion."""

    def test_json_serialization_iso8601_timestamp(self):
        """Test that model_dump_json produces ISO-8601 timestamp format."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            timestamp_utc=1743948896.789012
        )

        json_str = msg.model_dump_json()
        data = json.loads(json_str)

        # Verify timestamp is in ISO-8601 format with Z suffix
        assert data["timestamp_utc"].endswith("Z")
        assert "T" in data["timestamp_utc"]
        # The timestamp 1743948896.789012 corresponds to 2025-04-06
        assert "2025-04-06" in data["timestamp_utc"]

    def test_json_serialization_all_fields(self):
        """Test that JSON serialization includes all 8 fields."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            metadata={"protocol_source": "test"},
            timestamp_utc=1743948896.789012
        )

        json_str = msg.model_dump_json()
        data = json.loads(json_str)

        assert "message_id" in data
        assert "source_agent_id" in data
        assert "conversation_id" in data
        assert "message_type" in data
        assert "intent" in data
        assert "payload" in data
        assert "metadata" in data
        assert "timestamp_utc" in data

    def test_json_serialization_preserves_payload(self):
        """Test that JSON serialization preserves payload content."""
        payload = {
            "document_id": "doc_abc123",
            "format": "pdf",
            "size": 1048576,
            "extra": "data"
        }
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload=payload
        )

        json_str = msg.model_dump_json()
        data = json.loads(json_str)

        assert data["payload"] == payload


class TestJSONDeserialization:
    """Test JSON deserialization with timestamp conversion."""

    def test_json_deserialization_iso8601_to_unix(self):
        """Test that model_validate_json converts ISO-8601 to unix float."""
        # Use a valid v4 UUID
        v4_uuid = str(uuid.uuid4())
        json_str = f"""{{
            "message_id": "{v4_uuid}",
            "source_agent_id": "agent",
            "conversation_id": "conv",
            "message_type": "request",
            "intent": "analyze",
            "payload": {{"document_id": "doc", "format": "pdf", "size": 100}},
            "metadata": {{}},
            "timestamp_utc": "2026-04-05T12:34:56.789012Z"
        }}"""

        msg = CanonicalMessage.model_validate_json(json_str)

        # Verify timestamp is converted to unix float
        assert isinstance(msg.timestamp_utc, float)
        assert msg.timestamp_utc > 0

    def test_json_deserialization_round_trip(self):
        """Test that serialize → deserialize preserves fields."""
        original_msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_agent",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_abc123",
                "format": "pdf",
                "size": 1024
            },
            metadata={
                "protocol_source": "langchain"
            },
            timestamp_utc=1743948896.789012
        )

        # Serialize to JSON
        json_str = original_msg.model_dump_json()

        # Deserialize back
        deserialized_msg = CanonicalMessage.model_validate_json(json_str)

        # Verify key fields are preserved
        assert deserialized_msg.message_id == original_msg.message_id
        assert deserialized_msg.source_agent_id == original_msg.source_agent_id
        assert deserialized_msg.conversation_id == original_msg.conversation_id
        assert deserialized_msg.message_type == original_msg.message_type
        assert deserialized_msg.intent == original_msg.intent
        assert deserialized_msg.payload == original_msg.payload
        # Timestamp might have slight precision loss, so check approximate equality
        assert abs(deserialized_msg.timestamp_utc - original_msg.timestamp_utc) < 0.001

    def test_json_deserialization_with_unix_timestamp(self):
        """Test that model_validate_json handles unix float timestamps."""
        # Use a valid v4 UUID
        v4_uuid = str(uuid.uuid4())
        json_str = f"""{{
            "message_id": "{v4_uuid}",
            "source_agent_id": "agent",
            "conversation_id": "conv",
            "message_type": "request",
            "intent": "analyze",
            "payload": {{"document_id": "doc", "format": "pdf", "size": 100}},
            "metadata": {{}},
            "timestamp_utc": 1743948896.789012
        }}"""

        msg = CanonicalMessage.model_validate_json(json_str)
        assert msg.timestamp_utc == 1743948896.789012

    def test_json_deserialization_creates_valid_message(self):
        """Test that deserialized message is fully valid."""
        # Use a valid v4 UUID
        v4_uuid = str(uuid.uuid4())
        json_str = f"""{{
            "message_id": "{v4_uuid}",
            "source_agent_id": "test_agent",
            "conversation_id": "conv",
            "message_type": "request",
            "intent": "delegate",
            "payload": {{
                "target_agent": "other_agent",
                "task_description": "Do something",
                "deadline": 123456.0
            }},
            "metadata": {{}},
            "timestamp_utc": "2026-04-05T12:34:56Z"
        }}"""

        msg = CanonicalMessage.model_validate_json(json_str)

        # Verify all fields
        assert msg.message_id == v4_uuid
        assert msg.source_agent_id == "test_agent"
        assert msg.conversation_id == "conv"
        assert msg.message_type == "request"
        assert msg.intent == "delegate"
        assert msg.payload["target_agent"] == "other_agent"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_special_characters_in_fields(self):
        """Test that special characters are preserved in string fields."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent-with-dashes_and_underscores",
            conversation_id="conv/with/slashes.and.dots",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc with spaces & symbols!",
                "format": "pdf",
                "size": 100
            }
        )
        assert "dashes" in msg.source_agent_id
        assert "/" in msg.conversation_id

    def test_large_payload_values(self):
        """Test handling of large values in payload."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc",
                "format": "pdf",
                "size": 1_000_000_000  # 1 GB
            }
        )
        assert msg.payload["size"] == 1_000_000_000

    def test_boundary_timestamps(self):
        """Test edge case timestamps (epoch, far future)."""
        # Unix epoch
        msg_epoch = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            timestamp_utc=0.0
        )
        assert msg_epoch.timestamp_utc == 0.0

        # Far future (year 3000)
        msg_future = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            timestamp_utc=32503680000.0
        )
        assert msg_future.timestamp_utc == 32503680000.0

    def test_deeply_nested_payload(self):
        """Test deeply nested structures in payload."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc",
                "format": "pdf",
                "size": 100,
                "nested": {
                    "level1": {
                        "level2": {
                            "level3": "deep_value"
                        }
                    }
                }
            }
        )
        assert msg.payload["nested"]["level1"]["level2"]["level3"] == "deep_value"

    def test_payload_with_null_values(self):
        """Test payload with null/None values (as allowed extras)."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc",
                "format": "pdf",
                "size": 100,
                "optional_field": None
            }
        )
        assert msg.payload["optional_field"] is None


class TestCriticalFieldsConstant:
    """Test the CRITICAL_FIELDS_BY_INTENT constant."""

    def test_constant_exists(self):
        """Test that CRITICAL_FIELDS_BY_INTENT constant is defined."""
        assert CRITICAL_FIELDS_BY_INTENT is not None
        assert isinstance(CRITICAL_FIELDS_BY_INTENT, dict)

    def test_constant_has_all_intents(self):
        """Test that constant contains all three intent types."""
        assert "analyze" in CRITICAL_FIELDS_BY_INTENT
        assert "delegate" in CRITICAL_FIELDS_BY_INTENT
        assert "stream_result" in CRITICAL_FIELDS_BY_INTENT

    def test_constant_field_values(self):
        """Test that constant has correct critical fields."""
        assert CRITICAL_FIELDS_BY_INTENT["analyze"] == {"document_id", "format", "size"}
        assert CRITICAL_FIELDS_BY_INTENT["delegate"] == {"target_agent", "task_description", "deadline"}
        assert CRITICAL_FIELDS_BY_INTENT["stream_result"] == {"chunk_index", "total_chunks", "data"}

    def test_constant_is_exportable(self):
        """Test that constant can be imported from the module."""
        from openclaw_gateway.canonical_message import CRITICAL_FIELDS_BY_INTENT as imported
        assert imported is not None
        assert "analyze" in imported


class TestMessageTypeValidation:
    """Test message_type literal validation."""

    @pytest.mark.parametrize("valid_type", ["request", "response", "state_update"])
    def test_valid_message_types(self, valid_type):
        """Test that all valid message types are accepted (parametrized)."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id="conv",
            message_type=valid_type,
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        assert msg.message_type == valid_type

    def test_invalid_message_type(self):
        """Test that invalid message type is rejected."""
        with pytest.raises(ValueError):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent",
                conversation_id="conv",
                message_type="invalid_type",  # type: ignore
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )


class TestConversationIdFormats:
    """Test various conversation_id formats."""

    @pytest.mark.parametrize("conv_id", [
        "conv_123",
        "root_123.subtask_1",
        "root_123.subtask_1.subsubtask_2",
        "hierarchy/with/slashes",
        "uuid-like-id-with-dashes",
    ])
    def test_various_conversation_id_formats(self, conv_id):
        """Test that various conversation_id formats are accepted."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent",
            conversation_id=conv_id,
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        assert msg.conversation_id == conv_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
