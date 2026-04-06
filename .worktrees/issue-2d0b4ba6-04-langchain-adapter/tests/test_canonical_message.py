"""
Unit tests for CanonicalMessage Pydantic model.

Tests cover:
- Valid message creation with all required fields
- UUID v4 validation for message_id
- Intent enum validation (analyze, delegate, stream_result)
- Payload field validation per intent type
- JSON serialization/deserialization with timestamp conversion
- Metadata field defaults
- Error cases and validation failures
"""

import json
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from openclaw_gateway.canonical_message import (
    CRITICAL_FIELDS_BY_INTENT,
    CanonicalMessage,
)


class TestValidMessageCreation:
    """Test successful creation of valid messages."""

    def test_create_valid_analyze_message(self):
        """Create a valid analyze message with all required fields."""
        msg_id = str(uuid.uuid4())
        msg = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="langchain_analyzer",
            conversation_id="root_123",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc_abc123", "format": "pdf", "size": 1048576},
        )
        assert msg.message_id == msg_id
        assert msg.source_agent_id == "langchain_analyzer"
        assert msg.conversation_id == "root_123"
        assert msg.message_type == "request"
        assert msg.intent == "analyze"
        assert msg.payload["document_id"] == "doc_abc123"

    def test_create_valid_delegate_message(self):
        """Create a valid delegate message."""
        msg_id = str(uuid.uuid4())
        msg = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="autogpt_orchestrator",
            conversation_id="root_123.task_1",
            message_type="request",
            intent="delegate",
            payload={
                "target_agent": "spacy_ner",
                "task_description": "Extract named entities",
                "deadline": 1743948896.0,
            },
        )
        assert msg.intent == "delegate"
        assert msg.payload["target_agent"] == "spacy_ner"

    def test_create_valid_stream_result_message(self):
        """Create a valid stream_result message."""
        msg_id = str(uuid.uuid4())
        msg = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="event_stream",
            conversation_id="root_123.stream_1",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 0,
                "total_chunks": 10,
                "data": "chunk_data_here",
            },
        )
        assert msg.intent == "stream_result"
        assert msg.payload["chunk_index"] == 0

    def test_timestamp_default_creation(self):
        """Verify timestamp_utc is auto-populated with current time."""
        msg_id = str(uuid.uuid4())
        before = datetime.now(timezone.utc).timestamp()
        msg = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="test_agent",
            conversation_id="test_conv",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc_1", "format": "txt", "size": 100},
        )
        after = datetime.now(timezone.utc).timestamp()
        assert before <= msg.timestamp_utc <= after


class TestMessageIdValidation:
    """Test message_id UUID v4 validation."""

    def test_valid_uuid_v4_message_id(self):
        """Accept a valid UUID v4 message_id."""
        valid_uuid = str(uuid.uuid4())
        msg = CanonicalMessage(
            message_id=valid_uuid,
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
        )
        assert msg.message_id == valid_uuid

    def test_invalid_uuid_format(self):
        """Reject invalid UUID format."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id="not-a-uuid",
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "d1", "format": "txt", "size": 100},
            )
        assert "Invalid UUID-v4" in str(exc_info.value)

    def test_uuid_v1_rejected(self):
        """Reject UUID v1 (only v4 allowed)."""
        # Generate a v1 UUID
        v1_uuid = str(uuid.uuid1())
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=v1_uuid,
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "d1", "format": "txt", "size": 100},
            )
        assert "version" in str(exc_info.value).lower()

    def test_empty_message_id(self):
        """Reject empty message_id."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id="",
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "d1", "format": "txt", "size": 100},
            )
        assert "Invalid UUID-v4" in str(exc_info.value)


class TestIntentValidation:
    """Test intent enum validation."""

    @pytest.mark.parametrize("valid_intent", ["analyze", "delegate", "stream_result"])
    def test_valid_intent_values(self, valid_intent):
        """Accept all valid intent values."""
        msg_id = str(uuid.uuid4())
        if valid_intent == "analyze":
            payload = {"document_id": "d1", "format": "txt", "size": 100}
        elif valid_intent == "delegate":
            payload = {
                "target_agent": "agent1",
                "task_description": "task",
                "deadline": 1743948896.0,
            }
        else:  # stream_result
            payload = {"chunk_index": 0, "total_chunks": 1, "data": "data"}

        msg = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent=valid_intent,
            payload=payload,
        )
        assert msg.intent == valid_intent

    def test_invalid_intent_value(self):
        """Reject invalid intent values."""
        with pytest.raises(ValidationError):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="invalid_intent",
                payload={"document_id": "d1", "format": "txt", "size": 100},
            )


class TestPayloadFieldValidation:
    """Test critical field validation per intent type."""

    def test_analyze_missing_document_id(self):
        """Reject analyze message missing document_id."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"format": "pdf", "size": 100},
            )
        assert "Missing critical fields" in str(exc_info.value)
        assert "document_id" in str(exc_info.value)

    def test_analyze_missing_format(self):
        """Reject analyze message missing format."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc_1", "size": 100},
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_analyze_missing_size(self):
        """Reject analyze message missing size."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc_1", "format": "pdf"},
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_delegate_missing_target_agent(self):
        """Reject delegate message missing target_agent."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="delegate",
                payload={
                    "task_description": "task",
                    "deadline": 1743948896.0,
                },
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_delegate_missing_task_description(self):
        """Reject delegate message missing task_description."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="delegate",
                payload={
                    "target_agent": "agent1",
                    "deadline": 1743948896.0,
                },
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_delegate_missing_deadline(self):
        """Reject delegate message missing deadline."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="request",
                intent="delegate",
                payload={
                    "target_agent": "agent1",
                    "task_description": "task",
                },
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_stream_result_missing_chunk_index(self):
        """Reject stream_result message missing chunk_index."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="response",
                intent="stream_result",
                payload={"total_chunks": 10, "data": "chunk"},
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_stream_result_missing_total_chunks(self):
        """Reject stream_result message missing total_chunks."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="response",
                intent="stream_result",
                payload={"chunk_index": 0, "data": "chunk"},
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_stream_result_missing_data(self):
        """Reject stream_result message missing data."""
        with pytest.raises(ValidationError) as exc_info:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="test",
                conversation_id="conv_1",
                message_type="response",
                intent="stream_result",
                payload={"chunk_index": 0, "total_chunks": 10},
            )
        assert "Missing critical fields" in str(exc_info.value)

    def test_analyze_with_extra_fields(self):
        """Allow extra fields in payload beyond critical fields."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_1",
                "format": "pdf",
                "size": 100,
                "extra_field": "extra_value",
                "another_field": 123,
            },
        )
        assert msg.payload["extra_field"] == "extra_value"


class TestMetadataDefaults:
    """Test metadata field default population."""

    def test_empty_metadata_gets_defaults(self):
        """Empty metadata dict gets populated with defaults."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            metadata={},
        )
        assert msg.metadata["protocol_source"] == "unknown"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}

    def test_partial_metadata_gets_filled(self):
        """Partial metadata dict gets missing fields populated."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            metadata={"protocol_source": "langchain"},
        )
        assert msg.metadata["protocol_source"] == "langchain"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}

    def test_full_metadata_preserved(self):
        """Full metadata dict is preserved as-is."""
        custom_metadata = {
            "protocol_source": "custom",
            "conversation_chain": ["agent1", "agent2"],
            "vector_clock": {"agent1": 2, "agent2": 1},
        }
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            metadata=custom_metadata,
        )
        assert msg.metadata == custom_metadata

    def test_metadata_none_gets_defaults(self):
        """Metadata=None gets replaced with defaults."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            metadata=None,
        )
        assert msg.metadata["protocol_source"] == "unknown"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}


class TestJSONSerialization:
    """Test JSON serialization with timestamp conversion to ISO-8601."""

    def test_model_dump_json_produces_valid_json(self):
        """model_dump_json produces valid JSON string."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
        )
        json_str = msg.model_dump_json()
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed is not None

    def test_timestamp_serialized_as_iso8601(self):
        """timestamp_utc serialized as ISO-8601 in JSON."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            timestamp_utc=1743948896.789012,  # Use a specific timestamp
        )
        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)
        # Should be ISO-8601 with Z suffix
        assert isinstance(parsed["timestamp_utc"], str)
        assert "Z" in parsed["timestamp_utc"]
        assert "T" in parsed["timestamp_utc"]  # ISO-8601 has T separator

    def test_json_includes_all_fields(self):
        """JSON includes all required fields."""
        msg_id = str(uuid.uuid4())
        msg = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
        )
        json_str = msg.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["message_id"] == msg_id
        assert parsed["source_agent_id"] == "test_agent"
        assert parsed["conversation_id"] == "conv_1"
        assert parsed["message_type"] == "request"
        assert parsed["intent"] == "analyze"
        assert parsed["payload"]["document_id"] == "d1"
        assert "timestamp_utc" in parsed
        assert "metadata" in parsed


class TestJSONDeserialization:
    """Test JSON deserialization with ISO-8601 to unix float conversion."""

    def test_model_validate_json_valid_message(self):
        """model_validate_json accepts valid JSON string."""
        json_str = """{
            "message_id": "933ce97f-7574-4bdd-a900-206e3c67ec92",
            "source_agent_id": "test",
            "conversation_id": "conv_1",
            "message_type": "request",
            "intent": "analyze",
            "payload": {"document_id": "d1", "format": "txt", "size": 100},
            "metadata": {},
            "timestamp_utc": "2026-04-05T12:34:56.789012Z"
        }"""
        msg = CanonicalMessage.model_validate_json(json_str)
        assert msg.message_id == "933ce97f-7574-4bdd-a900-206e3c67ec92"
        assert msg.source_agent_id == "test"

    def test_iso8601_timestamp_converted_to_unix_float(self):
        """ISO-8601 timestamp in JSON converted to unix float internally."""
        json_str = """{
            "message_id": "b26b3ca3-a210-44ec-9c44-c5bbbaa5935c",
            "source_agent_id": "test",
            "conversation_id": "conv_1",
            "message_type": "request",
            "intent": "analyze",
            "payload": {"document_id": "d1", "format": "txt", "size": 100},
            "metadata": {},
            "timestamp_utc": "2026-04-05T12:34:56.789012Z"
        }"""
        msg = CanonicalMessage.model_validate_json(json_str)
        # timestamp_utc should be a float (unix timestamp)
        assert isinstance(msg.timestamp_utc, float)
        # The value should be in the right range (2026 is ~1750000000+ seconds)
        assert msg.timestamp_utc > 1743948896  # April 5, 2026 approx

    def test_deserialized_message_has_metadata_defaults(self):
        """Deserialized message gets metadata defaults if not provided."""
        json_str = """{
            "message_id": "31088e95-d636-4fcc-a531-d4f8da6df238",
            "source_agent_id": "test_agent",
            "conversation_id": "conv_1",
            "message_type": "request",
            "intent": "analyze",
            "payload": {"document_id": "d1", "format": "txt", "size": 100},
            "timestamp_utc": "2026-04-05T12:34:56.789012Z"
        }"""
        msg = CanonicalMessage.model_validate_json(json_str)
        assert msg.metadata["protocol_source"] == "unknown"
        assert "test_agent" in msg.metadata["conversation_chain"]

    def test_invalid_json_raises_error(self):
        """Invalid JSON raises ValidationError."""
        with pytest.raises(ValidationError):
            CanonicalMessage.model_validate_json("{invalid json}")

    def test_invalid_iso8601_timestamp_raises_error(self):
        """Invalid ISO-8601 timestamp raises error."""
        json_str = """{
            "message_id": "123e4567-e89b-12d3-a456-426614174000",
            "source_agent_id": "test",
            "conversation_id": "conv_1",
            "message_type": "request",
            "intent": "analyze",
            "payload": {"document_id": "d1", "format": "txt", "size": 100},
            "metadata": {},
            "timestamp_utc": "not-a-timestamp"
        }"""
        with pytest.raises(ValidationError):
            CanonicalMessage.model_validate_json(json_str)


class TestJSONRoundTrip:
    """Test round-trip JSON serialization and deserialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Message survives round-trip through JSON."""
        original = CanonicalMessage(
            message_id="933ce97f-7574-4bdd-a900-206e3c67ec92",
            source_agent_id="langchain_analyzer",
            conversation_id="root_123",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc_abc123", "format": "pdf", "size": 1048576},
            timestamp_utc=1743948896.789012,
        )

        json_str = original.model_dump_json()
        restored = CanonicalMessage.model_validate_json(json_str)

        # Check all fields match (allowing small float tolerance for timestamp)
        assert restored.message_id == original.message_id
        assert restored.source_agent_id == original.source_agent_id
        assert restored.conversation_id == original.conversation_id
        assert restored.message_type == original.message_type
        assert restored.intent == original.intent
        assert restored.payload == original.payload
        assert abs(restored.timestamp_utc - original.timestamp_utc) < 0.001

    def test_roundtrip_preserves_metadata(self):
        """Metadata survives round-trip."""
        metadata = {
            "protocol_source": "langchain",
            "conversation_chain": ["agent1", "agent2"],
            "vector_clock": {"agent1": 2, "agent2": 1},
        }
        original = CanonicalMessage(
            message_id="b26b3ca3-a210-44ec-9c44-c5bbbaa5935c",
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            metadata=metadata,
        )

        json_str = original.model_dump_json()
        restored = CanonicalMessage.model_validate_json(json_str)

        assert restored.metadata == original.metadata


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_payload(self):
        """Message with empty payload (but valid critical fields per intent)."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 0},
        )
        assert msg.payload["size"] == 0

    def test_large_timestamp_value(self):
        """Large timestamp values are handled correctly."""
        large_timestamp = 9999999999.999999
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "d1", "format": "txt", "size": 100},
            timestamp_utc=large_timestamp,
        )
        assert msg.timestamp_utc == large_timestamp

    def test_special_characters_in_fields(self):
        """Special characters in string fields are handled correctly."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test@agent.com",
            conversation_id="conv/root/sub",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc-123_456",
                "format": "application/pdf",
                "size": 100,
            },
        )
        assert "@" in msg.source_agent_id
        assert "/" in msg.conversation_id

    def test_unicode_in_payload(self):
        """Unicode characters in payload are preserved."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="conv_1",
            message_type="request",
            intent="stream_result",
            payload={
                "chunk_index": 0,
                "total_chunks": 1,
                "data": "Unicode: 你好世界 🚀 Ñoño",
            },
        )
        json_str = msg.model_dump_json()
        restored = CanonicalMessage.model_validate_json(json_str)
        assert "你好世界" in restored.payload["data"]


class TestCriticalFieldsConstant:
    """Test CRITICAL_FIELDS_BY_INTENT constant."""

    def test_constant_exists_and_has_all_intents(self):
        """CRITICAL_FIELDS_BY_INTENT contains all intent types."""
        assert "analyze" in CRITICAL_FIELDS_BY_INTENT
        assert "delegate" in CRITICAL_FIELDS_BY_INTENT
        assert "stream_result" in CRITICAL_FIELDS_BY_INTENT

    def test_analyze_critical_fields(self):
        """analyze intent has correct critical fields."""
        expected = {"document_id", "format", "size"}
        assert CRITICAL_FIELDS_BY_INTENT["analyze"] == expected

    def test_delegate_critical_fields(self):
        """delegate intent has correct critical fields."""
        expected = {"target_agent", "task_description", "deadline"}
        assert CRITICAL_FIELDS_BY_INTENT["delegate"] == expected

    def test_stream_result_critical_fields(self):
        """stream_result intent has correct critical fields."""
        expected = {"chunk_index", "total_chunks", "data"}
        assert CRITICAL_FIELDS_BY_INTENT["stream_result"] == expected
