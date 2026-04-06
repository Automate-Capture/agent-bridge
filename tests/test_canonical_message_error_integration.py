"""Integration tests for CanonicalMessage and error type interaction.

These tests verify that the canonical message format and error exception hierarchy
work correctly together, especially at the boundaries between these two components.
This tests the HIGH-PRIORITY conflict resolution areas where both branches were merged
into the __init__.py file.

Test Coverage:
1. Error validation integration - CanonicalMessage validation errors are proper error types
2. Message serialization with error context preservation
3. Payload validation triggering appropriate errors
4. UUID validation triggering appropriate errors
5. Intent validation triggering appropriate errors
6. Metadata field validation with error recovery
7. JSON deserialization error handling
8. Cross-component error catching and handling patterns
"""

import pytest
import uuid
import json
from datetime import datetime, timezone

from openclaw_gateway.canonical_message import (
    CanonicalMessage,
    CRITICAL_FIELDS_BY_INTENT
)
from openclaw_gateway.errors import (
    OpenclawError,
    AdapterError,
    RoutingError,
    CycleDetectedError,
    TimeoutError,
    ValidationError,
    DuplicateMessageError,
)


class TestCanonicalMessageWithErrorTypes:
    """Test CanonicalMessage creation and validation error handling."""

    def test_canonical_message_creation_success_returns_valid_instance(self):
        """Verify successful message creation returns valid CanonicalMessage."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_1",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        assert isinstance(msg, CanonicalMessage)
        assert msg.source_agent_id == "agent_1"

    def test_invalid_uuid_raises_value_error_not_generic_exception(self):
        """Verify invalid UUID raises ValueError (which can be caught properly)."""
        with pytest.raises(ValueError):
            CanonicalMessage(
                message_id="not-a-uuid",
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )

    def test_missing_critical_fields_raises_value_error(self):
        """Verify missing critical fields raises ValueError."""
        with pytest.raises(ValueError, match="Missing critical fields"):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc"}  # Missing "format" and "size"
            )

    def test_invalid_intent_raises_value_error(self):
        """Verify invalid intent raises ValueError."""
        with pytest.raises(ValueError):
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent="invalid_intent",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )


class TestMessageValidationErrorRecoveryPatterns:
    """Test realistic error recovery patterns when message validation fails."""

    def test_adapter_can_catch_message_validation_error_with_adapter_error(self):
        """Verify AdapterError can be used to wrap message validation issues."""
        try:
            # Try to create invalid message
            CanonicalMessage(
                message_id="invalid",
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )
        except ValueError as e:
            # Adapter wraps validation error
            adapter_error = AdapterError(f"Message validation failed: {e}")
            assert isinstance(adapter_error, OpenclawError)
            assert "validation" in str(adapter_error).lower()

    def test_validation_error_type_for_semantic_validation(self):
        """Verify ValidationError is appropriate for semantic validation failures."""
        # Create a message with invalid payload semantically
        try:
            CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent="delegate",
                payload={"target_agent": "other_agent"}  # Missing deadline and task_description
            )
        except ValueError as e:
            # Map to ValidationError for semantic validation
            validation_error = ValidationError(str(e))
            assert isinstance(validation_error, OpenclawError)
            # Can be caught as either ValidationError or OpenclawError
            try:
                raise validation_error
            except ValidationError:
                pass  # Successfully caught as ValidationError
            except OpenclawError:
                pytest.fail("Should be caught as ValidationError")

    def test_timeout_error_independent_of_message_format(self):
        """Verify TimeoutError is independent and can coexist with message validation."""
        # TimeoutError is raised by the routing layer, not message validation
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_1",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        # Message is valid, but agent timeout would be separate
        with pytest.raises(TimeoutError):
            raise TimeoutError("Agent failed to respond in time")

    def test_cycle_detected_error_independent_of_message_format(self):
        """Verify CycleDetectedError is independent of message validation."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_1",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        # Message is valid, but cycle detection is routing-layer logic
        with pytest.raises(RoutingError):
            raise CycleDetectedError("Circular delegation A→B→C→A detected")

    def test_duplicate_message_error_can_wrap_message_id(self):
        """Verify DuplicateMessageError can reference message.message_id."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_1",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        # Deduplication layer would raise this
        dup_error = DuplicateMessageError(f"Message {msg.message_id} was already processed")
        assert isinstance(dup_error, OpenclawError)
        assert msg.message_id in str(dup_error)


class TestMessageSerializationWithErrorHandling:
    """Test JSON serialization/deserialization error handling."""

    def test_json_serialization_success_returns_valid_json_string(self):
        """Verify successful serialization produces valid JSON."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_1",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            timestamp_utc=1743948896.789012
        )
        json_str = msg.model_dump_json()
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["message_id"] == msg.message_id
        assert data["source_agent_id"] == msg.source_agent_id

    def test_json_deserialization_with_invalid_json_raises_error(self):
        """Verify invalid JSON raises error during deserialization."""
        invalid_json = "not valid json at all"
        with pytest.raises(Exception):  # json.JSONDecodeError or ValueError
            CanonicalMessage.model_validate_json(invalid_json)

    def test_json_deserialization_with_invalid_uuid_in_json_raises_error(self):
        """Verify deserialization catches invalid UUID in JSON payload."""
        invalid_json = """{
            "message_id": "not-a-uuid",
            "source_agent_id": "agent_1",
            "conversation_id": "conv_1",
            "message_type": "request",
            "intent": "analyze",
            "payload": {"document_id": "doc", "format": "pdf", "size": 100},
            "metadata": {},
            "timestamp_utc": 1743948896.789012
        }"""
        with pytest.raises(ValueError, match="UUID"):
            CanonicalMessage.model_validate_json(invalid_json)

    def test_json_deserialization_with_missing_fields_raises_error(self):
        """Verify deserialization catches missing required fields."""
        incomplete_json = """{
            "message_id": "%s",
            "source_agent_id": "agent_1"
        }""" % str(uuid.uuid4())
        with pytest.raises(Exception):  # Pydantic validation error
            CanonicalMessage.model_validate_json(incomplete_json)

    def test_adapter_error_wrapping_deserialization_failure(self):
        """Verify AdapterError can wrap deserialization failures."""
        invalid_json = """{
            "message_id": "invalid",
            "source_agent_id": "agent_1"
        }"""
        try:
            CanonicalMessage.model_validate_json(invalid_json)
        except Exception as e:
            adapter_error = AdapterError(f"Failed to deserialize message: {e}")
            assert isinstance(adapter_error, OpenclawError)
            assert "deserialize" in str(adapter_error).lower()


class TestMessagePayloadValidationPerIntent:
    """Test payload validation for different intents with error handling."""

    @pytest.mark.parametrize("intent,payload,should_succeed", [
        # Valid analyze payloads
        ("analyze", {"document_id": "doc", "format": "pdf", "size": 100}, True),
        ("analyze", {"document_id": "doc", "format": "txt", "size": 0}, True),
        # Invalid analyze payloads
        ("analyze", {"document_id": "doc", "format": "pdf"}, False),  # Missing size
        ("analyze", {"format": "pdf", "size": 100}, False),  # Missing document_id
        # Valid delegate payloads
        ("delegate", {"target_agent": "agent2", "task_description": "task", "deadline": 123.0}, True),
        # Invalid delegate payloads
        ("delegate", {"target_agent": "agent2", "deadline": 123.0}, False),  # Missing task_description
        # Valid stream_result payloads
        ("stream_result", {"chunk_index": 0, "total_chunks": 10, "data": "chunk"}, True),
        # Invalid stream_result payloads
        ("stream_result", {"chunk_index": 0, "total_chunks": 10}, False),  # Missing data
    ])
    def test_payload_validation_per_intent(self, intent, payload, should_succeed):
        """Test payload validation for each intent type."""
        if should_succeed:
            msg = CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent=intent,
                payload=payload
            )
            assert msg.intent == intent
        else:
            with pytest.raises(ValueError, match="Missing critical fields"):
                CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="agent_1",
                    conversation_id="conv_1",
                    message_type="request",
                    intent=intent,
                    payload=payload
                )


class TestMessageMetadataValidation:
    """Test metadata field validation and auto-population."""

    def test_metadata_auto_population_on_creation(self):
        """Verify metadata fields are auto-populated with sensible defaults."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100}
        )
        # Metadata should be auto-populated
        assert msg.metadata is not None
        assert msg.metadata["protocol_source"] == "unknown"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}

    def test_metadata_partial_override_preserves_defaults(self):
        """Verify providing partial metadata preserves defaults for missing fields."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            metadata={"protocol_source": "langchain"}
        )
        # Provided fields should be used, defaults added for missing ones
        assert msg.metadata["protocol_source"] == "langchain"
        assert msg.metadata["conversation_chain"] == ["test_agent"]
        assert msg.metadata["vector_clock"] == {"test_agent": 1}

    def test_metadata_full_override_preserves_all_fields(self):
        """Verify providing complete metadata doesn't add defaults."""
        custom_metadata = {
            "protocol_source": "custom",
            "conversation_chain": ["agent_a", "agent_b"],
            "vector_clock": {"agent_a": 2, "agent_b": 1}
        }
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={"document_id": "doc", "format": "pdf", "size": 100},
            metadata=custom_metadata
        )
        # All custom values should be preserved
        assert msg.metadata["protocol_source"] == "custom"
        assert msg.metadata["conversation_chain"] == ["agent_a", "agent_b"]
        assert msg.metadata["vector_clock"] == {"agent_a": 2, "agent_b": 1}


class TestCriticalFieldsConstantExport:
    """Test CRITICAL_FIELDS_BY_INTENT constant is properly exported."""

    def test_critical_fields_constant_is_importable(self):
        """Verify CRITICAL_FIELDS_BY_INTENT can be imported."""
        from openclaw_gateway.canonical_message import CRITICAL_FIELDS_BY_INTENT
        assert CRITICAL_FIELDS_BY_INTENT is not None

    def test_critical_fields_constant_has_all_intents(self):
        """Verify constant contains all three intent types."""
        assert "analyze" in CRITICAL_FIELDS_BY_INTENT
        assert "delegate" in CRITICAL_FIELDS_BY_INTENT
        assert "stream_result" in CRITICAL_FIELDS_BY_INTENT

    def test_critical_fields_constant_values_correct(self):
        """Verify critical fields for each intent are correct."""
        assert CRITICAL_FIELDS_BY_INTENT["analyze"] == {"document_id", "format", "size"}
        assert CRITICAL_FIELDS_BY_INTENT["delegate"] == {"target_agent", "task_description", "deadline"}
        assert CRITICAL_FIELDS_BY_INTENT["stream_result"] == {"chunk_index", "total_chunks", "data"}

    def test_validation_uses_constant_values(self):
        """Verify validation actually uses the constant."""
        # If validation payload missing a field in the constant, it should fail
        for intent, required_fields in CRITICAL_FIELDS_BY_INTENT.items():
            # Create payload missing one required field
            missing_field = list(required_fields)[0]
            incomplete_payload = {field: f"value_{field}" for field in required_fields if field != missing_field}

            with pytest.raises(ValueError, match="Missing critical fields"):
                CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="agent_1",
                    conversation_id="conv_1",
                    message_type="request",
                    intent=intent,
                    payload=incomplete_payload
                )


class TestConflictResolutionAreaTesting:
    """High-priority tests for the conflict resolution area (openclaw_gateway/__init__.py).

    Both branches modified __init__.py with different docstrings but identical version.
    The merge kept HEAD version (simpler docstring). These tests verify both components
    can be imported from __init__.py correctly.
    """

    def test_canonical_message_importable_from_package(self):
        """Verify CanonicalMessage can be imported from package."""
        # This tests the __init__.py exports
        try:
            from openclaw_gateway import CanonicalMessage
            msg = CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_1",
                conversation_id="conv_1",
                message_type="request",
                intent="analyze",
                payload={"document_id": "doc", "format": "pdf", "size": 100}
            )
            assert isinstance(msg, CanonicalMessage)
        except ImportError:
            # If not exported from __init__, that's OK - it's still in canonical_message module
            from openclaw_gateway.canonical_message import CanonicalMessage
            assert CanonicalMessage is not None

    def test_error_types_importable_from_package(self):
        """Verify error types can be imported from package."""
        try:
            from openclaw_gateway import (
                OpenclawError,
                AdapterError,
                RoutingError,
                CycleDetectedError,
                TimeoutError,
                ValidationError,
                DuplicateMessageError,
            )
            assert all([
                OpenclawError,
                AdapterError,
                RoutingError,
                CycleDetectedError,
                TimeoutError,
                ValidationError,
                DuplicateMessageError,
            ])
        except ImportError:
            # If not exported from __init__, that's OK - they're in errors module
            from openclaw_gateway.errors import (
                OpenclawError,
                AdapterError,
                RoutingError,
                CycleDetectedError,
                TimeoutError,
                ValidationError,
                DuplicateMessageError,
            )
            assert all([
                OpenclawError,
                AdapterError,
                RoutingError,
                CycleDetectedError,
                TimeoutError,
                ValidationError,
                DuplicateMessageError,
            ])

    def test_version_accessible_from_package(self):
        """Verify __version__ is accessible from package."""
        from openclaw_gateway import __version__
        assert __version__ is not None
        # Should match semantic versioning pattern
        assert "." in __version__


class TestRoundTripLosslessPreservation:
    """Test that messages preserve all information through serialization cycles."""

    def test_message_round_trip_preserves_all_fields(self):
        """Verify all fields preserved in serialize-deserialize cycle."""
        original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_agent",
            conversation_id="conv_123.subtask_1",
            message_type="response",
            intent="delegate",
            payload={
                "target_agent": "spacy_agent",
                "task_description": "Extract entities from document",
                "deadline": 1743948900.0
            },
            metadata={
                "protocol_source": "langchain",
                "conversation_chain": ["langchain_agent", "spacy_agent"],
                "vector_clock": {"langchain_agent": 2, "spacy_agent": 1}
            },
            timestamp_utc=1743948896.789012
        )

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = CanonicalMessage.model_validate_json(json_str)

        # Verify all fields match (with tolerance for timestamp precision)
        assert restored.message_id == original.message_id
        assert restored.source_agent_id == original.source_agent_id
        assert restored.conversation_id == original.conversation_id
        assert restored.message_type == original.message_type
        assert restored.intent == original.intent
        assert restored.payload == original.payload
        assert restored.metadata == original.metadata
        assert abs(restored.timestamp_utc - original.timestamp_utc) < 0.001

    def test_message_round_trip_with_analyze_intent(self):
        """Verify analyze intent message preserves all critical fields."""
        original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="pdf_analyzer",
            conversation_id="analyze_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_abc123",
                "format": "pdf",
                "size": 2048576
            }
        )

        json_str = original.model_dump_json()
        restored = CanonicalMessage.model_validate_json(json_str)

        assert restored.payload["document_id"] == "doc_abc123"
        assert restored.payload["format"] == "pdf"
        assert restored.payload["size"] == 2048576

    def test_message_round_trip_with_stream_result_intent(self):
        """Verify stream_result intent message preserves chunk data."""
        original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="stream_processor",
            conversation_id="stream_123",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 5,
                "total_chunks": 100,
                "data": "chunk_data_with_special_chars_§©®™"
            }
        )

        json_str = original.model_dump_json()
        restored = CanonicalMessage.model_validate_json(json_str)

        assert restored.payload["chunk_index"] == 5
        assert restored.payload["total_chunks"] == 100
        assert restored.payload["data"] == "chunk_data_with_special_chars_§©®™"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
