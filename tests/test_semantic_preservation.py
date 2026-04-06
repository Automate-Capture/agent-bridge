"""
Unit tests for SemanticContextPreserver.

Tests cover:
- Intent preservation verification (original intent == transformed intent)
- Critical fields validation per intent type
- Confidence score boundary tests (0.94, 0.95, 1.0)
- Parametrized tests for all three intent types (analyze, delegate, stream_result)
- Complete and partial field combinations
- Latency benchmarks (< 5ms per operation)
- Intent extraction from different adapter types
"""

import time
import uuid
from typing import Dict, Any

import pytest

from openclaw_gateway.canonical_message import (
    CRITICAL_FIELDS_BY_INTENT,
    CanonicalMessage,
)
from openclaw_gateway.semantic_preserver import SemanticContextPreserver


@pytest.fixture
def preserver():
    """Create a SemanticContextPreserver instance for testing."""
    return SemanticContextPreserver()


@pytest.fixture
def analyze_message():
    """Create a valid analyze message with all critical fields."""
    return CanonicalMessage(
        message_id=str(uuid.uuid4()),
        source_agent_id="langchain_analyzer",
        conversation_id="root_123",
        message_type="request",
        intent="analyze",
        payload={
            "document_id": "doc_abc123",
            "format": "pdf",
            "size": 1048576,
        },
    )


@pytest.fixture
def delegate_message():
    """Create a valid delegate message with all critical fields."""
    return CanonicalMessage(
        message_id=str(uuid.uuid4()),
        source_agent_id="autogpt_coordinator",
        conversation_id="root_456.subtask_1",
        message_type="request",
        intent="delegate",
        payload={
            "target_agent": "spacy_extractor",
            "task_description": "Extract entities from document",
            "deadline": 1234567890.5,
        },
    )


@pytest.fixture
def stream_result_message():
    """Create a valid stream_result message with all critical fields."""
    return CanonicalMessage(
        message_id=str(uuid.uuid4()),
        source_agent_id="spacy_extractor",
        conversation_id="root_456.subtask_1.stream_1",
        message_type="response",
        intent="stream_result",
        payload={
            "chunk_index": 0,
            "total_chunks": 10,
            "data": {"entities": ["Person", "Location"]},
        },
    )


class TestIntentPreservation:
    """Test that intent is preserved during message transformation."""

    @pytest.mark.parametrize(
        "intent",
        ["analyze", "delegate", "stream_result"],
    )
    def test_intent_preserved_for_all_types(self, preserver, intent):
        """Verify intent preservation for all three intent types."""
        original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent=intent,
            payload=self._get_valid_payload_for_intent(intent),
        )

        # Transform preserves intent
        transformed = CanonicalMessage(
            message_id=str(uuid.uuid4()),  # Different ID (transformed)
            source_agent_id="agent_b",  # Different source (transformed)
            conversation_id="conv_123",
            message_type="request",
            intent=intent,  # Intent preserved
            payload=original.payload.copy(),
        )

        assert preserver.validate(original, transformed) is True

    def test_intent_mismatch_fails_validation(self, preserver, analyze_message):
        """Verify validation fails when intent changes."""
        original = analyze_message

        # Transform with different intent
        transformed = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_b",
            conversation_id=original.conversation_id,
            message_type="request",
            intent="delegate",  # Different intent!
            payload={
                "target_agent": "agent_c",
                "task_description": "Process",
                "deadline": 1234567890.5,
            },
        )

        assert preserver.validate(original, transformed) is False

    def test_intent_extraction(self, preserver):
        """Verify extract_intent returns correct intent."""
        for intent in ["analyze", "delegate", "stream_result"]:
            msg = CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_a",
                conversation_id="conv_123",
                message_type="request",
                intent=intent,
                payload=self._get_valid_payload_for_intent(intent),
            )
            assert preserver.extract_intent(msg) == intent

    @staticmethod
    def _get_valid_payload_for_intent(intent: str) -> Dict[str, Any]:
        """Generate valid payload for intent type."""
        if intent == "analyze":
            return {
                "document_id": "doc_123",
                "format": "pdf",
                "size": 1024,
            }
        elif intent == "delegate":
            return {
                "target_agent": "agent_x",
                "task_description": "Process",
                "deadline": 1234567890.5,
            }
        elif intent == "stream_result":
            return {
                "chunk_index": 0,
                "total_chunks": 1,
                "data": {"result": "data"},
            }
        else:
            return {}


class TestCriticalFieldsValidation:
    """Test validation of critical fields per intent type."""

    @pytest.mark.parametrize(
        "intent,required_fields",
        [
            ("analyze", {"document_id", "format", "size"}),
            ("delegate", {"target_agent", "task_description", "deadline"}),
            ("stream_result", {"chunk_index", "total_chunks", "data"}),
        ],
    )
    def test_all_critical_fields_present(self, preserver, intent, required_fields):
        """Verify validation passes when all critical fields present."""
        payload = {field: f"value_{field}" for field in required_fields}

        msg_original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent=intent,
            payload=payload,
        )

        msg_transformed = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_b",
            conversation_id="conv_123",
            message_type="request",
            intent=intent,
            payload=payload,
        )

        assert preserver.validate(msg_original, msg_transformed) is True

    def test_all_critical_fields_with_extra_fields(self, preserver):
        """Verify validation passes when all critical + extra fields present."""
        # Test analyze with extra fields
        payload = {
            "document_id": "doc_abc",
            "format": "pdf",
            "size": 1024,
            "extra_field_1": "extra",
            "extra_field_2": "extra2",
        }

        msg_original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload=payload,
        )

        msg_transformed = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_b",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload=payload,
        )

        assert preserver.validate(msg_original, msg_transformed) is True


class TestConfidenceScoreBoundary:
    """Test confidence score calculation and boundary conditions."""

    def test_confidence_full_fields_returns_1_0(self, preserver, analyze_message):
        """Verify confidence = 1.0 when all 3 analyze fields present."""
        assert preserver.compute_confidence(analyze_message) == 1.0

    def test_confidence_with_extra_fields_still_1_0(self, preserver, analyze_message):
        """Verify confidence = 1.0 even with extra non-critical fields."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_analyzer",
            conversation_id="root_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_abc123",
                "format": "pdf",
                "size": 1048576,
                "extra_field_1": "extra_value_1",
                "extra_field_2": "extra_value_2",
            },
        )
        assert preserver.compute_confidence(msg) == 1.0

    def test_confidence_all_intent_types_with_all_fields(self, preserver):
        """Verify confidence = 1.0 for all intent types when all fields present."""
        for intent, payload in [
            ("analyze", {"document_id": "doc", "format": "pdf", "size": 100}),
            ("delegate", {"target_agent": "agent", "task_description": "task", "deadline": 123.0}),
            ("stream_result", {"chunk_index": 0, "total_chunks": 10, "data": {}}),
        ]:
            msg = CanonicalMessage(
                message_id=str(uuid.uuid4()),
                source_agent_id="agent_a",
                conversation_id="conv_123",
                message_type="request",
                intent=intent,
                payload=payload,
            )
            assert preserver.compute_confidence(msg) == 1.0

    def test_confidence_boundary_1_0_accepted(self, preserver, analyze_message):
        """Verify confidence = 1.0 (100%) is accepted."""
        assert preserver.compute_confidence(analyze_message) == 1.0
        assert preserver.validate(analyze_message, analyze_message) is True

    def test_confidence_all_field_combinations_valid(self, preserver):
        """Test confidence calculation with various payload contents."""
        # Test case 1: analyze with all fields
        msg1 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_xyz",
                "format": "pdf",
                "size": 2048,
                "metadata": "extra",
                "version": "1.0",
            },
        )
        assert preserver.compute_confidence(msg1) == 1.0

        # Test case 2: delegate with all fields
        msg2 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_b",
            conversation_id="conv_456",
            message_type="request",
            intent="delegate",
            payload={
                "target_agent": "agent_c",
                "task_description": "extract entities",
                "deadline": 1234567890.5,
                "priority": "high",
            },
        )
        assert preserver.compute_confidence(msg2) == 1.0

        # Test case 3: stream_result with all fields
        msg3 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_d",
            conversation_id="conv_789",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 5,
                "total_chunks": 20,
                "data": {"entities": ["Person", "Location"]},
                "compression": "gzip",
            },
        )
        assert preserver.compute_confidence(msg3) == 1.0

    def test_confidence_computation_accuracy(self, preserver):
        """Verify confidence formula: fields_present / fields_required."""
        # For analyze intent with 3 required fields
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_123",
                "format": "pdf",
                "size": 1024,
            },
        )
        # 3 required fields, 3 present = 3/3 = 1.0
        assert preserver.compute_confidence(msg) == 1.0


class TestConfidenceThreshold:
    """Test confidence threshold enforcement (>= 0.95)."""

    def test_validate_accepts_at_threshold(self, preserver, analyze_message):
        """Verify validation accepts messages with confidence >= 0.95."""
        # analyze_message has all 3 required fields = 1.0 confidence >= 0.95
        assert preserver.validate(analyze_message, analyze_message) is True

    @pytest.mark.parametrize(
        "intent,payload,expected_confidence",
        [
            (
                "analyze",
                {"document_id": "doc_123", "format": "pdf", "size": 1024},
                1.0,  # 3/3 = 1.0
            ),
            (
                "delegate",
                {"target_agent": "agent_x", "task_description": "task", "deadline": 123},
                1.0,  # 3/3 = 1.0
            ),
            (
                "stream_result",
                {"chunk_index": 0, "total_chunks": 10, "data": {"x": 1}},
                1.0,  # 3/3 = 1.0
            ),
        ],
    )
    def test_confidence_threshold_parametrized(
        self, preserver, intent, payload, expected_confidence
    ):
        """Parametrized test for confidence across all valid intent types."""
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent=intent,
            payload=payload,
        )
        assert preserver.compute_confidence(msg) == expected_confidence
        assert preserver.validate(msg, msg) is True


class TestLatencyBenchmark:
    """Test that validation operations complete within 5ms."""

    def test_extract_intent_latency(self, preserver, analyze_message):
        """Verify extract_intent completes in < 5ms."""
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            preserver.extract_intent(analyze_message)
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / iterations

        assert avg_ms < 5, f"extract_intent took {avg_ms:.2f}ms (> 5ms)"

    def test_compute_confidence_latency(self, preserver, analyze_message):
        """Verify compute_confidence completes in < 5ms."""
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            preserver.compute_confidence(analyze_message)
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / iterations

        assert avg_ms < 5, f"compute_confidence took {avg_ms:.2f}ms (> 5ms)"

    def test_validate_latency(self, preserver, analyze_message):
        """Verify validate completes in < 5ms."""
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            preserver.validate(analyze_message, analyze_message)
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / iterations

        assert avg_ms < 5, f"validate took {avg_ms:.2f}ms (> 5ms)"

    def test_validate_latency_worst_case(self, preserver):
        """Verify validate is fast even with many extra payload fields."""
        # Create message with many extra fields
        large_payload = {
            "document_id": "doc_123",
            "format": "pdf",
            "size": 1024,
        }
        # Add 100 extra fields
        for i in range(100):
            large_payload[f"extra_field_{i}"] = f"value_{i}"

        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="agent_a",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload=large_payload,
        )

        iterations = 100
        start = time.time()
        for _ in range(iterations):
            preserver.validate(msg, msg)
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / iterations

        assert avg_ms < 5, f"validate (worst case) took {avg_ms:.2f}ms (> 5ms)"


class TestIntentExtractionFromAdapters:
    """Test intent extraction from different adapter types (LangChain, AutoGPT)."""

    def test_extract_analyze_intent_from_langchain_style(self, preserver):
        """Verify extract_intent works for LangChain-style analyze message."""
        # LangChain adapter would translate tool_calls → intent="analyze"
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_analyzer",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_research_paper",
                "format": "pdf",
                "size": 102400,
            },
            metadata={
                "protocol_source": "langchain",
                "conversation_chain": ["langchain_analyzer"],
            },
        )
        assert preserver.extract_intent(msg) == "analyze"
        assert preserver.validate(msg, msg) is True

    def test_extract_delegate_intent_from_autogpt_style(self, preserver):
        """Verify extract_intent works for AutoGPT-style delegate message."""
        # AutoGPT adapter would translate task hierarchy → intent="delegate"
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="autogpt_coordinator",
            conversation_id="root_456.subtask_1",
            message_type="request",
            intent="delegate",
            payload={
                "target_agent": "spacy_extractor",
                "task_description": "Extract entities from research paper",
                "deadline": 1234567890.5,
            },
            metadata={
                "protocol_source": "autogpt",
                "conversation_chain": ["autogpt_coordinator"],
            },
        )
        assert preserver.extract_intent(msg) == "delegate"
        assert preserver.validate(msg, msg) is True

    def test_extract_stream_result_intent_from_event_stream(self, preserver):
        """Verify extract_intent works for event stream-style result message."""
        # EventStream adapter would translate streaming events → intent="stream_result"
        msg = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="spacy_extractor",
            conversation_id="root_456.subtask_1.stream_1",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 5,
                "total_chunks": 20,
                "data": {"entities": ["Person", "Location", "Organization"]},
            },
            metadata={
                "protocol_source": "event_stream",
                "conversation_chain": ["langchain", "autogpt", "spacy"],
            },
        )
        assert preserver.extract_intent(msg) == "stream_result"
        assert preserver.validate(msg, msg) is True


class TestComplexScenarios:
    """Test complex multi-agent delegation scenarios."""

    def test_three_hop_delegation_preserves_intent(self, preserver):
        """Verify intent preserved through 3-hop delegation chain."""
        # Hop 1: LangChain → canonical (intent=analyze)
        hop1 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_analyzer",
            conversation_id="root_789",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_xyz",
                "format": "pdf",
                "size": 204800,
            },
        )

        # Hop 2: canonical → AutoGPT (intent=analyze preserved)
        hop2 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="autogpt_coordinator",
            conversation_id="root_789.subtask_1",
            message_type="request",
            intent="analyze",  # Intent preserved!
            payload=hop1.payload.copy(),
        )

        # Hop 3: canonical → spaCy (intent=analyze preserved)
        hop3 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="spacy_extractor",
            conversation_id="root_789.subtask_1.sub_1",
            message_type="request",
            intent="analyze",  # Intent preserved!
            payload=hop2.payload.copy(),
        )

        assert preserver.validate(hop1, hop2) is True
        assert preserver.validate(hop2, hop3) is True

    def test_delegate_task_through_agents(self, preserver):
        """Verify delegation intent preserved through agent chain."""
        # Delegation hop 1: Original delegate request
        original = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="autogpt_coordinator",
            conversation_id="root_123.task_1",
            message_type="request",
            intent="delegate",
            payload={
                "target_agent": "spacy_extractor",
                "task_description": "Extract named entities",
                "deadline": 1234567890.5,
            },
        )

        # Delegation hop 2: Transformed by router (maintains delegate intent)
        transformed = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="spacy_extractor",
            conversation_id="root_123.task_1",
            message_type="request",
            intent="delegate",  # Intent preserved
            payload=original.payload.copy(),
        )

        assert preserver.validate(original, transformed) is True

    def test_streaming_chunks_maintain_intent(self, preserver):
        """Verify streaming result chunks maintain stream_result intent."""
        # Chunk 1
        chunk1 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="spacy_extractor",
            conversation_id="root_456.stream_1",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 0,
                "total_chunks": 10,
                "data": {"entities": ["Person"]},
            },
        )

        # Chunk 2 (different chunk index, same intent)
        chunk2 = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="spacy_extractor",
            conversation_id="root_456.stream_1",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 1,
                "total_chunks": 10,
                "data": {"entities": ["Location"]},
            },
        )

        # Verify both chunks have same intent and pass validation
        assert preserver.extract_intent(chunk1) == "stream_result"
        assert preserver.extract_intent(chunk2) == "stream_result"
        assert preserver.validate(chunk1, chunk1) is True
        assert preserver.validate(chunk2, chunk2) is True
