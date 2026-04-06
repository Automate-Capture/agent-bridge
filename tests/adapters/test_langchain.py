"""
Unit tests for LangChainAdapter.

Tests cover:
- Single tool_call ingest with analyze/delegate/stream_result intents
- Intent extraction from function names
- Multiple tool_calls preservation
- Round-trip lossless preservation: tool_calls → canonical → tool_calls
- edge cases: empty tool_calls, malformed JSON, missing fields
- agent_id property returns consistent identifier
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from openclaw_gateway.adapters import LangChainAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class TestLangChainAdapterInitialization:
    """Test adapter initialization and properties."""

    def test_adapter_initialization_with_default_agent_id(self):
        """Test adapter initializes with default agent_id."""
        adapter = LangChainAdapter()
        assert adapter.agent_id == "langchain_agent"
        assert adapter.protocol_name == "langchain"

    def test_adapter_initialization_with_custom_agent_id(self):
        """Test adapter initializes with custom agent_id."""
        adapter = LangChainAdapter(agent_id="custom_langchain_123")
        assert adapter.agent_id == "custom_langchain_123"
        assert adapter.protocol_name == "langchain"

    def test_agent_id_property_returns_consistent_identifier(self):
        """Test agent_id property returns consistent value."""
        adapter = LangChainAdapter(agent_id="test_langchain")
        assert adapter.agent_id == "test_langchain"
        assert adapter.agent_id == adapter.agent_id  # Idempotent


class TestLangChainIngestSingleToolCall:
    """Test ingest() with single tool_call."""

    @pytest.mark.asyncio
    async def test_ingest_single_analyze_tool_call(self):
        """Test ingesting single tool_call with analyze function."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert isinstance(canonical, CanonicalMessage)
        assert canonical.intent == "analyze"
        assert canonical.source_agent_id == "langchain_agent"
        assert canonical.message_type == "request"
        assert canonical.payload["document_id"] == "doc_abc"
        assert canonical.metadata["protocol_source"] == "langchain"

    @pytest.mark.asyncio
    async def test_ingest_single_delegate_tool_call(self):
        """Test ingesting single tool_call with delegate function."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": "delegate_task",
                    "arguments": {
                        "target_agent": "spacy_ner",
                        "task_description": "Extract entities",
                        "deadline": 1234567890.0,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"
        assert canonical.payload["target_agent"] == "spacy_ner"

    @pytest.mark.asyncio
    async def test_ingest_single_stream_result_tool_call(self):
        """Test ingesting single tool_call with stream_result function."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_789",
                    "function": "stream_result",
                    "arguments": {
                        "chunk_index": 0,
                        "total_chunks": 10,
                        "data": "chunk_data",
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "stream_result"
        assert canonical.payload["chunk_index"] == 0


class TestLangChainIngestIntentExtraction:
    """Test intent extraction from function names."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "function_name,expected_intent,required_fields",
        [
            ("analyze_document", "analyze", {"document_id": "doc_1", "format": "pdf", "size": 100}),
            ("delegate_task", "delegate", {"target_agent": "agent_1", "task_description": "Task", "deadline": 1234567890.0}),
            ("stream_result", "stream_result", {"chunk_index": 0, "total_chunks": 1, "data": "data"}),
        ],
    )
    async def test_intent_extraction_from_function_name(
        self, function_name, expected_intent, required_fields
    ):
        """Test intent is correctly extracted from function name."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_test",
                    "function": function_name,
                    "arguments": required_fields,
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == expected_intent

    @pytest.mark.asyncio
    async def test_unknown_function_name_defaults_to_delegate(self):
        """Test unknown function name defaults to delegate intent."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_unknown",
                    "function": "unknown_function_xyz",
                    "arguments": {
                        "target_agent": "some_agent",
                        "task_description": "Do something",
                        "deadline": 1234567890.0,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"


class TestLangChainIngestMultipleToolCalls:
    """Test ingest() with multiple tool_calls."""

    @pytest.mark.asyncio
    async def test_ingest_multiple_tool_calls_preserves_all(self):
        """Test multiple tool_calls are preserved in canonical metadata."""
        adapter = LangChainAdapter()
        tool_calls = [
            {
                "id": "call_1",
                "function": "analyze_document",
                "arguments": {"document_id": "doc_1", "format": "pdf", "size": 100},
            },
            {
                "id": "call_2",
                "function": "delegate_task",
                "arguments": {
                    "target_agent": "agent_2",
                    "task_description": "Task 2",
                    "deadline": 1234567890.0,
                },
            },
            {
                "id": "call_3",
                "function": "stream_result",
                "arguments": {"chunk_index": 0, "total_chunks": 5, "data": "data"},
            },
        ]
        message = {"tool_calls": tool_calls}
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        # Verify all tool_calls preserved in metadata
        assert "tool_calls" in canonical.metadata
        assert len(canonical.metadata["tool_calls"]) == 3
        assert canonical.metadata["tool_calls"] == tool_calls

    @pytest.mark.asyncio
    async def test_ingest_multiple_tool_calls_uses_first_for_intent(self):
        """Test intent is extracted from first tool_call."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_1",
                        "format": "pdf",
                        "size": 100,
                    },
                },
                {
                    "id": "call_2",
                    "function": "delegate_task",
                    "arguments": {
                        "target_agent": "agent_2",
                        "task_description": "Task 2",
                        "deadline": 1234567890.0,
                    },
                },
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        # Intent from first tool_call
        assert canonical.intent == "analyze"


class TestLangChainIngestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_ingest_empty_tool_calls_array_raises_error(self):
        """Test empty tool_calls array raises ValueError."""
        adapter = LangChainAdapter()
        message = {"tool_calls": []}
        raw = json.dumps(message).encode("utf-8")

        with pytest.raises(ValueError, match="No tool_calls"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_missing_tool_calls_key_raises_error(self):
        """Test missing tool_calls key raises ValueError."""
        adapter = LangChainAdapter()
        message = {"other_key": "value"}
        raw = json.dumps(message).encode("utf-8")

        with pytest.raises(ValueError, match="No tool_calls"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_malformed_json_raises_error(self):
        """Test malformed JSON raises ValueError."""
        adapter = LangChainAdapter()
        raw = b"not valid json {"

        with pytest.raises(ValueError, match="Failed to decode"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_invalid_utf8_raises_error(self):
        """Test invalid UTF-8 raises ValueError."""
        adapter = LangChainAdapter()
        raw = b"\x80\x81\x82\x83"  # Invalid UTF-8

        with pytest.raises(ValueError, match="Failed to decode"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_missing_function_field(self):
        """Test missing function field defaults to 'unknown'."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    # missing "function" field
                    "arguments": {
                        "target_agent": "agent_1",
                        "task_description": "Task",
                        "deadline": 1234567890.0,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        # Unknown function defaults to delegate
        assert canonical.intent == "delegate"

    @pytest.mark.asyncio
    async def test_ingest_missing_arguments_field_raises_validation_error(self):
        """Test missing arguments field causes validation error."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "delegate_task",
                    # missing "arguments" field
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        # CanonicalMessage validation requires critical fields for delegate intent
        # so this should raise ValidationError
        with pytest.raises(Exception):  # Could be ValueError or ValidationError
            await adapter.ingest(raw)


class TestLangChainEgressSingleToolCall:
    """Test egress() with canonical message."""

    @pytest.mark.asyncio
    async def test_egress_canonical_to_langchain_format(self):
        """Test egress converts CanonicalMessage to tool_calls format."""
        adapter = LangChainAdapter()
        canonical = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_agent",
            conversation_id="conv_123",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_abc",
                "format": "pdf",
                "size": 1024,
            },
        )

        result = await adapter.egress(canonical)

        result_data = json.loads(result.decode("utf-8"))
        assert "tool_calls" in result_data
        assert len(result_data["tool_calls"]) == 1
        assert result_data["tool_calls"][0]["function"] == "analyze_document"
        assert result_data["tool_calls"][0]["arguments"]["document_id"] == "doc_abc"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "intent,expected_function",
        [
            ("analyze", "analyze_document"),
            ("delegate", "delegate_task"),
            ("stream_result", "stream_result"),
        ],
    )
    async def test_egress_intent_to_function_mapping(self, intent, expected_function):
        """Test intent is correctly mapped to function name."""
        adapter = LangChainAdapter()
        canonical = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="langchain_agent",
            conversation_id="conv_123",
            message_type="request",
            intent=intent,
            payload={
                "document_id": "doc_1",
                "format": "pdf",
                "size": 100,
            }
            if intent == "analyze"
            else {
                "target_agent": "agent_1",
                "task_description": "Task",
                "deadline": 1234567890.0,
            }
            if intent == "delegate"
            else {
                "chunk_index": 0,
                "total_chunks": 1,
                "data": "data",
            },
        )

        result = await adapter.egress(canonical)

        result_data = json.loads(result.decode("utf-8"))
        assert result_data["tool_calls"][0]["function"] == expected_function


class TestLangChainRoundTrip:
    """Test round-trip preservation: tool_calls → canonical → tool_calls."""

    @pytest.mark.asyncio
    async def test_round_trip_single_tool_call_lossless(self):
        """Test round-trip preserves single tool_call exactly."""
        adapter = LangChainAdapter()
        original_message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw_in = json.dumps(original_message).encode("utf-8")

        # Ingest
        canonical = await adapter.ingest(raw_in)

        # Egress
        raw_out = await adapter.egress(canonical)
        output_message = json.loads(raw_out.decode("utf-8"))

        # Verify structurally identical - all tool_calls should be preserved from metadata
        assert len(output_message["tool_calls"]) == len(original_message["tool_calls"])
        assert output_message["tool_calls"][0]["function"] == original_message["tool_calls"][0]["function"]
        assert output_message["tool_calls"][0]["arguments"] == original_message["tool_calls"][0]["arguments"]

    @pytest.mark.asyncio
    async def test_round_trip_multiple_tool_calls_lossless(self):
        """Test round-trip preserves multiple tool_calls exactly."""
        adapter = LangChainAdapter()
        original_tool_calls = [
            {
                "id": "call_1",
                "function": "analyze_document",
                "arguments": {"document_id": "doc_1", "format": "pdf", "size": 100},
            },
            {
                "id": "call_2",
                "function": "delegate_task",
                "arguments": {
                    "target_agent": "agent_2",
                    "task_description": "Task 2",
                    "deadline": 1234567890.0,
                },
            },
            {
                "id": "call_3",
                "function": "stream_result",
                "arguments": {"chunk_index": 0, "total_chunks": 5, "data": "data"},
            },
        ]
        original_message = {"tool_calls": original_tool_calls}
        raw_in = json.dumps(original_message).encode("utf-8")

        # Ingest
        canonical = await adapter.ingest(raw_in)

        # Egress
        raw_out = await adapter.egress(canonical)
        output_message = json.loads(raw_out.decode("utf-8"))

        # Verify all tool_calls preserved exactly from metadata
        assert len(output_message["tool_calls"]) == 3
        for i, original_tool_call in enumerate(original_tool_calls):
            assert output_message["tool_calls"][i] == original_tool_call

    @pytest.mark.asyncio
    async def test_round_trip_preserves_tool_call_ids(self):
        """Test round-trip preserves original tool_call IDs from metadata."""
        adapter = LangChainAdapter()
        original_ids = ["call_1", "call_2", "call_3"]
        original_message = {
            "tool_calls": [
                {
                    "id": original_ids[0],
                    "function": "analyze_document",
                    "arguments": {"document_id": "doc_1", "format": "pdf", "size": 100},
                },
                {
                    "id": original_ids[1],
                    "function": "delegate_task",
                    "arguments": {
                        "target_agent": "agent_2",
                        "task_description": "Task 2",
                        "deadline": 1234567890.0,
                    },
                },
                {
                    "id": original_ids[2],
                    "function": "stream_result",
                    "arguments": {"chunk_index": 0, "total_chunks": 1, "data": "data"},
                },
            ]
        }
        raw_in = json.dumps(original_message).encode("utf-8")

        # Ingest
        canonical = await adapter.ingest(raw_in)

        # Egress
        raw_out = await adapter.egress(canonical)
        output_message = json.loads(raw_out.decode("utf-8"))

        # All original IDs should be preserved
        for i, original_id in enumerate(original_ids):
            assert output_message["tool_calls"][i]["id"] == original_id


class TestLangChainConversationId:
    """Test conversation_id handling."""

    @pytest.mark.asyncio
    async def test_ingest_extracts_conversation_id_from_arguments(self):
        """Test conversation_id extracted from arguments if provided."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "conversation_id": "custom_conv_123",
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.conversation_id == "custom_conv_123"
        # conversation_id should be removed from payload
        assert "conversation_id" not in canonical.payload

    @pytest.mark.asyncio
    async def test_ingest_generates_conversation_id_when_missing(self):
        """Test conversation_id is generated when not provided."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.conversation_id.startswith("conv_")
        assert len(canonical.conversation_id) > 5  # conv_ + some hex


class TestLangChainMetadata:
    """Test metadata handling."""

    @pytest.mark.asyncio
    async def test_ingest_sets_protocol_source_metadata(self):
        """Test protocol_source metadata is set to 'langchain'."""
        adapter = LangChainAdapter()
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.metadata["protocol_source"] == "langchain"

    @pytest.mark.asyncio
    async def test_ingest_sets_conversation_chain_metadata(self):
        """Test conversation_chain metadata is set with source_agent_id."""
        adapter = LangChainAdapter(agent_id="my_langchain_agent")
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.metadata["conversation_chain"] == ["my_langchain_agent"]

    @pytest.mark.asyncio
    async def test_ingest_sets_vector_clock_metadata(self):
        """Test vector_clock metadata is initialized."""
        adapter = LangChainAdapter(agent_id="test_agent")
        message = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_abc",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.metadata["vector_clock"]["test_agent"] == 1

    @pytest.mark.asyncio
    async def test_ingest_preserves_tool_calls_in_metadata(self):
        """Test tool_calls are preserved in metadata."""
        adapter = LangChainAdapter()
        tool_calls = [
            {
                "id": "call_1",
                "function": "analyze_document",
                "arguments": {"document_id": "doc_1", "format": "pdf", "size": 100},
            },
        ]
        message = {"tool_calls": tool_calls}
        raw = json.dumps(message).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert "tool_calls" in canonical.metadata
        assert canonical.metadata["tool_calls"] == tool_calls
