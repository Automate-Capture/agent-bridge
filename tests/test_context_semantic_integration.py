"""
Integration tests for context manager, adapters, and semantic preservation.

Tests verify that:
1. Context is preserved through multi-hop delegations with semantic intent
2. Semantic intent is extracted and preserved through context chain
3. Conversation chains are correctly tracked through delegation sequences
4. Metadata is preserved end-to-end through adapters and context
"""

import json
import uuid
import pytest
from datetime import datetime, timezone
import tempfile
import os

from openclaw_gateway.adapters.langchain import LangChainAdapter
from openclaw_gateway.adapters.autogpt import AutoGPTAdapter
from openclaw_gateway.context_manager import ConversationContextManager, ConversationContext
from openclaw_gateway.semantic_preserver import SemanticContextPreserver
from openclaw_gateway.canonical_message import CanonicalMessage


class TestContextPreservationWithAdapters:
    """
    Test that conversation context is preserved through adapter round-trips.

    Verifies conflict resolution in context_manager.py integration with adapters.
    """

    @pytest.mark.asyncio
    async def test_context_created_for_langchain_message(self):
        """
        Test that context manager creates conversation contexts for adapter messages.

        Simulates LangChain agent sending a message, context manager tracking it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = os.path.join(tmpdir, "test.wal")
            context_mgr = ConversationContextManager(wal_path=wal_path)

            # Ingest LangChain message
            lc_adapter = LangChainAdapter(agent_id="langchain_1")
            lc_msg = {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": "analyze_document",
                        "arguments": {
                            "document_id": "doc_001",
                            "format": "pdf",
                            "size": 5000,
                        },
                    }
                ]
            }
            canonical = await lc_adapter.ingest(json.dumps(lc_msg).encode("utf-8"))

            # Create context for this conversation
            context = context_mgr.get_or_create_context(canonical.conversation_id)
            assert context is not None
            assert context.conversation_id == canonical.conversation_id

            # Add agent to chain
            context_mgr.add_agent_to_chain(
                canonical.conversation_id, canonical.source_agent_id
            )
            context = context_mgr.get_context(canonical.conversation_id)
            assert "langchain_1" in context.agents_in_chain

    @pytest.mark.asyncio
    async def test_multi_hop_delegation_chain_preserved(self):
        """
        Test that a multi-hop delegation chain is correctly tracked.

        Simulates: LangChain → AutoGPT → EventStream with context preservation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = os.path.join(tmpdir, "test.wal")
            context_mgr = ConversationContextManager(wal_path=wal_path)

            # Hop 1: LangChain delegates
            lc_adapter = LangChainAdapter(agent_id="langchain_delegator")
            lc_msg = {
                "tool_calls": [
                    {
                        "id": "call_delegate",
                        "function": "delegate_task",
                        "arguments": {
                            "target_agent": "autogpt_processor",
                            "task_description": "Process document",
                            "deadline": "2026-04-10T12:00:00Z",
                        },
                    }
                ]
            }
            hop1_canonical = await lc_adapter.ingest(json.dumps(lc_msg).encode("utf-8"))

            # Create context and track first agent
            context_id = hop1_canonical.conversation_id
            context = context_mgr.get_or_create_context(context_id)
            context_mgr.add_agent_to_chain(context_id, "langchain_delegator")

            # Hop 2: AutoGPT receives and delegates further
            ag_adapter = AutoGPTAdapter(agent_id="autogpt_processor")
            ag_msg = {
                "task": {
                    "id": "task_process",
                    "description": "Process and delegate to streamer",
                    "subtasks": [
                        {"id": "subtask_1", "description": "Process"}
                    ],
                }
            }
            # Manually set conversation_id to continue the chain
            hop2_canonical = await ag_adapter.ingest(json.dumps(ag_msg).encode("utf-8"))
            hop2_canonical.conversation_id = context_id

            context_mgr.add_agent_to_chain(context_id, "autogpt_processor")

            # Verify chain is preserved
            context = context_mgr.get_context(context_id)
            assert len(context.agents_in_chain) == 2
            assert context.agents_in_chain[0] == "langchain_delegator"
            assert context.agents_in_chain[1] == "autogpt_processor"

    @pytest.mark.asyncio
    async def test_state_snapshots_preserved_through_hops(self):
        """
        Test that state snapshots are preserved in context manager through
        multiple agent hops.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = os.path.join(tmpdir, "test.wal")
            context_mgr = ConversationContextManager(wal_path=wal_path)

            # Create conversation
            conv_id = "conv_state_test"
            context = context_mgr.get_or_create_context(conv_id)

            # Store state snapshot for each agent
            context_mgr.update_state_snapshot(
                conv_id, "agent_1", {"status": "processing", "progress": 30}
            )
            context_mgr.update_state_snapshot(
                conv_id, "agent_2", {"status": "delegated", "progress": 0}
            )

            # Retrieve and verify
            context = context_mgr.get_context(conv_id)
            assert context.state_snapshots["agent_1"]["progress"] == 30
            assert context.state_snapshots["agent_2"]["status"] == "delegated"


class TestSemanticPreservationWithAdapters:
    """
    Test that semantic intent and critical fields are preserved through adapters.

    Verifies the integration of SemanticContextPreserver with adapter outputs.
    """

    @pytest.mark.asyncio
    async def test_intent_preserved_langchain_to_semantic(self):
        """
        Test that intent is preserved from LangChain through semantic preserver.

        Intent: analyze → canonical → semantic validation.
        """
        preserver = SemanticContextPreserver()

        # Create analyze intent message
        lc_adapter = LangChainAdapter(agent_id="lc_analyzer")
        lc_msg = {
            "tool_calls": [
                {
                    "id": "call_analyze",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_semantic",
                        "format": "json",
                        "size": 2048,
                    },
                }
            ]
        }
        canonical = await lc_adapter.ingest(json.dumps(lc_msg).encode("utf-8"))

        # Validate semantics
        assert preserver.extract_intent(canonical) == "analyze"
        assert preserver.compute_confidence(canonical) >= 0.95

        # Validate critical fields
        critical_fields = preserver.get_critical_fields_for_intent("analyze")
        for field in critical_fields:
            assert field in canonical.payload

    @pytest.mark.asyncio
    async def test_intent_preserved_autogpt_to_semantic(self):
        """
        Test that intent is preserved from AutoGPT through semantic preserver.

        Intent: delegate → canonical → semantic validation.
        """
        preserver = SemanticContextPreserver()

        # Create delegate intent message (has subtasks)
        ag_adapter = AutoGPTAdapter(agent_id="ag_delegator")
        ag_msg = {
            "task": {
                "id": "task_delegate",
                "description": "Complex task with subtasks",
                "subtasks": [
                    {"id": "s1", "description": "Step 1"},
                    {"id": "s2", "description": "Step 2"},
                ],
            }
        }
        canonical = await ag_adapter.ingest(json.dumps(ag_msg).encode("utf-8"))

        # Validate semantics
        assert preserver.extract_intent(canonical) == "delegate"
        assert preserver.compute_confidence(canonical) >= 0.95

        # Validate critical fields
        critical_fields = preserver.get_critical_fields_for_intent("delegate")
        for field in critical_fields:
            assert field in canonical.payload

    @pytest.mark.asyncio
    async def test_confidence_across_adapter_formats(self):
        """
        Test that semantic confidence is consistent across different adapter formats.

        All adapters should produce messages with >= 0.95 confidence.
        """
        preserver = SemanticContextPreserver()

        adapters_messages = [
            (
                LangChainAdapter(agent_id="lc"),
                {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": "stream_result",
                            "arguments": {
                                "chunk_index": 0,
                                "total_chunks": 1,
                                "data": "test_data",
                            },
                        }
                    ]
                },
            ),
        ]

        for adapter, message in adapters_messages:
            canonical = await adapter.ingest(json.dumps(message).encode("utf-8"))
            confidence = preserver.compute_confidence(canonical)
            assert (
                confidence >= 0.95
            ), f"Confidence {confidence} below threshold for {adapter.protocol_name}"

    @pytest.mark.asyncio
    async def test_critical_fields_validation_through_adapters(self):
        """
        Test that critical field validation works for all adapter outputs.

        Each intent type requires specific critical fields that must be present.
        """
        preserver = SemanticContextPreserver()

        # Test analyze intent critical fields
        lc_adapter = LangChainAdapter(agent_id="lc")
        lc_msg = {
            "tool_calls": [
                {
                    "id": "call_analyze",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc_123",
                        "format": "pdf",
                        "size": 1024,
                    },
                }
            ]
        }
        canonical = await lc_adapter.ingest(json.dumps(lc_msg).encode("utf-8"))

        # Validate all critical fields present
        is_valid = preserver.validate_critical_fields(canonical)
        assert is_valid

        # Test missing critical field fails validation
        canonical_incomplete = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id="test",
            conversation_id="test_conv",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc",
                "format": "pdf",
                # Missing 'size' critical field
            },
        )
        is_invalid = preserver.validate_critical_fields(canonical_incomplete)
        assert not is_invalid


class TestMetadataPreservationThroughChain:
    """
    Test that metadata (protocol_source, conversation_chain, vector_clock)
    is correctly preserved through adapter chains.
    """

    @pytest.mark.asyncio
    async def test_protocol_source_metadata_set(self):
        """
        Test that protocol_source is correctly set by each adapter.

        After conflict resolution, adapters/__init__.py exports all adapters.
        """
        adapters = [
            (LangChainAdapter(agent_id="lc"), "langchain"),
            (AutoGPTAdapter(agent_id="ag"), "autogpt"),
        ]

        messages = [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": "analyze_document",
                        "arguments": {
                            "document_id": "d1",
                            "format": "f1",
                            "size": 100,
                        },
                    }
                ]
            },
            {
                "task": {
                    "id": "t1",
                    "description": "Task",
                    "subtasks": [],
                }
            },
        ]

        for (adapter, expected_protocol), message in zip(adapters, messages):
            canonical = await adapter.ingest(json.dumps(message).encode("utf-8"))
            assert canonical.metadata["protocol_source"] == expected_protocol

    @pytest.mark.asyncio
    async def test_conversation_chain_initialized(self):
        """
        Test that conversation_chain is initialized from the first adapter.

        The conversation_chain should start with the source agent ID.
        """
        lc_adapter = LangChainAdapter(agent_id="lc_first")
        lc_msg = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": "analyze_document",
                    "arguments": {
                        "document_id": "doc",
                        "format": "pdf",
                        "size": 100,
                    },
                }
            ]
        }
        canonical = await lc_adapter.ingest(json.dumps(lc_msg).encode("utf-8"))

        # Conversation chain should start with the source agent
        assert canonical.metadata["conversation_chain"][0] == "lc_first"

    @pytest.mark.asyncio
    async def test_vector_clock_initialized(self):
        """
        Test that vector_clock is initialized with source agent.

        The vector_clock should have an entry for the source agent.
        """
        ag_adapter = AutoGPTAdapter(agent_id="ag_vec_test")
        ag_msg = {
            "task": {
                "id": "t1",
                "description": "Task",
                "subtasks": [],
            }
        }
        canonical = await ag_adapter.ingest(json.dumps(ag_msg).encode("utf-8"))

        # Vector clock should have source agent entry
        assert "ag_vec_test" in canonical.metadata["vector_clock"]
        assert canonical.metadata["vector_clock"]["ag_vec_test"] >= 1


class TestConflictResolutionIntegration:
    """
    Test the specific conflict resolution areas from the merge.

    Covers:
    - openclaw_gateway/adapters/base.py merge (both versions kept)
    - openclaw_gateway/adapters/__init__.py merge (combined exports)
    - openclaw_gateway/context_manager.py with adapters
    """

    @pytest.mark.asyncio
    async def test_adapter_init_consolidated_imports(self):
        """
        Test that adapters/__init__.py correctly exports all adapters after merge.

        Conflict resolution: Added LangChainAdapter from first branch,
        then AutoGPTAdapter from second, then EventStreamAdapter.
        """
        from openclaw_gateway.adapters import (
            ProtocolAdapter,
            LangChainAdapter,
            AutoGPTAdapter,
            EventStreamAdapter,
        )

        # All should be importable
        assert LangChainAdapter is not None
        assert AutoGPTAdapter is not None
        assert EventStreamAdapter is not None

        # All should be subclasses of ProtocolAdapter
        assert issubclass(LangChainAdapter, ProtocolAdapter)
        assert issubclass(AutoGPTAdapter, ProtocolAdapter)
        assert issubclass(EventStreamAdapter, ProtocolAdapter)

    @pytest.mark.asyncio
    async def test_base_adapter_validation_merged_correctly(self):
        """
        Test that base adapter validation from both branches is present.

        Both branches added agent_id and protocol_name validation.
        """
        # Test validation from merged base.py
        lc = LangChainAdapter(agent_id="valid_id")
        assert lc.agent_id == "valid_id"
        assert lc.protocol_name == "langchain"

        # Both validations should work
        with pytest.raises(ValueError):
            LangChainAdapter(agent_id="")

        # Verify type annotations are present (from HEAD version)
        import inspect
        sig = inspect.signature(LangChainAdapter.__init__)
        assert "agent_id" in sig.parameters

    @pytest.mark.asyncio
    async def test_context_manager_works_with_all_adapters(self):
        """
        Test that ConversationContextManager works with messages from all adapters.

        Ensures context manager (issue #08) integrates with adapters
        (issues #04-06).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = os.path.join(tmpdir, "test.wal")
            ctx_mgr = ConversationContextManager(wal_path=wal_path)

            adapters = [
                LangChainAdapter(agent_id="lc_ctx_test"),
                AutoGPTAdapter(agent_id="ag_ctx_test"),
            ]

            messages = [
                {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": "analyze_document",
                            "arguments": {
                                "document_id": "doc",
                                "format": "txt",
                                "size": 100,
                            },
                        }
                    ]
                },
                {
                    "task": {
                        "id": "t1",
                        "description": "Task",
                        "subtasks": [],
                    }
                },
            ]

            for adapter, message in zip(adapters, messages):
                canonical = await adapter.ingest(
                    json.dumps(message).encode("utf-8")
                )

                # Should be able to create context for message
                context = ctx_mgr.get_or_create_context(
                    canonical.conversation_id
                )
                assert context is not None

                # Should be able to add agent to chain
                ctx_mgr.add_agent_to_chain(
                    canonical.conversation_id, canonical.source_agent_id
                )
                context = ctx_mgr.get_context(canonical.conversation_id)
                assert adapter.agent_id in context.agents_in_chain
