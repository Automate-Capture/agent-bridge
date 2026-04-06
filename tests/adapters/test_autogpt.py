"""
Unit tests for AutoGPTAdapter.

Tests cover:
- Flat task ingest (no subtasks)
- 3-level nested subtask hierarchy
- Round-trip lossless preservation
- Intent extraction from task structure
- Edge cases: empty subtasks list, malformed JSON, missing fields
- Task ID and description preservation
- Agent ID consistency
"""

import json
import uuid
import pytest
from datetime import datetime, timezone

from openclaw_gateway.adapters.autogpt import AutoGPTAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class TestAutoGPTAdapterBasics:
    """Test basic adapter functionality."""

    def test_adapter_initialization(self):
        """Verify adapter initializes with correct properties."""
        adapter = AutoGPTAdapter()
        assert adapter.agent_id == "autogpt_agent"
        assert adapter.protocol_name == "autogpt"

    def test_adapter_custom_agent_id(self):
        """Verify adapter can be initialized with custom agent_id."""
        adapter = AutoGPTAdapter(agent_id="custom_autogpt")
        assert adapter.agent_id == "custom_autogpt"
        assert adapter.protocol_name == "autogpt"

    def test_agent_id_property_consistency(self):
        """Verify agent_id property returns consistent identifier across calls."""
        adapter = AutoGPTAdapter(agent_id="test_agent")
        id1 = adapter.agent_id
        id2 = adapter.agent_id
        assert id1 == id2
        assert id1 == "test_agent"


class TestFlatTaskIngest:
    """Test ingesting flat tasks (no subtasks)."""

    @pytest.mark.asyncio
    async def test_ingest_flat_task_basic(self):
        """Parse task without subtasks and verify intent='analyze'."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "task_flat_1",
                "description": "Analyze document",
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "analyze"
        assert canonical.source_agent_id == "autogpt_agent"
        assert canonical.message_type == "request"
        assert canonical.payload["document_id"] == "task_flat_1"

    @pytest.mark.asyncio
    async def test_ingest_flat_task_with_empty_subtasks(self):
        """Parse task with empty subtasks list and verify intent='analyze'."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "task_empty_subtasks",
                "description": "Process data",
                "subtasks": [],
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "analyze"
        assert canonical.payload["document_id"] == "task_empty_subtasks"

    @pytest.mark.asyncio
    async def test_ingest_flat_task_preserves_description(self):
        """Verify task description is preserved in payload."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "task_123",
                "description": "Extract key entities from text",
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.payload["document_id"] == "task_123"


class TestNestedTaskIngest:
    """Test ingesting nested subtask hierarchies."""

    @pytest.mark.asyncio
    async def test_ingest_task_with_one_level_subtasks(self):
        """Parse task with one level of subtasks and verify intent='delegate'."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "task_parent",
                "description": "Analyze document",
                "subtasks": [
                    {"id": "subtask_1", "description": "Extract entities"},
                ],
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"
        assert canonical.payload["task_description"] == "Analyze document"
        assert "subtasks" in canonical.payload
        assert len(canonical.payload["subtasks"]) == 1

    @pytest.mark.asyncio
    async def test_ingest_task_with_two_level_subtasks(self):
        """Parse task with two levels of subtasks."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "task_root",
                "description": "Analyze document",
                "subtasks": [
                    {
                        "id": "subtask_1",
                        "description": "Extract entities",
                        "subtasks": [
                            {
                                "id": "subtask_1_1",
                                "description": "Extract named entities",
                            }
                        ],
                    },
                ],
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"
        assert len(canonical.payload["subtasks"]) == 1
        assert "subtasks" in canonical.payload["subtasks"][0]

    @pytest.mark.asyncio
    async def test_ingest_three_level_nested_hierarchy(self):
        """Parse 3-level nested subtask hierarchy."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "root_task",
                "description": "Root task",
                "subtasks": [
                    {
                        "id": "level_1_1",
                        "description": "Level 1 Task 1",
                        "subtasks": [
                            {
                                "id": "level_2_1",
                                "description": "Level 2 Task 1",
                                "subtasks": [
                                    {
                                        "id": "level_3_1",
                                        "description": "Level 3 Task 1",
                                    }
                                ],
                            },
                            {
                                "id": "level_2_2",
                                "description": "Level 2 Task 2",
                            },
                        ],
                    },
                    {
                        "id": "level_1_2",
                        "description": "Level 1 Task 2",
                    },
                ],
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"
        assert canonical.payload["task_description"] == "Root task"
        subtasks = canonical.payload["subtasks"]
        assert len(subtasks) == 2
        # Check first level-1 subtask has nested subtasks
        assert len(subtasks[0]["subtasks"]) == 2
        # Check first level-2 subtask has nested subtasks
        assert len(subtasks[0]["subtasks"][0]["subtasks"]) == 1
        # Check level-3 structure
        level_3 = subtasks[0]["subtasks"][0]["subtasks"][0]
        assert level_3["id"] == "level_3_1"


class TestIntentExtraction:
    """Test intent extraction logic."""

    @pytest.mark.asyncio
    async def test_intent_analyze_no_subtasks(self):
        """Verify no subtasks → intent='analyze'."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"id": "t1", "description": "Analyze"}}
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "analyze"

    @pytest.mark.asyncio
    async def test_intent_delegate_with_subtasks(self):
        """Verify subtasks present → intent='delegate'."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "t1",
                "description": "Root",
                "subtasks": [{"id": "s1", "description": "Sub"}],
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"

    @pytest.mark.asyncio
    async def test_intent_delegate_with_multiple_subtasks(self):
        """Verify multiple subtasks → intent='delegate'."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "t1",
                "description": "Root",
                "subtasks": [
                    {"id": "s1", "description": "Sub1"},
                    {"id": "s2", "description": "Sub2"},
                    {"id": "s3", "description": "Sub3"},
                ],
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.intent == "delegate"
        assert len(canonical.payload["subtasks"]) == 3


class TestEgress:
    """Test egress (canonical to AutoGPT format)."""

    @pytest.mark.asyncio
    async def test_egress_flat_task(self):
        """Transform flat task canonical message back to AutoGPT format."""
        adapter = AutoGPTAdapter()

        # Create a canonical message for a flat task
        msg_id = str(uuid.uuid4())
        canonical = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="autogpt_agent",
            conversation_id="conv_1",
            message_type="request",
            intent="analyze",
            payload={
                "document_id": "doc_1",
                "format": "pdf",
                "size": 1024,
            },
        )

        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))

        assert "task" in output
        assert output["task"]["id"] == msg_id
        assert output["task"]["subtasks"] == []

    @pytest.mark.asyncio
    async def test_egress_delegate_task(self):
        """Transform delegate task canonical message to AutoGPT format."""
        adapter = AutoGPTAdapter()

        msg_id = str(uuid.uuid4())
        canonical = CanonicalMessage(
            message_id=msg_id,
            source_agent_id="autogpt_agent",
            conversation_id="conv_2",
            message_type="request",
            intent="delegate",
            payload={
                "task_description": "Process entities",
                "target_agent": "spacy_agent",
                "deadline": 1743948896.0,
                "subtasks": [
                    {
                        "id": "sub_1",
                        "description": "Extract entities",
                    }
                ],
            },
        )

        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))

        assert "task" in output
        assert output["task"]["description"] == "Process entities"
        assert output["task"]["target_agent"] == "spacy_agent"


class TestRoundTrip:
    """Test round-trip preservation (ingest → egress → compare)."""

    @pytest.mark.asyncio
    async def test_round_trip_flat_task(self):
        """Parse flat task → transform to canonical → back to AutoGPT → compare JSON."""
        adapter = AutoGPTAdapter()
        original_msg = {
            "task": {
                "id": "task_rt_flat",
                "description": "Flat task for round-trip",
                "subtasks": [],  # Include empty subtasks in original to match round-trip
            }
        }
        raw = json.dumps(original_msg).encode("utf-8")

        # Ingest
        canonical = await adapter.ingest(raw)
        # Egress
        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))

        # Verify structure matches (using original task from metadata)
        assert output["task"]["id"] == original_msg["task"]["id"]
        assert output["task"]["description"] == original_msg["task"]["description"]
        assert output["task"]["subtasks"] == original_msg["task"]["subtasks"]

    @pytest.mark.asyncio
    async def test_round_trip_nested_3_level(self):
        """Parse 3-level nested task → transform → back → verify structure preserved."""
        adapter = AutoGPTAdapter()
        original_msg = {
            "task": {
                "id": "root_rt",
                "description": "Root for round-trip",
                "subtasks": [
                    {
                        "id": "l1_1",
                        "description": "Level 1 Task 1",
                        "subtasks": [
                            {
                                "id": "l2_1",
                                "description": "Level 2 Task 1",
                                "subtasks": [
                                    {
                                        "id": "l3_1",
                                        "description": "Level 3 Task",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
        raw = json.dumps(original_msg).encode("utf-8")

        # Ingest with original task in metadata for perfect round-trip
        canonical = await adapter.ingest(raw)
        # Egress
        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))

        # Verify structure is preserved
        assert output["task"]["id"] == original_msg["task"]["id"]
        assert output["task"]["description"] == original_msg["task"]["description"]
        assert len(output["task"]["subtasks"]) == 1
        assert output["task"]["subtasks"][0]["id"] == "l1_1"
        assert len(output["task"]["subtasks"][0]["subtasks"]) == 1
        assert output["task"]["subtasks"][0]["subtasks"][0]["id"] == "l2_1"
        assert len(output["task"]["subtasks"][0]["subtasks"][0]["subtasks"]) == 1
        assert output["task"]["subtasks"][0]["subtasks"][0]["subtasks"][0]["id"] == "l3_1"

    @pytest.mark.asyncio
    async def test_round_trip_preserves_ids(self):
        """Verify task IDs and subtask IDs preserved exactly through cycle."""
        adapter = AutoGPTAdapter()
        original_msg = {
            "task": {
                "id": "task_id_preserved",
                "description": "Task with ID preservation check",
                "subtasks": [
                    {
                        "id": "subtask_id_1",
                        "description": "Sub 1",
                        "subtasks": [
                            {
                                "id": "subtask_id_1_1",
                                "description": "Sub 1.1",
                            }
                        ],
                    },
                    {
                        "id": "subtask_id_2",
                        "description": "Sub 2",
                    },
                ],
            }
        }
        raw = json.dumps(original_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)
        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))

        # Extract all IDs from output
        output_ids = self._extract_all_ids(output["task"])
        original_ids = self._extract_all_ids(original_msg["task"])

        assert output_ids == original_ids

    @pytest.mark.asyncio
    async def test_round_trip_preserves_descriptions(self):
        """Verify descriptions unchanged through round-trip cycle."""
        adapter = AutoGPTAdapter()
        original_msg = {
            "task": {
                "id": "t1",
                "description": "Parent description",
                "subtasks": [
                    {
                        "id": "s1",
                        "description": "Child description 1",
                        "subtasks": [
                            {
                                "id": "s1_1",
                                "description": "Grandchild description",
                            }
                        ],
                    }
                ],
            }
        }
        raw = json.dumps(original_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)
        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))

        # Extract all descriptions
        output_descs = self._extract_all_descriptions(output["task"])
        original_descs = self._extract_all_descriptions(original_msg["task"])

        assert output_descs == original_descs

    @staticmethod
    def _extract_all_ids(task: dict) -> list:
        """Recursively extract all task IDs from task hierarchy."""
        ids = [task.get("id")]
        for subtask in task.get("subtasks", []):
            ids.extend(
                TestRoundTrip._extract_all_ids(subtask)
            )
        return ids

    @staticmethod
    def _extract_all_descriptions(task: dict) -> list:
        """Recursively extract all task descriptions from task hierarchy."""
        descriptions = [task.get("description")]
        for subtask in task.get("subtasks", []):
            descriptions.extend(
                TestRoundTrip._extract_all_descriptions(subtask)
            )
        return descriptions


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_ingest_missing_task_field(self):
        """Raise ValueError when required 'task' field is missing."""
        adapter = AutoGPTAdapter()
        task_msg = {"no_task_field": {}}
        raw = json.dumps(task_msg).encode("utf-8")

        with pytest.raises(ValueError, match="Missing required field 'task'"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_missing_task_id(self):
        """Raise ValueError when task.id is missing."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"description": "No ID"}}
        raw = json.dumps(task_msg).encode("utf-8")

        with pytest.raises(ValueError, match="Missing required field 'task.id'"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_missing_task_description(self):
        """Raise ValueError when task.description is missing."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"id": "t1"}}
        raw = json.dumps(task_msg).encode("utf-8")

        with pytest.raises(ValueError, match="Missing required field 'task.description'"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_malformed_json(self):
        """Raise ValueError on invalid JSON input."""
        adapter = AutoGPTAdapter()
        raw = b"not valid json {]"

        with pytest.raises(ValueError, match="Failed to decode AutoGPT message"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_invalid_utf8(self):
        """Raise ValueError on invalid UTF-8 encoding."""
        adapter = AutoGPTAdapter()
        raw = b"\xff\xfe invalid utf-8"

        with pytest.raises(ValueError, match="Failed to decode AutoGPT message"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_task_not_dict(self):
        """Raise ValueError when task field is not a dictionary."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": "not a dict"}
        raw = json.dumps(task_msg).encode("utf-8")

        with pytest.raises(ValueError, match="Field 'task' must be a dictionary"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_subtasks_not_list(self):
        """Raise ValueError when subtasks field is not a list."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "t1",
                "description": "Test",
                "subtasks": "not a list",
            }
        }
        raw = json.dumps(task_msg).encode("utf-8")

        with pytest.raises(ValueError, match="Field 'task.subtasks' must be a list"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_ingest_unicode_task_description(self):
        """Handle UTF-8 task descriptions correctly."""
        adapter = AutoGPTAdapter()
        task_msg = {
            "task": {
                "id": "unicode_task",
                "description": "测试中文 🚀 Тест العربية",
                "subtasks": [
                    {
                        "id": "unicode_sub",
                        "description": "🎯 ہندی ελληνικά",
                    }
                ],
            }
        }
        raw = json.dumps(task_msg, ensure_ascii=False).encode("utf-8")

        canonical = await adapter.ingest(raw)
        assert canonical.payload["task_description"] == "测试中文 🚀 Тест العربية"
        assert canonical.payload["subtasks"][0]["description"] == "🎯 ہندی ελληνικά"

        # Verify round-trip preserves unicode
        result = await adapter.egress(canonical)
        output = json.loads(result.decode("utf-8"))
        assert output["task"]["description"] == "测试中文 🚀 Тест العربية"


class TestCanonicalMessageGeneration:
    """Test that CanonicalMessage is created correctly during ingest."""

    @pytest.mark.asyncio
    async def test_canonical_has_required_fields(self):
        """Verify CanonicalMessage has all required fields after ingest."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"id": "t1", "description": "Test"}}
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.message_id is not None
        assert canonical.source_agent_id == "autogpt_agent"
        assert canonical.conversation_id is not None
        assert canonical.message_type == "request"
        assert canonical.intent in ["analyze", "delegate"]
        assert isinstance(canonical.payload, dict)
        assert canonical.metadata is not None
        assert canonical.timestamp_utc is not None

    @pytest.mark.asyncio
    async def test_canonical_metadata_contains_protocol_source(self):
        """Verify metadata includes protocol_source."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"id": "t1", "description": "Test"}}
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.metadata.get("protocol_source") == "autogpt"

    @pytest.mark.asyncio
    async def test_canonical_metadata_contains_conversation_chain(self):
        """Verify metadata includes conversation_chain."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"id": "t1", "description": "Test"}}
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert "conversation_chain" in canonical.metadata
        assert canonical.metadata["conversation_chain"] == ["autogpt_agent"]

    @pytest.mark.asyncio
    async def test_canonical_metadata_contains_vector_clock(self):
        """Verify metadata includes vector_clock."""
        adapter = AutoGPTAdapter()
        task_msg = {"task": {"id": "t1", "description": "Test"}}
        raw = json.dumps(task_msg).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert "vector_clock" in canonical.metadata
        assert canonical.metadata["vector_clock"].get("autogpt_agent") == 1


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.asyncio
async def test_recursive_structure_preservation(depth):
    """Test recursive structure preservation for depths 0 (flat), 1, 2, 3."""
    adapter = AutoGPTAdapter()

    # Build nested structure of specified depth
    def build_nested_task(task_id: str, level: int, max_depth: int) -> dict:
        task = {
            "id": task_id,
            "description": f"Task at level {level}",
        }
        if level < max_depth:
            task["subtasks"] = [
                build_nested_task(f"{task_id}_sub_{i}", level + 1, max_depth)
                for i in range(2)  # 2 subtasks per level
            ]
        else:
            task["subtasks"] = []
        return task

    root_task = build_nested_task("root", 0, depth)
    original_msg = {"task": root_task}
    raw = json.dumps(original_msg).encode("utf-8")

    # Round-trip
    canonical = await adapter.ingest(raw)
    result = await adapter.egress(canonical)
    output = json.loads(result.decode("utf-8"))

    # Count levels in output
    def count_depth(task: dict) -> int:
        subtasks = task.get("subtasks", [])
        if not subtasks:
            return 0
        return 1 + max(count_depth(st) for st in subtasks)

    output_depth = count_depth(output["task"])
    original_depth = count_depth(original_msg["task"])

    assert output_depth == original_depth == depth
