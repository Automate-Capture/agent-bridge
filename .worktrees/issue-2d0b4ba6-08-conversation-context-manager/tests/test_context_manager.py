"""
Tests for ConversationContextManager with WAL persistence and LRU eviction.

Covers:
- Round-trip persistence (store → WAL → recover)
- 3-hop delegation chain preservation
- LRU eviction boundary conditions (1000, 1001, 1002)
- WAL recovery with zero data loss
- Context lookup latency < 10ms
- Edge cases (empty WAL, malformed JSON, concurrent isolation)
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from openclaw_gateway.context_manager import (
    ConversationContext,
    ConversationContextManager,
)
from openclaw_gateway.wal import WAL


@pytest.fixture
def temp_wal_dir():
    """Create a temporary directory for WAL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_wal_path(temp_wal_dir):
    """Create a temporary WAL file path."""
    return os.path.join(temp_wal_dir, "test.wal")


@pytest.fixture
def context_manager(temp_wal_path):
    """Create a ConversationContextManager instance with temp WAL."""
    return ConversationContextManager(max_conversations=1000, wal_path=temp_wal_path)


@pytest.fixture
def context_manager_small(temp_wal_path):
    """Create a ConversationContextManager with smaller capacity for LRU testing."""
    return ConversationContextManager(max_conversations=5, wal_path=temp_wal_path)


class TestConversationContextBasics:
    """Test basic ConversationContext dataclass functionality."""

    def test_context_creation(self):
        """Test ConversationContext instantiation."""
        context = ConversationContext(
            conversation_id="conv_1",
            created_at=1234567890.0,
            agents_in_chain=["agent_a"],
            state_snapshots={},
            status="active",
            wal_entry_id=1,
        )
        assert context.conversation_id == "conv_1"
        assert context.created_at == 1234567890.0
        assert context.agents_in_chain == ["agent_a"]
        assert context.status == "active"

    def test_context_to_dict(self):
        """Test serialization to dict."""
        context = ConversationContext(
            conversation_id="conv_1",
            created_at=1234567890.0,
            agents_in_chain=["agent_a", "agent_b"],
            state_snapshots={"agent_a": {"key": "value"}},
            status="completed",
            wal_entry_id=5,
        )
        data = context.to_dict()

        assert data["conversation_id"] == "conv_1"
        assert data["created_at"] == 1234567890.0
        assert data["agents_in_chain"] == ["agent_a", "agent_b"]
        assert data["state_snapshots"] == {"agent_a": {"key": "value"}}
        assert data["status"] == "completed"
        assert data["wal_entry_id"] == 5

    def test_context_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "conversation_id": "conv_1",
            "created_at": 1234567890.0,
            "agents_in_chain": ["agent_a", "agent_b"],
            "state_snapshots": {"agent_a": {"key": "value"}},
            "status": "completed",
            "wal_entry_id": 5,
        }
        context = ConversationContext.from_dict(data)

        assert context.conversation_id == "conv_1"
        assert context.created_at == 1234567890.0
        assert context.agents_in_chain == ["agent_a", "agent_b"]
        assert context.state_snapshots == {"agent_a": {"key": "value"}}
        assert context.status == "completed"
        assert context.wal_entry_id == 5

    def test_context_round_trip_json_equality(self):
        """Test JSON round-trip equality."""
        context = ConversationContext(
            conversation_id="conv_1",
            created_at=1234567890.123,
            agents_in_chain=["agent_a", "agent_b", "agent_c"],
            state_snapshots={
                "agent_a": {"nested": {"deep": "value"}},
                "agent_b": {"list": [1, 2, 3]},
            },
            status="partially_recovered",
            wal_entry_id=42,
        )

        # Serialize and deserialize
        data = context.to_dict()
        recovered = ConversationContext.from_dict(data)

        # JSON equality check
        assert json.dumps(data, sort_keys=True) == json.dumps(
            recovered.to_dict(), sort_keys=True
        )


class TestStoreAndRetrieve:
    """Test storing and retrieving contexts."""

    def test_create_and_get_context(self, context_manager):
        """Test create_context followed by get_context."""
        now = time.time()
        context = context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        assert context.conversation_id == "conv_1"
        assert context.agents_in_chain == ["agent_a"]
        assert context.status == "active"

        # Retrieve context
        retrieved = context_manager.get_context("conv_1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv_1"
        assert retrieved.agents_in_chain == ["agent_a"]
        assert retrieved.created_at == now

    def test_get_nonexistent_context(self, context_manager):
        """Test get_context for non-existent conversation."""
        result = context_manager.get_context("nonexistent")
        assert result is None

    def test_add_to_chain(self, context_manager):
        """Test adding agents to delegation chain."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        context_manager.add_to_chain("conv_1", "agent_b")
        context = context_manager.get_context("conv_1")
        assert context.agents_in_chain == ["agent_a", "agent_b"]

        context_manager.add_to_chain("conv_1", "agent_c")
        context = context_manager.get_context("conv_1")
        assert context.agents_in_chain == ["agent_a", "agent_b", "agent_c"]

    def test_add_duplicate_agent_to_chain(self, context_manager):
        """Test that duplicate agents are not added to chain."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        context_manager.add_to_chain("conv_1", "agent_a")  # Duplicate
        context = context_manager.get_context("conv_1")
        assert context.agents_in_chain == ["agent_a"]  # No duplicate

    def test_remove_from_chain(self, context_manager):
        """Test removing agents from delegation chain."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )
        context_manager.add_to_chain("conv_1", "agent_b")
        context_manager.add_to_chain("conv_1", "agent_c")

        context_manager.remove_from_chain("conv_1", "agent_b")
        context = context_manager.get_context("conv_1")
        assert context.agents_in_chain == ["agent_a", "agent_c"]

    def test_update_status(self, context_manager):
        """Test updating conversation status."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        assert context_manager.get_context("conv_1").status == "active"

        context_manager.update_status("conv_1", "completed")
        assert context_manager.get_context("conv_1").status == "completed"

        context_manager.update_status("conv_1", "failed")
        assert context_manager.get_context("conv_1").status == "failed"

    def test_update_state_snapshot(self, context_manager):
        """Test updating state snapshots."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        snapshot = {"key": "value", "nested": {"data": 123}}
        context_manager.update_state_snapshot("conv_1", "agent_a", snapshot)

        context = context_manager.get_context("conv_1")
        assert context.state_snapshots["agent_a"] == snapshot


class TestThreeHopDelegation:
    """Test 3-hop delegation chain preservation."""

    def test_context_preservation_three_hops(self, context_manager):
        """Test context through 3-hop delegation: agent_a → agent_b → agent_c → agent_d."""
        now = time.time()

        # Create context with initial agent
        context_manager.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        # Add 3 more agents (3 hops total = 4 agents in chain)
        context_manager.add_to_chain("conv_1", "agent_b")
        context_manager.add_to_chain("conv_1", "agent_c")
        context_manager.add_to_chain("conv_1", "agent_d")

        # Retrieve and verify
        context = context_manager.get_context("conv_1")
        assert context.agents_in_chain == ["agent_a", "agent_b", "agent_c", "agent_d"]
        assert context.conversation_id == "conv_1"
        assert context.created_at == now
        assert context.status == "active"

        # Store state snapshots for each agent
        context_manager.update_state_snapshot(
            "conv_1", "agent_a", {"step": 1, "data": "initial"}
        )
        context_manager.update_state_snapshot(
            "conv_1", "agent_b", {"step": 2, "data": "delegated"}
        )
        context_manager.update_state_snapshot(
            "conv_1", "agent_c", {"step": 3, "data": "subdelegated"}
        )
        context_manager.update_state_snapshot(
            "conv_1", "agent_d", {"step": 4, "data": "final"}
        )

        # Verify complete chain with snapshots
        final = context_manager.get_context("conv_1")
        assert len(final.agents_in_chain) == 4
        assert final.state_snapshots["agent_a"]["step"] == 1
        assert final.state_snapshots["agent_d"]["step"] == 4


class TestLRUEviction:
    """Test LRU eviction at capacity boundary."""

    def test_lru_eviction_boundary_1000(self, temp_wal_path):
        """Test that 1000 contexts fit without eviction."""
        mgr = ConversationContextManager(max_conversations=1000, wal_path=temp_wal_path)
        now = time.time()

        # Create exactly 1000 contexts
        for i in range(1000):
            mgr.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        # All 1000 should be present
        assert mgr.get_active_contexts_count() == 1000
        for i in range(1000):
            assert mgr.get_context(f"conv_{i}") is not None

    def test_lru_eviction_boundary_1001(self, temp_wal_path):
        """Test that 1001st context evicts oldest."""
        mgr = ConversationContextManager(max_conversations=1000, wal_path=temp_wal_path)
        now = time.time()

        # Create 1000 contexts
        for i in range(1000):
            mgr.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        # Add 1001st - oldest (conv_0) should be evicted
        mgr.create_context(
            conversation_id="conv_1000", initial_agent="agent_a", created_at=now + 1000
        )

        # Verify 1000 contexts remain
        assert mgr.get_active_contexts_count() == 1000

        # Verify oldest was evicted
        assert mgr.get_context("conv_0") is None

        # Verify newest is present
        assert mgr.get_context("conv_1000") is not None

        # Verify some middle contexts are still present
        assert mgr.get_context("conv_500") is not None
        assert mgr.get_context("conv_999") is not None

    def test_lru_eviction_boundary_1002(self, temp_wal_path):
        """Test eviction continues correctly beyond 1001."""
        mgr = ConversationContextManager(max_conversations=1000, wal_path=temp_wal_path)
        now = time.time()

        # Create 1002 contexts
        for i in range(1002):
            mgr.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        # Verify exactly 1000 contexts remain
        assert mgr.get_active_contexts_count() == 1000

        # Verify 2 oldest were evicted
        assert mgr.get_context("conv_0") is None
        assert mgr.get_context("conv_1") is None

        # Verify newest ones are present
        assert mgr.get_context("conv_1000") is not None
        assert mgr.get_context("conv_1001") is not None

    def test_lru_get_moves_to_end(self, context_manager_small):
        """Test that get_context updates LRU order."""
        now = time.time()

        # Create 5 contexts
        for i in range(5):
            context_manager_small.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        # Access conv_0 (moves it to end - most recently used)
        context_manager_small.get_context("conv_0")

        # Add 6th context - conv_1 should be evicted (now oldest)
        context_manager_small.create_context(
            conversation_id="conv_5", initial_agent="agent_a", created_at=now + 5
        )

        # conv_0 should still be present (it was accessed)
        assert context_manager_small.get_context("conv_0") is not None

        # conv_1 should be evicted
        assert context_manager_small.get_context("conv_1") is None


class TestWALPersistence:
    """Test WAL persistence and recovery."""

    def test_wal_write_on_create(self, temp_wal_path):
        """Test that create_context writes to WAL."""
        mgr = ConversationContextManager(wal_path=temp_wal_path)
        now = time.time()

        mgr.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )

        # Read WAL file
        with open(temp_wal_path, "r") as f:
            lines = f.readlines()

        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["component"] == "context_manager"
        assert entry["operation_type"] == "create_context"
        assert entry["conversation_id"] == "conv_1"

    def test_wal_write_on_add_to_chain(self, temp_wal_path):
        """Test that add_to_chain writes to WAL."""
        mgr = ConversationContextManager(wal_path=temp_wal_path)
        now = time.time()

        mgr.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )
        mgr.add_to_chain("conv_1", "agent_b")

        # Read WAL file
        with open(temp_wal_path, "r") as f:
            lines = f.readlines()

        assert len(lines) >= 2
        entry = json.loads(lines[1])
        assert entry["operation_type"] == "add_to_chain"
        assert entry["conversation_id"] == "conv_1"
        assert "agent_b" in entry["data"]["agents_in_chain"]

    def test_wal_write_on_update_status(self, temp_wal_path):
        """Test that update_status writes to WAL."""
        mgr = ConversationContextManager(wal_path=temp_wal_path)
        now = time.time()

        mgr.create_context(
            conversation_id="conv_1", initial_agent="agent_a", created_at=now
        )
        mgr.update_status("conv_1", "completed")

        # Read WAL file
        with open(temp_wal_path, "r") as f:
            lines = f.readlines()

        assert len(lines) >= 2
        entry = json.loads(lines[1])
        assert entry["operation_type"] == "update_status"
        assert entry["data"]["status"] == "completed"


class TestWALRecovery:
    """Test WAL recovery on startup."""

    @pytest.mark.asyncio
    async def test_wal_recovery_zero_loss(self, temp_wal_path):
        """Test complete recovery of 100 contexts with delegations."""
        # Create initial state
        mgr1 = ConversationContextManager(wal_path=temp_wal_path)
        now = time.time()

        contexts_data = {}
        for i in range(100):
            mgr1.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )
            # Add chain
            mgr1.add_to_chain(f"conv_{i}", "agent_b")
            mgr1.add_to_chain(f"conv_{i}", "agent_c")
            # Add state snapshots
            mgr1.update_state_snapshot(
                f"conv_{i}",
                "agent_a",
                {"index": i, "status": "delegated"},
            )
            mgr1.update_state_snapshot(
                f"conv_{i}",
                "agent_b",
                {"index": i, "result": f"processed_{i}"},
            )

            # Store expected data for verification (capture after all updates)
            ctx = mgr1.get_context(f"conv_{i}")
            contexts_data[f"conv_{i}"] = ctx.to_dict()

        # Simulate restart - create new manager and recover
        mgr2 = ConversationContextManager(wal_path=temp_wal_path)
        await mgr2.recover_from_wal()

        # Verify all contexts recovered
        assert mgr2.get_active_contexts_count() == 100

        # Verify each context matches exactly
        for conv_id, expected_data in contexts_data.items():
            recovered_ctx = mgr2.get_context(conv_id)
            assert recovered_ctx is not None

            # Specific field checks (most important)
            assert recovered_ctx.conversation_id == expected_data["conversation_id"]
            assert recovered_ctx.created_at == expected_data["created_at"]
            assert recovered_ctx.agents_in_chain == expected_data["agents_in_chain"]
            assert (
                recovered_ctx.state_snapshots == expected_data["state_snapshots"]
            )
            assert recovered_ctx.status == expected_data["status"]

    @pytest.mark.asyncio
    async def test_wal_recovery_empty_file(self, temp_wal_path):
        """Test recovery with non-existent WAL file."""
        mgr = ConversationContextManager(wal_path=temp_wal_path)
        # Should not raise error
        await mgr.recover_from_wal()
        assert mgr.get_active_contexts_count() == 0

    @pytest.mark.asyncio
    async def test_wal_recovery_with_malformed_json(self, temp_wal_path):
        """Test recovery skips malformed JSON lines."""
        # Write some good and bad entries
        with open(temp_wal_path, "w") as f:
            # Good entry
            entry1 = {
                "entry_id": 1,
                "timestamp": time.time(),
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": time.time(),
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1,
                },
            }
            f.write(json.dumps(entry1) + "\n")

            # Malformed entry
            f.write("this is not json\n")

            # Another good entry
            entry2 = {
                "entry_id": 2,
                "timestamp": time.time(),
                "component": "context_manager",
                "conversation_id": "conv_2",
                "operation_type": "create_context",
                "data": {
                    "conversation_id": "conv_2",
                    "created_at": time.time(),
                    "agents_in_chain": ["agent_b"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 2,
                },
            }
            f.write(json.dumps(entry2) + "\n")

        mgr = ConversationContextManager(wal_path=temp_wal_path)
        await mgr.recover_from_wal()

        # Should have recovered 2 good contexts, skipped malformed
        assert mgr.get_active_contexts_count() == 2
        assert mgr.get_context("conv_1") is not None
        assert mgr.get_context("conv_2") is not None


class TestContextLookupLatency:
    """Test context lookup performance."""

    def test_context_lookup_latency_1000_contexts(self, temp_wal_path):
        """Benchmark context lookup with 1000 contexts in memory."""
        mgr = ConversationContextManager(max_conversations=1000, wal_path=temp_wal_path)
        now = time.time()

        # Create 1000 contexts
        for i in range(1000):
            mgr.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        # Measure lookup latency (should be O(1) + LRU update)
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            mgr.get_context("conv_500")
        end = time.perf_counter()

        avg_latency_ms = ((end - start) / iterations) * 1000
        # Assert < 10ms per lookup
        assert avg_latency_ms < 10.0, f"Lookup latency {avg_latency_ms:.2f}ms exceeds 10ms"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_add_to_chain_nonexistent_conversation(self, context_manager):
        """Test add_to_chain on non-existent conversation (should not error)."""
        context_manager.add_to_chain("nonexistent", "agent_a")
        # Should silently do nothing
        assert context_manager.get_context("nonexistent") is None

    def test_remove_from_chain_nonexistent_conversation(self, context_manager):
        """Test remove_from_chain on non-existent conversation (should not error)."""
        context_manager.remove_from_chain("nonexistent", "agent_a")
        # Should silently do nothing
        assert context_manager.get_context("nonexistent") is None

    def test_update_status_nonexistent_conversation(self, context_manager):
        """Test update_status on non-existent conversation (should not error)."""
        context_manager.update_status("nonexistent", "completed")
        # Should silently do nothing
        assert context_manager.get_context("nonexistent") is None

    def test_concurrent_conversation_isolation(self, context_manager):
        """Test state snapshot isolation between conversations."""
        now = time.time()

        # Create 5 conversations
        for i in range(5):
            context_manager.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        # Add different state snapshots to each
        for i in range(5):
            context_manager.update_state_snapshot(
                f"conv_{i}", "agent_a", {"conversation_index": i}
            )

        # Verify no cross-contamination
        for i in range(5):
            ctx = context_manager.get_context(f"conv_{i}")
            assert ctx.state_snapshots["agent_a"]["conversation_index"] == i

    def test_get_all_contexts(self, context_manager):
        """Test get_all_contexts method."""
        now = time.time()

        # Create 3 contexts
        for i in range(3):
            context_manager.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )

        all_contexts = context_manager.get_all_contexts()
        assert len(all_contexts) == 3
        assert "conv_0" in all_contexts
        assert "conv_1" in all_contexts
        assert "conv_2" in all_contexts


class TestCriticalEdgeCases:
    """Test critical edge cases and boundary conditions."""

    def test_create_context_with_empty_agent_id(self, context_manager):
        """Test creating context with empty agent ID (should still work)."""
        now = time.time()
        context = context_manager.create_context(
            conversation_id="conv_empty_agent", initial_agent="", created_at=now
        )
        assert context.agents_in_chain == [""]
        retrieved = context_manager.get_context("conv_empty_agent")
        assert retrieved is not None

    def test_create_context_with_special_characters_in_id(self, context_manager):
        """Test context creation with special characters in conversation ID."""
        now = time.time()
        special_id = "conv_123.456/subconv-ABC_XYZ"
        context = context_manager.create_context(
            conversation_id=special_id, initial_agent="agent_a", created_at=now
        )
        assert context.conversation_id == special_id
        retrieved = context_manager.get_context(special_id)
        assert retrieved is not None
        assert retrieved.conversation_id == special_id

    def test_create_context_with_very_large_timestamp(self, context_manager):
        """Test context creation with future timestamp."""
        far_future = 9999999999.99
        context = context_manager.create_context(
            conversation_id="conv_future", initial_agent="agent_a", created_at=far_future
        )
        assert context.created_at == far_future
        retrieved = context_manager.get_context("conv_future")
        assert retrieved.created_at == far_future

    def test_create_context_with_zero_timestamp(self, context_manager):
        """Test context creation with zero timestamp (epoch)."""
        context = context_manager.create_context(
            conversation_id="conv_epoch", initial_agent="agent_a", created_at=0.0
        )
        assert context.created_at == 0.0
        retrieved = context_manager.get_context("conv_epoch")
        assert retrieved.created_at == 0.0

    def test_add_many_agents_to_chain(self, context_manager):
        """Test adding many agents to a single conversation chain."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_long_chain", initial_agent="agent_0", created_at=now
        )

        # Add 100 agents to the chain
        for i in range(1, 101):
            context_manager.add_to_chain("conv_long_chain", f"agent_{i}")

        ctx = context_manager.get_context("conv_long_chain")
        assert len(ctx.agents_in_chain) == 101
        assert ctx.agents_in_chain[0] == "agent_0"
        assert ctx.agents_in_chain[100] == "agent_100"

    def test_state_snapshot_with_empty_dict(self, context_manager):
        """Test updating state snapshot with empty dict."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_empty_snapshot", initial_agent="agent_a", created_at=now
        )
        context_manager.update_state_snapshot(
            "conv_empty_snapshot", "agent_a", {}
        )
        ctx = context_manager.get_context("conv_empty_snapshot")
        assert ctx.state_snapshots["agent_a"] == {}

    def test_state_snapshot_with_none_values(self, context_manager):
        """Test updating state snapshot with None values."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_none_snapshot", initial_agent="agent_a", created_at=now
        )
        snapshot = {"key1": None, "key2": "value", "key3": None}
        context_manager.update_state_snapshot(
            "conv_none_snapshot", "agent_a", snapshot
        )
        ctx = context_manager.get_context("conv_none_snapshot")
        assert ctx.state_snapshots["agent_a"] == snapshot
        assert ctx.state_snapshots["agent_a"]["key1"] is None

    def test_state_snapshot_with_deeply_nested_structure(self, context_manager):
        """Test state snapshot with complex nested data structures."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_deep", initial_agent="agent_a", created_at=now
        )
        deep_snapshot = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "data": "value",
                            "list": [1, 2, 3],
                            "nested_list": [[1, 2], [3, 4]],
                        }
                    }
                }
            }
        }
        context_manager.update_state_snapshot("conv_deep", "agent_a", deep_snapshot)
        ctx = context_manager.get_context("conv_deep")
        assert (
            ctx.state_snapshots["agent_a"]["level1"]["level2"]["level3"]["level4"][
                "data"
            ]
            == "value"
        )

    def test_multiple_snapshots_per_agent_overwrites(self, context_manager):
        """Test that multiple updates to same agent snapshot overwrite."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_overwrite", initial_agent="agent_a", created_at=now
        )

        # First snapshot
        context_manager.update_state_snapshot(
            "conv_overwrite", "agent_a", {"version": 1, "data": "first"}
        )
        ctx = context_manager.get_context("conv_overwrite")
        assert ctx.state_snapshots["agent_a"]["version"] == 1

        # Second snapshot (overwrites)
        context_manager.update_state_snapshot(
            "conv_overwrite", "agent_a", {"version": 2, "data": "second"}
        )
        ctx = context_manager.get_context("conv_overwrite")
        assert ctx.state_snapshots["agent_a"]["version"] == 2
        assert ctx.state_snapshots["agent_a"]["data"] == "second"

    def test_remove_all_agents_from_chain(self, context_manager):
        """Test removing all agents from a chain (leaves empty list)."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_remove_all", initial_agent="agent_a", created_at=now
        )
        context_manager.add_to_chain("conv_remove_all", "agent_b")

        context_manager.remove_from_chain("conv_remove_all", "agent_a")
        context_manager.remove_from_chain("conv_remove_all", "agent_b")

        ctx = context_manager.get_context("conv_remove_all")
        assert ctx.agents_in_chain == []

    def test_remove_nonexistent_agent_from_chain(self, context_manager):
        """Test removing agent that doesn't exist in chain."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_remove_nonexistent", initial_agent="agent_a", created_at=now
        )

        # Try to remove agent that was never added (should not error)
        context_manager.remove_from_chain("conv_remove_nonexistent", "agent_z")
        ctx = context_manager.get_context("conv_remove_nonexistent")
        assert ctx.agents_in_chain == ["agent_a"]

    def test_status_transitions_all_states(self, context_manager):
        """Test all valid status transitions."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_status_transition",
            initial_agent="agent_a",
            created_at=now,
        )

        # Test all status values
        valid_statuses = ["active", "completed", "failed", "partially_recovered"]
        for status in valid_statuses:
            context_manager.update_status("conv_status_transition", status)
            ctx = context_manager.get_context("conv_status_transition")
            assert ctx.status == status

    def test_get_context_count_empty(self, context_manager):
        """Test get_active_contexts_count on empty manager."""
        assert context_manager.get_active_contexts_count() == 0

    def test_get_all_contexts_empty(self, context_manager):
        """Test get_all_contexts on empty manager."""
        assert context_manager.get_all_contexts() == {}

    def test_wal_entry_id_monotonic_increase(self, temp_wal_path):
        """Test that WAL entry IDs increase monotonically."""
        mgr = ConversationContextManager(wal_path=temp_wal_path)
        now = time.time()

        entry_ids = []
        for i in range(10):
            mgr.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )
            ctx = mgr.get_context(f"conv_{i}")
            entry_ids.append(ctx.wal_entry_id)

        # All entry IDs should be unique and increasing
        assert len(set(entry_ids)) == len(entry_ids)
        assert entry_ids == sorted(entry_ids)

    def test_context_json_serializable(self, context_manager):
        """Test that all contexts can be JSON serialized."""
        now = time.time()
        context_manager.create_context(
            conversation_id="conv_json", initial_agent="agent_a", created_at=now
        )
        context_manager.add_to_chain("conv_json", "agent_b")
        context_manager.update_state_snapshot(
            "conv_json",
            "agent_a",
            {"complex": {"nested": [1, 2, 3]}, "null": None},
        )

        ctx = context_manager.get_context("conv_json")
        # Should not raise
        json_str = json.dumps(ctx.to_dict())
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    @pytest.mark.asyncio
    async def test_recover_from_wal_idempotent(self, temp_wal_path):
        """Test that recovery is idempotent - recovering twice yields same results."""
        # Create initial state with multiple operations
        mgr1 = ConversationContextManager(wal_path=temp_wal_path)
        now = time.time()

        for i in range(5):
            mgr1.create_context(
                conversation_id=f"conv_{i}", initial_agent="agent_a", created_at=now + i
            )
            mgr1.add_to_chain(f"conv_{i}", "agent_b")

        # First recovery
        mgr2 = ConversationContextManager(wal_path=temp_wal_path)
        await mgr2.recover_from_wal()

        recovered_once = {}
        for i in range(5):
            ctx = mgr2.get_context(f"conv_{i}")
            recovered_once[f"conv_{i}"] = (ctx.agents_in_chain, ctx.conversation_id)

        # Second recovery (from same WAL)
        mgr3 = ConversationContextManager(wal_path=temp_wal_path)
        await mgr3.recover_from_wal()

        recovered_twice = {}
        for i in range(5):
            ctx = mgr3.get_context(f"conv_{i}")
            recovered_twice[f"conv_{i}"] = (ctx.agents_in_chain, ctx.conversation_id)

        # Results should be identical
        assert recovered_once == recovered_twice
