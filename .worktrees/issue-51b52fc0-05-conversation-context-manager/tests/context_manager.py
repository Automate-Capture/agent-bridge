"""Tests for conversation context manager with LRU eviction and WAL recovery.

This test suite covers:
1. Context creation and retrieval
2. Agent chain growth via add_agent_to_chain()
3. State snapshot isolation between conversations
4. Status updates with wal_entry_id tracking
5. LRU eviction at capacity (1001 conversations triggers eviction)
6. WAL persistence and recovery with zero data loss
7. 3-hop delegation context preservation
8. Latency benchmarking (< 10ms per lookup)
"""

import pytest
import json
import time
from datetime import datetime, timezone

from openclaw_gateway import ConversationContextManager, ConversationContext


class TestContextCreation:
    """Test context creation and basic operations."""

    def test_create_context_basic(self):
        """Test basic context creation with correct initial values."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        context = manager.store_context(
            conversation_id="conv_123",
            initial_agent="agent_a",
            created_at=created_at
        )

        assert context.conversation_id == "conv_123"
        assert context.created_at == created_at
        assert context.agents_in_chain == ["agent_a"]
        assert context.status == "active"
        assert context.state_snapshots == {}
        assert context.wal_entry_id == 1

    def test_create_context_multiple(self):
        """Test creating multiple contexts with incrementing wal_entry_ids."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        ctx1 = manager.store_context("conv_1", "agent_a", created_at)
        ctx2 = manager.store_context("conv_2", "agent_b", created_at)
        ctx3 = manager.store_context("conv_3", "agent_c", created_at)

        assert ctx1.wal_entry_id == 1
        assert ctx2.wal_entry_id == 2
        assert ctx3.wal_entry_id == 3

    def test_get_context_exists(self):
        """Test retrieving an existing context."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        retrieved = manager.get_context("conv_123")

        assert retrieved is not None
        assert retrieved.conversation_id == "conv_123"
        assert retrieved.agents_in_chain == ["agent_a"]

    def test_get_context_nonexistent(self):
        """Test retrieving a non-existent context returns None."""
        manager = ConversationContextManager()

        result = manager.get_context("nonexistent")

        assert result is None

    def test_get_context_moves_to_end_lru(self):
        """Test that get_context updates LRU order (moves to end)."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_1", "agent_a", created_at)
        manager.store_context("conv_2", "agent_b", created_at)
        manager.store_context("conv_3", "agent_c", created_at)

        # Access conv_1, should move to end
        manager.get_context("conv_1")

        # The order should now be: conv_2, conv_3, conv_1
        keys = list(manager._contexts.keys())
        assert keys == ["conv_2", "conv_3", "conv_1"]


class TestAgentChain:
    """Test agent chain tracking via add_agent_to_chain()."""

    def test_add_agent_to_chain_single(self):
        """Test adding one agent to the chain."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        manager.add_agent_to_chain("conv_123", "agent_b")

        context = manager.get_context("conv_123")
        assert context.agents_in_chain == ["agent_a", "agent_b"]

    def test_add_agent_to_chain_multiple(self):
        """Test adding multiple agents to the chain."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        manager.add_agent_to_chain("conv_123", "agent_b")
        manager.add_agent_to_chain("conv_123", "agent_c")

        context = manager.get_context("conv_123")
        assert context.agents_in_chain == ["agent_a", "agent_b", "agent_c"]

    def test_add_agent_to_chain_three_hops(self):
        """Test 3-hop delegation: create → add_to_chain × 2."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        # Create with initial agent
        manager.store_context("conv_123", "agent_1", created_at)
        # Add 2 more agents for 3-hop delegation
        manager.add_agent_to_chain("conv_123", "agent_2")
        manager.add_agent_to_chain("conv_123", "agent_3")

        # Retrieve and verify
        context = manager.get_context("conv_123")
        assert context.agents_in_chain == ["agent_1", "agent_2", "agent_3"]

    def test_add_agent_to_chain_duplicate_ignored(self):
        """Test that adding the same agent twice is idempotent."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        manager.add_agent_to_chain("conv_123", "agent_b")
        manager.add_agent_to_chain("conv_123", "agent_b")  # duplicate

        context = manager.get_context("conv_123")
        assert context.agents_in_chain == ["agent_a", "agent_b"]

    def test_add_agent_to_chain_increments_wal_entry_id(self):
        """Test that each add_to_chain increments wal_entry_id."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        initial_wal_id = manager.get_context("conv_123").wal_entry_id

        manager.add_agent_to_chain("conv_123", "agent_b")
        new_wal_id = manager.get_context("conv_123").wal_entry_id

        assert new_wal_id == initial_wal_id + 1

    def test_add_agent_to_chain_nonexistent_context(self):
        """Test that add_to_chain on nonexistent context is safe."""
        manager = ConversationContextManager()

        # Should not raise
        manager.add_agent_to_chain("nonexistent", "agent_a")


class TestStateSnapshots:
    """Test state snapshot isolation between conversations."""

    def test_state_snapshot_isolation(self):
        """Test that state snapshots don't interfere between conversations."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        # Create two contexts
        manager.store_context("conv_1", "agent_a", created_at)
        manager.store_context("conv_2", "agent_b", created_at)

        # Modify state snapshots of first context
        ctx1 = manager.get_context("conv_1")
        ctx1.state_snapshots["agent_a"] = {"state": "data_1"}

        # Verify second context is unchanged
        ctx2 = manager.get_context("conv_2")
        assert ctx2.state_snapshots == {}

    def test_state_snapshots_per_agent(self):
        """Test storing state snapshots per agent in a conversation."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        manager.add_agent_to_chain("conv_123", "agent_b")

        context = manager.get_context("conv_123")
        context.state_snapshots["agent_a"] = {"progress": 0.5}
        context.state_snapshots["agent_b"] = {"progress": 0.3}

        retrieved = manager.get_context("conv_123")
        assert retrieved.state_snapshots["agent_a"] == {"progress": 0.5}
        assert retrieved.state_snapshots["agent_b"] == {"progress": 0.3}


class TestStatusUpdates:
    """Test status updates with wal_entry_id tracking."""

    def test_update_status_active_to_completed(self):
        """Test updating status from active to completed."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        manager.update_status("conv_123", "completed")

        context = manager.get_context("conv_123")
        assert context.status == "completed"

    def test_update_status_increments_wal_entry_id(self):
        """Test that status update increments wal_entry_id."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        initial_wal_id = manager.get_context("conv_123").wal_entry_id

        manager.update_status("conv_123", "failed")
        new_wal_id = manager.get_context("conv_123").wal_entry_id

        assert new_wal_id == initial_wal_id + 1

    def test_update_status_all_values(self):
        """Test updating to all valid status values."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        statuses = ["active", "completed", "failed", "partially_recovered"]

        for i, status in enumerate(statuses):
            manager.store_context(f"conv_{i}", "agent_a", created_at)
            manager.update_status(f"conv_{i}", status)

            context = manager.get_context(f"conv_{i}")
            assert context.status == status

    def test_update_status_nonexistent_context(self):
        """Test that update_status on nonexistent context is safe."""
        manager = ConversationContextManager()

        # Should not raise
        manager.update_status("nonexistent", "completed")


class TestLRUEviction:
    """Test LRU eviction at capacity (1001 contexts)."""

    def test_lru_eviction_at_capacity(self):
        """Test that after 1001 creations, oldest is evicted."""
        manager = ConversationContextManager(max_conversations=1000)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create 1001 contexts
        for i in range(1001):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)

        # After 1001 creations, should have 1000
        assert len(manager._contexts) == 1000

        # First context should be evicted
        assert manager.get_context("conv_0") is None

        # Most recent should still be there
        assert manager.get_context("conv_1000") is not None

    def test_lru_eviction_threshold_exactly_1001(self):
        """Test that eviction happens at exactly 1001 (not 1000)."""
        manager = ConversationContextManager(max_conversations=1000)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create 1000 contexts - should all be kept
        for i in range(1000):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)
        assert len(manager._contexts) == 1000

        # Create the 1001st - should trigger eviction
        manager.store_context("conv_1000", "agent_1000", created_at)
        assert len(manager._contexts) == 1000

        # First should be evicted
        assert manager.get_context("conv_0") is None

    def test_lru_eviction_oldest_removed(self):
        """Test that LRU eviction removes the oldest (first inserted)."""
        manager = ConversationContextManager(max_conversations=3)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create 3 contexts
        manager.store_context("a", "agent_a", created_at)
        manager.store_context("b", "agent_b", created_at)
        manager.store_context("c", "agent_c", created_at)

        # Add a 4th - should evict "a"
        manager.store_context("d", "agent_d", created_at)

        assert manager.get_context("a") is None
        assert manager.get_context("b") is not None
        assert manager.get_context("c") is not None
        assert manager.get_context("d") is not None

    def test_lru_eviction_respects_get_access_order(self):
        """Test that accessing a context moves it to end (newer)."""
        manager = ConversationContextManager(max_conversations=3)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create 3 contexts
        manager.store_context("a", "agent_a", created_at)
        manager.store_context("b", "agent_b", created_at)
        manager.store_context("c", "agent_c", created_at)

        # Access "a" - moves it to end
        manager.get_context("a")

        # Add a 4th - should evict "b" (oldest now), not "a"
        manager.store_context("d", "agent_d", created_at)

        assert manager.get_context("a") is not None
        assert manager.get_context("b") is None
        assert manager.get_context("c") is not None
        assert manager.get_context("d") is not None

    def test_lru_capacity_custom_limit(self):
        """Test that LRU respects custom capacity limit."""
        manager = ConversationContextManager(max_conversations=5)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create 7 contexts
        for i in range(7):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)

        # Should keep only 5 most recent
        assert len(manager._contexts) == 5

        # First 2 should be evicted
        assert manager.get_context("conv_0") is None
        assert manager.get_context("conv_1") is None

        # Last 5 should remain
        for i in range(2, 7):
            assert manager.get_context(f"conv_{i}") is not None


class TestWALPersistence:
    """Test WAL persistence and recovery."""

    def test_conversation_context_to_dict(self):
        """Test ConversationContext.to_dict() serialization."""
        context = ConversationContext(
            conversation_id="conv_123",
            created_at=1704067200.0,
            agents_in_chain=["agent_a", "agent_b"],
            state_snapshots={"agent_a": {"key": "value"}},
            status="completed",
            wal_entry_id=5
        )

        data = context.to_dict()

        assert data["conversation_id"] == "conv_123"
        assert data["created_at"] == 1704067200.0
        assert data["agents_in_chain"] == ["agent_a", "agent_b"]
        assert data["state_snapshots"] == {"agent_a": {"key": "value"}}
        assert data["status"] == "completed"
        assert data["wal_entry_id"] == 5

    def test_conversation_context_to_dict_is_json_serializable(self):
        """Test that to_dict() output is JSON serializable."""
        context = ConversationContext(
            conversation_id="conv_123",
            created_at=1704067200.0,
            agents_in_chain=["agent_a", "agent_b"],
            state_snapshots={"agent_a": {"key": "value"}},
            status="completed",
            wal_entry_id=5
        )

        data = context.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_conversation_context_from_dict(self):
        """Test ConversationContext.from_dict() deserialization."""
        data = {
            "conversation_id": "conv_123",
            "created_at": 1704067200.0,
            "agents_in_chain": ["agent_a", "agent_b"],
            "state_snapshots": {"agent_a": {"key": "value"}},
            "status": "completed",
            "wal_entry_id": 5
        }

        context = ConversationContext.from_dict(data)

        assert context.conversation_id == "conv_123"
        assert context.created_at == 1704067200.0
        assert context.agents_in_chain == ["agent_a", "agent_b"]
        assert context.state_snapshots == {"agent_a": {"key": "value"}}
        assert context.status == "completed"
        assert context.wal_entry_id == 5

    def test_wal_round_trip(self):
        """Test round-trip: to_dict → from_dict preserves all fields."""
        original = ConversationContext(
            conversation_id="conv_123",
            created_at=1704067200.0,
            agents_in_chain=["agent_a", "agent_b", "agent_c"],
            state_snapshots={"agent_a": {"state": 1}, "agent_b": {"state": 2}},
            status="partially_recovered",
            wal_entry_id=42
        )

        # to_dict → from_dict
        reconstructed = ConversationContext.from_dict(original.to_dict())

        # All fields should match
        assert reconstructed.conversation_id == original.conversation_id
        assert reconstructed.created_at == original.created_at
        assert reconstructed.agents_in_chain == original.agents_in_chain
        assert reconstructed.state_snapshots == original.state_snapshots
        assert reconstructed.status == original.status
        assert reconstructed.wal_entry_id == original.wal_entry_id


class TestWALRecovery:
    """Test WAL recovery with zero data loss."""

    @pytest.mark.asyncio
    async def test_recover_from_wal_create_context(self):
        """Test recovering create_context entries from WAL."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        # Should recover the context
        context = manager.get_context("conv_1")
        assert context is not None
        assert context.conversation_id == "conv_1"
        assert context.agents_in_chain == ["agent_a"]

    @pytest.mark.asyncio
    async def test_recover_from_wal_multiple_contexts(self):
        """Test recovering multiple contexts from WAL."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1
                }
            },
            {
                "component": "context_manager",
                "conversation_id": "conv_2",
                "operation_type": "create_context",
                "entry_id": 2,
                "data": {
                    "conversation_id": "conv_2",
                    "created_at": 1704067300.0,
                    "agents_in_chain": ["agent_b"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 2
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        assert len(manager._contexts) == 2
        assert manager.get_context("conv_1") is not None
        assert manager.get_context("conv_2") is not None

    @pytest.mark.asyncio
    async def test_recover_from_wal_add_to_chain(self):
        """Test recovering add_to_chain entries from WAL."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1
                }
            },
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "add_to_chain",
                "entry_id": 2,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a", "agent_b"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 2
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        context = manager.get_context("conv_1")
        assert context.agents_in_chain == ["agent_a", "agent_b"]

    @pytest.mark.asyncio
    async def test_recover_from_wal_update_status(self):
        """Test recovering update_status entries from WAL."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1
                }
            },
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "update_status",
                "entry_id": 2,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "completed",
                    "wal_entry_id": 2
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        context = manager.get_context("conv_1")
        assert context.status == "completed"

    @pytest.mark.asyncio
    async def test_recover_from_wal_restores_wal_entry_id_sequence(self):
        """Test that recovery restores _next_wal_id correctly."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 5,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 5
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        # _next_wal_id should be 6 (max_entry_id + 1)
        assert manager._next_wal_id == 6

    @pytest.mark.asyncio
    async def test_recover_from_wal_zero_data_loss(self):
        """Test recovery with zero data loss (all fields preserved)."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {"agent_a": {"key": "value"}},
                    "status": "active",
                    "wal_entry_id": 1
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        recovered = manager.get_context("conv_1")

        # Verify all fields are preserved
        assert recovered.conversation_id == "conv_1"
        assert recovered.created_at == 1704067200.0
        assert recovered.agents_in_chain == ["agent_a"]
        assert recovered.state_snapshots == {"agent_a": {"key": "value"}}
        assert recovered.status == "active"
        assert recovered.wal_entry_id == 1

    @pytest.mark.asyncio
    async def test_recover_from_wal_ignores_non_context_manager_entries(self):
        """Test that recovery ignores entries from other components."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "other_component",
                "conversation_id": "conv_1",
                "operation_type": "some_op",
                "entry_id": 1,
                "data": {}
            },
            {
                "component": "context_manager",
                "conversation_id": "conv_2",
                "operation_type": "create_context",
                "entry_id": 2,
                "data": {
                    "conversation_id": "conv_2",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_b"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 2
                }
            }
        ]

        await manager.recover_from_wal(wal_entries)

        # Should only recover conv_2
        assert manager.get_context("conv_1") is None
        assert manager.get_context("conv_2") is not None


class TestContextChainPreservation:
    """Test context preservation across 3-hop delegations."""

    def test_context_chain_preservation_three_hops(self):
        """Test JSON equality of agents_in_chain after 3-hop delegation."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        # Create and add agents for 3-hop delegation
        manager.store_context("conv_123", "agent_1", created_at)
        manager.add_agent_to_chain("conv_123", "agent_2")
        manager.add_agent_to_chain("conv_123", "agent_3")

        # Retrieve context
        context = manager.get_context("conv_123")

        # Verify chain
        original_chain = ["agent_1", "agent_2", "agent_3"]
        retrieved_chain = context.agents_in_chain

        # JSON equality check
        assert json.dumps(original_chain, sort_keys=True) == json.dumps(retrieved_chain, sort_keys=True)
        assert retrieved_chain == original_chain

    def test_context_chain_preserved_through_wal_recovery(self):
        """Test that context chain is preserved through WAL round-trip."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_123",
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_123",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_1"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1
                }
            },
            {
                "component": "context_manager",
                "conversation_id": "conv_123",
                "operation_type": "add_to_chain",
                "entry_id": 2,
                "data": {
                    "conversation_id": "conv_123",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_1", "agent_2"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 2
                }
            },
            {
                "component": "context_manager",
                "conversation_id": "conv_123",
                "operation_type": "add_to_chain",
                "entry_id": 3,
                "data": {
                    "conversation_id": "conv_123",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_1", "agent_2", "agent_3"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 3
                }
            }
        ]

        # Simulate: delete all contexts and recover from WAL
        manager._contexts.clear()

        # Use sync version for test (would be async in production)
        import asyncio
        asyncio.run(manager.recover_from_wal(wal_entries))

        context = manager.get_context("conv_123")
        expected_chain = ["agent_1", "agent_2", "agent_3"]

        assert context.agents_in_chain == expected_chain


class TestLatencyBenchmarking:
    """Test context lookup latency < 10ms per lookup."""

    def test_get_context_latency_single_lookup(self):
        """Benchmark single get_context() lookup - verify < 10ms."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)

        # Warm up
        manager.get_context("conv_123")

        # Benchmark the lookup
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            result = manager.get_context("conv_123")
        elapsed = (time.time() - start) * 1000  # convert to ms

        # Verify result
        assert result is not None

        # Calculate mean latency per operation
        mean_latency = elapsed / iterations

        # Should be < 10ms per lookup (dict O(1) operation)
        assert mean_latency < 10.0, f"Mean latency {mean_latency}ms exceeds 10ms limit"

    def test_get_context_latency_many_contexts(self):
        """Benchmark get_context() with 1000 contexts - verify < 10ms."""
        manager = ConversationContextManager(max_conversations=1000)
        created_at = datetime.now(timezone.utc).timestamp()

        # Fill up the manager
        for i in range(1000):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)

        # Warm up
        manager.get_context("conv_500")

        # Benchmark lookup in a full cache
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            result = manager.get_context("conv_500")
        elapsed = (time.time() - start) * 1000  # convert to ms

        # Verify result
        assert result is not None

        # Calculate mean latency per operation
        mean_latency = elapsed / iterations

        # Should be < 10ms even with 1000 contexts (OrderedDict O(1))
        assert mean_latency < 10.0, f"Mean latency {mean_latency}ms exceeds 10ms limit"

    def test_get_context_mean_latency_lt_10ms(self):
        """Test that mean latency is < 10ms with random access patterns."""
        manager = ConversationContextManager(max_conversations=1000)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create some contexts
        for i in range(100):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)

        # Test with random access to multiple contexts
        import random
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            idx = random.randint(0, 99)
            result = manager.get_context(f"conv_{idx}")
        elapsed = (time.time() - start) * 1000  # convert to ms

        # Verify result
        assert result is not None

        # Calculate mean latency per operation
        mean_latency = elapsed / iterations

        # Should be < 10ms for all lookups
        assert mean_latency < 10.0, f"Mean latency {mean_latency}ms exceeds 10ms limit"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_conversation_id(self):
        """Test handling of empty string conversation_id."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        # Create context with empty string conversation_id
        context = manager.store_context("", "agent_a", created_at)
        assert context.conversation_id == ""

        # Should be retrievable
        retrieved = manager.get_context("")
        assert retrieved is not None
        assert retrieved.conversation_id == ""

    def test_very_long_conversation_id(self):
        """Test handling of very long conversation_id (1000+ chars)."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        long_id = "conv_" + ("x" * 10000)
        context = manager.store_context(long_id, "agent_a", created_at)
        assert context.conversation_id == long_id

        retrieved = manager.get_context(long_id)
        assert retrieved is not None
        assert retrieved.conversation_id == long_id

    def test_special_characters_in_conversation_id(self):
        """Test handling of special characters in conversation_id."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        special_id = "conv_123!@#$%^&*()_+-=[]{}|;:',.<>?/~"
        context = manager.store_context(special_id, "agent_a", created_at)
        assert context.conversation_id == special_id

        retrieved = manager.get_context(special_id)
        assert retrieved is not None

    def test_unicode_in_conversation_id(self):
        """Test handling of Unicode characters in conversation_id."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        unicode_id = "conv_🚀_世界_مرحبا"
        context = manager.store_context(unicode_id, "agent_a", created_at)
        assert context.conversation_id == unicode_id

        retrieved = manager.get_context(unicode_id)
        assert retrieved is not None

    def test_create_context_zero_timestamp(self):
        """Test context creation with zero timestamp (epoch start)."""
        manager = ConversationContextManager()

        context = manager.store_context("conv_zero", "agent_a", 0.0)
        assert context.created_at == 0.0

    def test_create_context_negative_timestamp(self):
        """Test context creation with negative timestamp (pre-epoch)."""
        manager = ConversationContextManager()

        context = manager.store_context("conv_neg", "agent_a", -86400.0)
        assert context.created_at == -86400.0

    def test_create_context_very_large_timestamp(self):
        """Test context creation with very large timestamp (far future)."""
        manager = ConversationContextManager()

        large_ts = 9999999999.0
        context = manager.store_context("conv_future", "agent_a", large_ts)
        assert context.created_at == large_ts

    def test_empty_agent_id(self):
        """Test adding empty string agent_id to chain."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "", created_at)
        manager.add_agent_to_chain("conv_123", "agent_b")

        context = manager.get_context("conv_123")
        assert context.agents_in_chain == ["", "agent_b"]

    def test_very_long_agent_id(self):
        """Test adding very long agent_id to chain."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        long_agent = "agent_" + ("x" * 10000)
        manager.store_context("conv_123", "agent_a", created_at)
        manager.add_agent_to_chain("conv_123", long_agent)

        context = manager.get_context("conv_123")
        assert long_agent in context.agents_in_chain

    def test_many_agents_in_chain(self):
        """Test adding many agents to a chain (stress test)."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_0", created_at)

        # Add 100 agents to the chain
        for i in range(1, 100):
            manager.add_agent_to_chain("conv_123", f"agent_{i}")

        context = manager.get_context("conv_123")
        assert len(context.agents_in_chain) == 100
        assert context.agents_in_chain[0] == "agent_0"
        assert context.agents_in_chain[99] == "agent_99"

    def test_state_snapshots_with_none_values(self):
        """Test state snapshots containing None values."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        context = manager.get_context("conv_123")
        context.state_snapshots["agent_a"] = {"key": None}

        retrieved = manager.get_context("conv_123")
        assert retrieved.state_snapshots["agent_a"]["key"] is None

    def test_state_snapshots_with_nested_structures(self):
        """Test state snapshots with deeply nested structures."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        context = manager.get_context("conv_123")

        nested = {
            "level_1": {
                "level_2": {
                    "level_3": {
                        "level_4": {"data": [1, 2, 3, {"deep": "value"}]}
                    }
                }
            }
        }
        context.state_snapshots["agent_a"] = nested

        retrieved = manager.get_context("conv_123")
        # Verify deep nesting is preserved
        assert retrieved.state_snapshots["agent_a"]["level_1"]["level_2"]["level_3"]["level_4"]["data"][3]["deep"] == "value"

    def test_state_snapshots_with_large_data(self):
        """Test state snapshots with large data structures."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        context = manager.get_context("conv_123")

        # Create large state snapshot
        large_data = [{"id": i, "data": "x" * 1000} for i in range(1000)]
        context.state_snapshots["agent_a"] = {"large": large_data}

        retrieved = manager.get_context("conv_123")
        assert len(retrieved.state_snapshots["agent_a"]["large"]) == 1000

    def test_state_snapshots_json_serializable(self):
        """Test that state snapshots remain JSON serializable."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)
        context = manager.get_context("conv_123")

        context.state_snapshots["agent_a"] = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }

        # Should be JSON serializable
        serialized = json.dumps(context.state_snapshots)
        assert isinstance(serialized, str)

        # Should round-trip
        deserialized = json.loads(serialized)
        assert deserialized == context.state_snapshots

    def test_get_context_missing_key_safety(self):
        """Test that get_context is safe when context doesn't exist."""
        manager = ConversationContextManager()

        # Multiple calls to nonexistent key should not raise
        for _ in range(10):
            result = manager.get_context("missing_key")
            assert result is None

    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access pattern (sequential but rapid)."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        # Create multiple contexts
        for i in range(10):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)

        # Simulate rapid access pattern
        for iteration in range(100):
            idx = iteration % 10
            context = manager.get_context(f"conv_{idx}")
            assert context is not None

            if iteration % 5 == 0:
                manager.add_agent_to_chain(f"conv_{idx}", f"agent_new_{iteration}")

        # Verify all contexts still exist and are correct
        for i in range(10):
            context = manager.get_context(f"conv_{i}")
            assert context is not None

    def test_status_updates_multiple_times(self):
        """Test updating status multiple times on same context."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        manager.store_context("conv_123", "agent_a", created_at)

        statuses_to_test = [
            "active",
            "completed",
            "failed",
            "partially_recovered",
            "active",
            "completed"
        ]

        for status in statuses_to_test:
            manager.update_status("conv_123", status)
            context = manager.get_context("conv_123")
            assert context.status == status

    def test_wal_entry_id_overflow(self):
        """Test behavior with very large wal_entry_id values."""
        manager = ConversationContextManager()
        created_at = datetime.now(timezone.utc).timestamp()

        # Manually set a very large _next_wal_id
        manager._next_wal_id = 2**31 - 1  # Max 32-bit int

        context = manager.store_context("conv_large_wal", "agent_a", created_at)
        assert context.wal_entry_id == 2**31 - 1

    def test_context_to_dict_preserves_types(self):
        """Test that to_dict() preserves field types."""
        context = ConversationContext(
            conversation_id="conv_123",
            created_at=1704067200.5,  # Float with decimal
            agents_in_chain=["agent_1", "agent_2"],
            state_snapshots={"agent_1": {"count": 42}},
            status="active",
            wal_entry_id=99
        )

        data = context.to_dict()

        assert isinstance(data["conversation_id"], str)
        assert isinstance(data["created_at"], float)
        assert isinstance(data["agents_in_chain"], list)
        assert isinstance(data["state_snapshots"], dict)
        assert isinstance(data["status"], str)
        assert isinstance(data["wal_entry_id"], int)

    @pytest.mark.asyncio
    async def test_recover_from_wal_with_missing_data_field(self):
        """Test WAL recovery handles missing data field (expected behavior is to skip or raise)."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_1",
                "operation_type": "create_context",
                "entry_id": 1
                # Missing "data" field
            }
        ]

        # The implementation will raise TypeError because data defaults to {} but ConversationContext
        # requires conversation_id, created_at, agents_in_chain. This is expected behavior.
        # In production, WAL entries should always have complete data.
        try:
            await manager.recover_from_wal(wal_entries)
        except TypeError:
            # Expected: malformed WAL entry cannot be deserialized
            pass

    @pytest.mark.asyncio
    async def test_recover_from_wal_empty_entries_list(self):
        """Test WAL recovery with empty entries list."""
        manager = ConversationContextManager()

        await manager.recover_from_wal([])

        assert len(manager._contexts) == 0

    @pytest.mark.asyncio
    async def test_recover_from_wal_with_malformed_context_id(self):
        """Test WAL recovery with missing conversation_id in entry."""
        manager = ConversationContextManager()

        wal_entries = [
            {
                "component": "context_manager",
                # Missing "conversation_id"
                "operation_type": "create_context",
                "entry_id": 1,
                "data": {
                    "conversation_id": "conv_1",
                    "created_at": 1704067200.0,
                    "agents_in_chain": ["agent_a"],
                    "state_snapshots": {},
                    "status": "active",
                    "wal_entry_id": 1
                }
            }
        ]

        # Should handle gracefully
        await manager.recover_from_wal(wal_entries)

    def test_lru_with_get_and_create_interaction(self):
        """Test LRU behavior when get_context and create_context interact."""
        manager = ConversationContextManager(max_conversations=5)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create 5 contexts: a, b, c, d, e
        for letter in ["a", "b", "c", "d", "e"]:
            manager.store_context(letter, f"agent_{letter}", created_at)

        # Access "a" to move it to end
        manager.get_context("a")

        # Create "f" - should evict "b" (oldest after "a" moved)
        manager.store_context("f", "agent_f", created_at)

        assert manager.get_context("a") is not None
        assert manager.get_context("b") is None
        assert manager.get_context("c") is not None
        assert manager.get_context("d") is not None
        assert manager.get_context("e") is not None
        assert manager.get_context("f") is not None

    def test_lru_eviction_with_large_snapshots(self):
        """Test that LRU evicts contexts even with large state snapshots."""
        manager = ConversationContextManager(max_conversations=3)
        created_at = datetime.now(timezone.utc).timestamp()

        # Create contexts with large state snapshots
        for i in range(4):
            manager.store_context(f"conv_{i}", f"agent_{i}", created_at)
            context = manager.get_context(f"conv_{i}")
            # Add large state snapshot
            context.state_snapshots[f"agent_{i}"] = {"data": "x" * 100000}

        # Should have evicted oldest
        assert len(manager._contexts) == 3
        assert manager.get_context("conv_0") is None

    def test_multiple_managers_are_independent(self):
        """Test that multiple ConversationContextManager instances are independent."""
        manager1 = ConversationContextManager(max_conversations=10)
        manager2 = ConversationContextManager(max_conversations=20)

        created_at = datetime.now(timezone.utc).timestamp()

        manager1.store_context("conv_1", "agent_a", created_at)
        manager2.store_context("conv_1", "agent_b", created_at)

        ctx1 = manager1.get_context("conv_1")
        ctx2 = manager2.get_context("conv_1")

        # Should have different agent IDs
        assert ctx1.agents_in_chain == ["agent_a"]
        assert ctx2.agents_in_chain == ["agent_b"]
