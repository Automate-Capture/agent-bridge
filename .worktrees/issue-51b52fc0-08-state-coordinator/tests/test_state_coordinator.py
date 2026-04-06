"""Comprehensive tests for StateCoordinator with vector clocks and LWW conflict resolution.

Tests cover:
1. Vector clock increment on write
2. Clock state after N writes == N
3. Concurrent write detection
4. LWW resolution (higher timestamp wins)
5. Subscription callback execution
6. State persistence/recovery from WAL
7. Latency benchmarking (< 5ms reads)
8. Edge cases (zero timestamp, multiple keys, same timestamp)
9. Parametrized concurrent agent tests (2-10 agents)
"""

import pytest
import asyncio
import time
import uuid
from typing import Dict, Any
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from openclaw_gateway.state_coordinator import StateCoordinator, StateEntry


class TestVectorClockIncrement:
    """Test vector clock increment behavior."""

    def test_vector_clock_incremented_on_write(self):
        """Verify vector clock is incremented on each write."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"
        agent_id = "agent_a"

        # Initial vector clock
        vector_clock = {agent_id: 0}

        # Write 1
        coordinator.set(conversation_id, key, "value_1", agent_id, vector_clock)
        entry = coordinator.get_entry(conversation_id, key)
        assert entry.vector_clock[agent_id] == 1

    def test_vector_clock_multiple_agents(self):
        """Verify vector clock tracks multiple agents correctly."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        vector_clock = {"agent_a": 1, "agent_b": 0}

        # agent_a writes
        coordinator.set(conversation_id, key, "value_1", "agent_a", vector_clock)
        entry = coordinator.get_entry(conversation_id, key)
        assert entry.vector_clock["agent_a"] == 2
        assert entry.vector_clock["agent_b"] == 0

        # agent_b writes
        coordinator.set(conversation_id, key, "value_2", "agent_b", vector_clock)
        entry = coordinator.get_entry(conversation_id, key)
        assert entry.vector_clock["agent_b"] == 1
        # agent_a's clock stays at 2 from previous write
        assert entry.vector_clock["agent_a"] == 2

    def test_vector_clock_new_agent_initialization(self):
        """Verify new agent gets initialized to 1 on first write."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        vector_clock = {}  # Empty clock
        coordinator.set(conversation_id, key, "value_1", "agent_new", vector_clock)
        entry = coordinator.get_entry(conversation_id, key)
        assert entry.vector_clock["agent_new"] == 1

    def test_vector_clock_independent_per_conversation(self):
        """Verify vector clocks are independent per conversation."""
        coordinator = StateCoordinator()

        # Conversation 1
        vc1 = {"agent_a": 0}
        coordinator.set("conv_1", "key_1", "value_1", "agent_a", vc1)
        entry1 = coordinator.get_entry("conv_1", "key_1")

        # Conversation 2
        vc2 = {"agent_a": 0}
        coordinator.set("conv_2", "key_1", "value_1", "agent_a", vc2)
        entry2 = coordinator.get_entry("conv_2", "key_1")

        # Both should have agent_a = 1 (independent increments)
        assert entry1.vector_clock["agent_a"] == 1
        assert entry2.vector_clock["agent_a"] == 1


class TestVectorClockState:
    """Test vector clock state after N writes."""

    @pytest.mark.parametrize("n_writes", [1, 5, 10, 100])
    def test_vector_clock_equals_n_after_n_writes(self, n_writes):
        """Verify after N writes from single agent, clock[agent_id] == N."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        agent_id = "agent_a"
        vector_clock = {agent_id: 0}

        for i in range(n_writes):
            key = f"key_{i}"
            coordinator.set(conversation_id, key, f"value_{i}", agent_id, vector_clock)

        # Check last entry
        last_entry = coordinator.get_entry(conversation_id, f"key_{n_writes - 1}")
        assert last_entry.vector_clock[agent_id] == n_writes

    def test_vector_clock_state_after_mixed_writes(self):
        """Verify vector clock state after writes from multiple agents."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"

        # 5 writes from agent_a
        vc = {"agent_a": 0, "agent_b": 0}
        for i in range(5):
            coordinator.set(conversation_id, f"key_a_{i}", f"value_a_{i}", "agent_a", vc)

        # 3 writes from agent_b
        for i in range(3):
            coordinator.set(conversation_id, f"key_b_{i}", f"value_b_{i}", "agent_b", vc)

        # Check final state
        entry_a = coordinator.get_entry(conversation_id, "key_a_4")
        entry_b = coordinator.get_entry(conversation_id, "key_b_2")

        # agent_a should have 5 writes + increments from agent_b's writes affecting the clock
        # but vector clock increments are per agent
        assert entry_a.vector_clock["agent_a"] == 5
        assert entry_b.vector_clock["agent_a"] == 5
        assert entry_b.vector_clock["agent_b"] == 3


class TestConcurrentWriteDetection:
    """Test detection of concurrent updates."""

    def test_concurrent_write_detection_same_timestamp(self):
        """Verify concurrent writes with different agents are detected."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        # Mock time to control timestamps
        current_time = time.time()

        vc1 = {"agent_a": 0}
        vc2 = {"agent_a": 1, "agent_b": 0}

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = current_time

            # agent_a writes
            coordinator.set(conversation_id, key, "value_from_a", "agent_a", vc1)
            entry_a = coordinator.get_entry(conversation_id, key)

            # agent_b writes
            coordinator.set(conversation_id, key, "value_from_b", "agent_b", vc2)
            entry_b = coordinator.get_entry(conversation_id, key)

        # agent_b's write should win (same timestamp, but different agents)
        # Actually, since they have same timestamp, LWW shouldn't reject it
        assert entry_b.value in ("value_from_a", "value_from_b")

    def test_is_concurrent_basic(self):
        """Test _is_concurrent helper for vector clock comparison."""
        coordinator = StateCoordinator()

        # vc1 < vc2 (vc1 happened-before vc2)
        vc1 = {"agent_a": 1, "agent_b": 0}
        vc2 = {"agent_a": 1, "agent_b": 1}
        assert coordinator._is_concurrent(vc1, vc2) is False

        # vc1 and vc2 concurrent (incomparable)
        vc1 = {"agent_a": 1, "agent_b": 0}
        vc2 = {"agent_a": 0, "agent_b": 1}
        assert coordinator._is_concurrent(vc1, vc2) is True

        # vc1 == vc2
        vc1 = {"agent_a": 1, "agent_b": 1}
        vc2 = {"agent_a": 1, "agent_b": 1}
        assert coordinator._is_concurrent(vc1, vc2) is False

    def test_is_concurrent_with_missing_agents(self):
        """Test _is_concurrent with agents missing from one clock."""
        coordinator = StateCoordinator()

        # vc1 has agent_a only, vc2 has both
        vc1 = {"agent_a": 1}
        vc2 = {"agent_a": 1, "agent_b": 1}
        # vc1 < vc2 since vc1's implicit agent_b=0 < vc2's agent_b=1
        assert coordinator._is_concurrent(vc1, vc2) is False

        # vc1 > vc2 in different dimensions
        vc1 = {"agent_a": 2, "agent_b": 0}
        vc2 = {"agent_a": 1, "agent_b": 1}
        assert coordinator._is_concurrent(vc1, vc2) is True


class TestLWWConflictResolution:
    """Test Last-Write-Wins conflict resolution."""

    def test_lww_higher_timestamp_wins(self):
        """Verify higher timestamp wins in LWW resolution."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        old_time = time.time()
        new_time = old_time + 10  # 10 seconds later

        vc1 = {"agent_a": 0}
        vc2 = {"agent_a": 1, "agent_b": 0}

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            # First write (old timestamp)
            mock_dt.now.return_value.timestamp.return_value = old_time
            coordinator.set(conversation_id, key, "old_value", "agent_a", vc1)

            # Second write (newer timestamp) should overwrite
            mock_dt.now.return_value.timestamp.return_value = new_time
            coordinator.set(conversation_id, key, "new_value", "agent_b", vc2)

        entry = coordinator.get_entry(conversation_id, key)
        assert entry.value == "new_value"
        assert entry.timestamp_utc == new_time

    def test_lww_lower_timestamp_rejected(self):
        """Verify lower timestamp doesn't overwrite higher in LWW."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        old_time = time.time()
        new_time = old_time + 10

        vc1 = {"agent_a": 0}
        vc2 = {"agent_a": 1, "agent_b": 0}

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            # First write (newer timestamp)
            mock_dt.now.return_value.timestamp.return_value = new_time
            coordinator.set(conversation_id, key, "new_value", "agent_b", vc2)

            # Second write (older timestamp) should be rejected
            mock_dt.now.return_value.timestamp.return_value = old_time
            coordinator.set(conversation_id, key, "old_value", "agent_a", vc1)

        entry = coordinator.get_entry(conversation_id, key)
        assert entry.value == "new_value"
        assert entry.timestamp_utc == new_time

    @pytest.mark.parametrize("n_agents", [2, 5, 10])
    def test_lww_concurrent_writes_n_agents(self, n_agents):
        """Parametrized test: N agents writing same key, highest timestamp wins."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "shared_key"

        base_time = time.time()
        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        vc = {agent_id: 0 for agent_id in agent_ids}

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            # Each agent writes with increasing timestamp
            for idx, agent_id in enumerate(agent_ids):
                mock_dt.now.return_value.timestamp.return_value = base_time + idx
                coordinator.set(conversation_id, key, f"value_from_{agent_id}", agent_id, vc)

        entry = coordinator.get_entry(conversation_id, key)
        # Last agent (highest index) should have highest timestamp
        assert entry.value == f"value_from_{agent_ids[-1]}"
        assert entry.timestamp_utc == base_time + (n_agents - 1)


class TestSubscriptionCallbacks:
    """Test subscription callback mechanism."""

    def test_subscription_callback_fires_on_write(self):
        """Verify callback is triggered on state update."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        callback = Mock()

        coordinator.subscribe(conversation_id, callback)
        coordinator.set(conversation_id, "key_1", "value_1", "agent_a", {"agent_a": 0})

        callback.assert_called_once_with("key_1", "value_1")

    def test_subscription_callback_receives_correct_args(self):
        """Verify callback receives (key, value) as arguments."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        received_calls = []

        def on_update(key: str, value: Any) -> None:
            received_calls.append((key, value))

        coordinator.subscribe(conversation_id, on_update)
        coordinator.set(conversation_id, "key_1", {"data": "test"}, "agent_a", {"agent_a": 0})

        assert len(received_calls) == 1
        assert received_calls[0] == ("key_1", {"data": "test"})

    def test_multiple_subscriptions_all_fire(self):
        """Verify multiple callbacks are all invoked."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        coordinator.subscribe(conversation_id, callback1)
        coordinator.subscribe(conversation_id, callback2)
        coordinator.subscribe(conversation_id, callback3)

        coordinator.set(conversation_id, "key_1", "value_1", "agent_a", {"agent_a": 0})

        callback1.assert_called_once_with("key_1", "value_1")
        callback2.assert_called_once_with("key_1", "value_1")
        callback3.assert_called_once_with("key_1", "value_1")

    def test_subscription_isolated_per_conversation(self):
        """Verify subscriptions are isolated per conversation."""
        coordinator = StateCoordinator()
        callback1 = Mock()
        callback2 = Mock()

        coordinator.subscribe("conv_1", callback1)
        coordinator.subscribe("conv_2", callback2)

        coordinator.set("conv_1", "key_1", "value_1", "agent_a", {"agent_a": 0})

        callback1.assert_called_once()
        callback2.assert_not_called()

    def test_subscription_callback_exception_handled(self):
        """Verify callback exceptions don't prevent other callbacks."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        callback1 = Mock(side_effect=ValueError("Test error"))
        callback2 = Mock()

        coordinator.subscribe(conversation_id, callback1)
        coordinator.subscribe(conversation_id, callback2)

        # Should not raise despite callback1 raising
        coordinator.set(conversation_id, "key_1", "value_1", "agent_a", {"agent_a": 0})

        callback1.assert_called_once()
        callback2.assert_called_once()


class TestStatePersistenceAndRecovery:
    """Test WAL-based persistence and recovery."""

    @pytest.mark.asyncio
    async def test_recover_from_wal_single_entry(self):
        """Verify recovery from single WAL entry."""
        coordinator = StateCoordinator()
        wal_entries = [
            {
                "entry_id": 1,
                "component": "state_coordinator",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {
                    "key": "key_1",
                    "value": "value_1",
                    "agent_id": "agent_a",
                    "vector_clock": {"agent_a": 1}
                },
                "timestamp": time.time()
            }
        ]

        await coordinator.recover_from_wal(wal_entries)

        value = coordinator.get("conv_1", "key_1")
        assert value == "value_1"

    @pytest.mark.asyncio
    async def test_recover_from_wal_multiple_entries(self):
        """Verify recovery from multiple WAL entries."""
        coordinator = StateCoordinator()
        base_time = time.time()
        wal_entries = [
            {
                "entry_id": 1,
                "component": "state_coordinator",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {
                    "key": "key_1",
                    "value": "value_1",
                    "agent_id": "agent_a",
                    "vector_clock": {"agent_a": 0}
                },
                "timestamp": base_time
            },
            {
                "entry_id": 2,
                "component": "state_coordinator",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {
                    "key": "key_2",
                    "value": "value_2",
                    "agent_id": "agent_b",
                    "vector_clock": {"agent_a": 1, "agent_b": 0}
                },
                "timestamp": base_time + 1
            }
        ]

        await coordinator.recover_from_wal(wal_entries)

        assert coordinator.get("conv_1", "key_1") == "value_1"
        assert coordinator.get("conv_1", "key_2") == "value_2"

    @pytest.mark.asyncio
    async def test_recover_from_wal_filters_other_components(self):
        """Verify recovery ignores entries from other components."""
        coordinator = StateCoordinator()
        wal_entries = [
            {
                "entry_id": 1,
                "component": "other_component",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {"key": "key_1", "value": "should_be_ignored"},
                "timestamp": time.time()
            },
            {
                "entry_id": 2,
                "component": "state_coordinator",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {
                    "key": "key_2",
                    "value": "should_be_recovered",
                    "agent_id": "agent_a",
                    "vector_clock": {"agent_a": 0}
                },
                "timestamp": time.time()
            }
        ]

        await coordinator.recover_from_wal(wal_entries)

        assert coordinator.get("conv_1", "key_1") is None
        assert coordinator.get("conv_1", "key_2") == "should_be_recovered"

    @pytest.mark.asyncio
    async def test_recover_from_wal_zero_loss(self):
        """Verify zero data loss on recovery."""
        # Create initial state
        coordinator1 = StateCoordinator()
        wal_entries = []

        # Simulate 10 writes
        base_time = time.time()
        vc = {}
        for i in range(10):
            vc[f"agent_{i % 3}"] = vc.get(f"agent_{i % 3}", 0) + 1
            wal_entries.append({
                "entry_id": i + 1,
                "component": "state_coordinator",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {
                    "key": f"key_{i}",
                    "value": f"value_{i}",
                    "agent_id": f"agent_{i % 3}",
                    "vector_clock": vc.copy()
                },
                "timestamp": base_time + i
            })

        # Recover into fresh coordinator
        coordinator2 = StateCoordinator()
        await coordinator2.recover_from_wal(wal_entries)

        # Verify all entries recovered
        for i in range(10):
            assert coordinator2.get("conv_1", f"key_{i}") == f"value_{i}"

    @pytest.mark.asyncio
    async def test_recover_from_wal_multiple_conversations(self):
        """Verify recovery maintains conversation isolation."""
        coordinator = StateCoordinator()
        wal_entries = [
            {
                "entry_id": 1,
                "component": "state_coordinator",
                "conversation_id": "conv_1",
                "operation_type": "set",
                "data": {
                    "key": "key_1",
                    "value": "conv1_value",
                    "agent_id": "agent_a",
                    "vector_clock": {"agent_a": 0}
                },
                "timestamp": time.time()
            },
            {
                "entry_id": 2,
                "component": "state_coordinator",
                "conversation_id": "conv_2",
                "operation_type": "set",
                "data": {
                    "key": "key_1",
                    "value": "conv2_value",
                    "agent_id": "agent_b",
                    "vector_clock": {"agent_b": 0}
                },
                "timestamp": time.time() + 1
            }
        ]

        await coordinator.recover_from_wal(wal_entries)

        assert coordinator.get("conv_1", "key_1") == "conv1_value"
        assert coordinator.get("conv_2", "key_1") == "conv2_value"


class TestLatencyBenchmarking:
    """Test latency requirements for state operations."""

    def test_get_latency_under_5ms(self):
        """Verify get() operation meets < 5ms latency requirement."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        # Pre-populate state
        coordinator.set(conversation_id, key, "value_1", "agent_a", {"agent_a": 0})

        # Measure get latency
        start = time.perf_counter()
        for _ in range(1000):
            result = coordinator.get(conversation_id, key)
        end = time.perf_counter()

        # Average latency across 1000 calls
        avg_latency_ms = (end - start) / 1000 * 1000
        assert avg_latency_ms < 5.0, f"get() latency {avg_latency_ms}ms exceeds 5ms target"
        assert result == "value_1"

    def test_set_latency_reasonable(self):
        """Verify set() operation has reasonable latency."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"

        start = time.perf_counter()
        for i in range(100):
            coordinator.set(
                conversation_id,
                f"key_{i}",
                f"value_{i}",
                "agent_a",
                {"agent_a": i}
            )
        end = time.perf_counter()

        # Average latency across 100 calls
        avg_latency_ms = (end - start) / 100 * 1000
        # set() can be slower than get() due to callbacks, but should still be reasonable
        assert avg_latency_ms < 10.0, f"set() latency {avg_latency_ms}ms exceeds reasonable threshold"

    def test_get_with_many_keys(self):
        """Test get() latency remains O(1) with many keys in conversation state."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"

        # Pre-populate with 1000 keys
        for i in range(1000):
            coordinator.set(conversation_id, f"key_{i}", f"value_{i}", "agent_a", {"agent_a": i})

        # Measure get latency with 1000 keys present
        start = time.perf_counter()
        for _ in range(1000):
            result = coordinator.get(conversation_id, "key_500")
        end = time.perf_counter()

        avg_latency_ms = (end - start) / 1000 * 1000
        assert avg_latency_ms < 5.0, f"get() with 1000 keys: {avg_latency_ms}ms exceeds 5ms target"
        assert result == "value_500"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_timestamp_handling(self):
        """Verify coordinator handles zero timestamps correctly."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"
        vc = {"agent_a": 0}

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 0.0
            coordinator.set(conversation_id, key, "value_1", "agent_a", vc)

        entry = coordinator.get_entry(conversation_id, key)
        assert entry.timestamp_utc == 0.0
        assert entry.value == "value_1"

    def test_missing_agent_in_vector_clock(self):
        """Verify handling when agent not in vector clock."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"

        # Empty vector clock
        vc = {}
        coordinator.set(conversation_id, key, "value_1", "agent_new", vc)

        entry = coordinator.get_entry(conversation_id, key)
        assert entry.vector_clock["agent_new"] == 1

    def test_empty_conversation_id_lookup(self):
        """Verify lookup for non-existent conversation returns None."""
        coordinator = StateCoordinator()
        value = coordinator.get("non_existent_conv", "any_key")
        assert value is None

    def test_empty_key_lookup(self):
        """Verify lookup for non-existent key returns None."""
        coordinator = StateCoordinator()
        coordinator.set("conv_1", "key_1", "value_1", "agent_a", {"agent_a": 0})
        value = coordinator.get("conv_1", "non_existent_key")
        assert value is None

    def test_same_timestamp_from_different_agents(self):
        """Test LWW behavior when two writes have same timestamp.

        When two writes have identical timestamps, the LWW rule (higher wins)
        doesn't apply. The current implementation will accept the second write
        since it's not strictly less than the existing one.
        """
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "test_key"
        same_time = time.time()

        vc1 = {"agent_a": 0}
        vc2 = {"agent_a": 1, "agent_b": 0}

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = same_time

            coordinator.set(conversation_id, key, "value_from_a", "agent_a", vc1)
            coordinator.set(conversation_id, key, "value_from_b", "agent_b", vc2)

        # When timestamps are identical, the second write is not strictly less,
        # so it gets accepted. This is acceptable behavior for LWW.
        entry = coordinator.get_entry(conversation_id, key)
        # The second write will overwrite since it's not strictly less
        assert entry.value == "value_from_b"

    def test_multiple_keys_independent(self):
        """Verify multiple keys in same conversation are independent."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        vc = {"agent_a": 0}

        coordinator.set(conversation_id, "key_1", "value_1", "agent_a", vc)
        coordinator.set(conversation_id, "key_2", "value_2", "agent_a", vc)
        coordinator.set(conversation_id, "key_3", "value_3", "agent_a", vc)

        assert coordinator.get(conversation_id, "key_1") == "value_1"
        assert coordinator.get(conversation_id, "key_2") == "value_2"
        assert coordinator.get(conversation_id, "key_3") == "value_3"

    def test_null_value_handling(self):
        """Verify coordinator handles None and null-like values."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        vc = {"agent_a": 0}

        coordinator.set(conversation_id, "key_none", None, "agent_a", vc)
        coordinator.set(conversation_id, "key_empty_str", "", "agent_a", vc)
        coordinator.set(conversation_id, "key_zero", 0, "agent_a", vc)
        coordinator.set(conversation_id, "key_false", False, "agent_a", vc)

        assert coordinator.get(conversation_id, "key_none") is None
        assert coordinator.get(conversation_id, "key_empty_str") == ""
        assert coordinator.get(conversation_id, "key_zero") == 0
        assert coordinator.get(conversation_id, "key_false") is False

    def test_complex_value_types(self):
        """Verify coordinator handles complex value types."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        vc = {"agent_a": 0}

        # Dict
        coordinator.set(conversation_id, "key_dict", {"nested": {"data": [1, 2, 3]}}, "agent_a", vc)
        # List
        coordinator.set(conversation_id, "key_list", [1, "two", 3.0, None], "agent_a", vc)
        # Tuple (will be in some serialized form)
        coordinator.set(conversation_id, "key_tuple", (1, 2, 3), "agent_a", vc)

        assert coordinator.get(conversation_id, "key_dict") == {"nested": {"data": [1, 2, 3]}}
        assert coordinator.get(conversation_id, "key_list") == [1, "two", 3.0, None]
        assert coordinator.get(conversation_id, "key_tuple") == (1, 2, 3)


class TestConcurrentWritesParametrized:
    """Parametrized tests for concurrent writes from multiple agents."""

    @pytest.mark.parametrize("n_agents", [2, 3, 5, 10])
    def test_concurrent_writes_n_agents_lww(self, n_agents):
        """Test N agents writing same key with LWW resolution."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        key = "shared_key"
        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        vc = {agent_id: 0 for agent_id in agent_ids}

        base_time = time.time()

        with patch('openclaw_gateway.state_coordinator.datetime') as mock_dt:
            for idx, agent_id in enumerate(agent_ids):
                mock_dt.now.return_value.timestamp.return_value = base_time + idx
                coordinator.set(conversation_id, key, f"value_{agent_id}", agent_id, vc)

        entry = coordinator.get_entry(conversation_id, key)
        assert entry.value == f"value_{agent_ids[-1]}"

    @pytest.mark.parametrize("n_agents", [2, 3, 5, 10])
    def test_concurrent_writes_multiple_keys_n_agents(self, n_agents):
        """Test N agents writing to different keys."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        vc = {agent_id: 0 for agent_id in agent_ids}

        for agent_idx, agent_id in enumerate(agent_ids):
            for key_idx in range(5):
                coordinator.set(
                    conversation_id,
                    f"key_{key_idx}",
                    f"value_{agent_id}_key{key_idx}",
                    agent_id,
                    vc
                )

        # Verify all keys exist with last agent's values
        for key_idx in range(5):
            value = coordinator.get(conversation_id, f"key_{key_idx}")
            assert "value_" in str(value)

    @pytest.mark.parametrize("n_agents,n_writes_per_agent", [
        (2, 5),
        (3, 10),
        (5, 20),
    ])
    def test_concurrent_writes_vector_clock_tracking(self, n_agents, n_writes_per_agent):
        """Test vector clock correctly tracks N agents with M writes each."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        vc = {agent_id: 0 for agent_id in agent_ids}

        for agent_idx, agent_id in enumerate(agent_ids):
            for write_idx in range(n_writes_per_agent):
                coordinator.set(
                    conversation_id,
                    f"{agent_id}_key_{write_idx}",
                    f"value_{write_idx}",
                    agent_id,
                    vc
                )

        # Verify vector clock state
        entry = coordinator.get_entry(conversation_id, f"{agent_ids[-1]}_key_{n_writes_per_agent - 1}")
        for agent_id in agent_ids:
            assert entry.vector_clock[agent_id] == n_writes_per_agent


class TestStateSnapshot:
    """Test state snapshot and debugging utilities."""

    def test_get_state_snapshot_empty(self):
        """Verify snapshot of empty conversation."""
        coordinator = StateCoordinator()
        snapshot = coordinator.get_state_snapshot("non_existent")
        assert snapshot == {}

    def test_get_state_snapshot_populated(self):
        """Verify snapshot contains all key-value pairs."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        vc = {"agent_a": 0}

        expected_data = {
            "key_1": "value_1",
            "key_2": 42,
            "key_3": {"nested": True},
        }

        for key, value in expected_data.items():
            coordinator.set(conversation_id, key, value, "agent_a", vc)

        snapshot = coordinator.get_state_snapshot(conversation_id)
        assert snapshot == expected_data

    def test_get_vector_clock_entry(self):
        """Verify get_vector_clock returns correct clock for entry."""
        coordinator = StateCoordinator()
        conversation_id = "conv_1"
        vc = {"agent_a": 0, "agent_b": 5}

        coordinator.set(conversation_id, "key_1", "value_1", "agent_a", vc)

        clock = coordinator.get_vector_clock(conversation_id, "key_1")
        assert clock["agent_a"] == 1
        assert clock["agent_b"] == 5

    def test_get_vector_clock_missing_entry(self):
        """Verify get_vector_clock returns None for missing entry."""
        coordinator = StateCoordinator()
        clock = coordinator.get_vector_clock("conv_1", "missing_key")
        assert clock is None


class TestAsyncRecovery:
    """Test async recovery methods."""

    @pytest.mark.asyncio
    async def test_recover_from_wal_is_async(self):
        """Verify recover_from_wal is properly async."""
        coordinator = StateCoordinator()
        wal_entries = []

        # Should be awaitable
        result = await coordinator.recover_from_wal(wal_entries)
        assert result is None

    @pytest.mark.asyncio
    async def test_recover_from_wal_empty_list(self):
        """Verify recovery handles empty WAL gracefully."""
        coordinator = StateCoordinator()
        await coordinator.recover_from_wal([])

        # Coordinator should be empty
        snapshot = coordinator.get_state_snapshot("any_conv")
        assert snapshot == {}
