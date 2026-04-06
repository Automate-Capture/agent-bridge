"""
Unit tests for StateCoordinator with vector clocks and LWW conflict resolution.

Tests verify:
1. Vector clock increment (N writes -> clock[agent_id]==N)
2. Concurrent writes from 5 agents with LWW winner selection
3. State persistence and recovery from WAL
4. Subscription callback firing
5. State lookup latency < 5ms
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from openclaw_gateway.state_coordinator import StateCoordinator, StateEntry


class TestVectorClockIncrement:
    """Test vector clock increment correctness."""

    def test_vector_clock_incremented_single_agent(self):
        """After N writes from single agent, clock[agent_id] == N."""
        coordinator = StateCoordinator()
        conversation_id = "conv_123"
        agent_id = "agent_1"

        vector_clock = {agent_id: 0}

        for i in range(1, 6):
            coordinator.set(
                conversation_id=conversation_id,
                key=f"key_{i}",
                value=f"value_{i}",
                agent_id=agent_id,
                vector_clock=vector_clock,
            )
            assert vector_clock[agent_id] == i, f"Expected {i}, got {vector_clock[agent_id]}"

    def test_vector_clock_incremented_multiple_agents(self):
        """Multiple agents independently increment their own clock entries."""
        coordinator = StateCoordinator()
        conversation_id = "conv_456"

        vc = {"agent_1": 0, "agent_2": 0}

        # Agent 1 writes twice
        coordinator.set(conversation_id, "key_1", "val_1", "agent_1", vc)
        assert vc["agent_1"] == 1
        assert vc["agent_2"] == 0

        coordinator.set(conversation_id, "key_2", "val_2", "agent_1", vc)
        assert vc["agent_1"] == 2
        assert vc["agent_2"] == 0

        # Agent 2 writes once
        coordinator.set(conversation_id, "key_3", "val_3", "agent_2", vc)
        assert vc["agent_1"] == 2
        assert vc["agent_2"] == 1

    def test_vector_clock_initialized_if_missing(self):
        """If agent_id not in clock, it's initialized to 0 then incremented."""
        coordinator = StateCoordinator()
        vc = {"agent_1": 1}

        coordinator.set("conv_789", "key", "val", "agent_2", vc)
        assert vc["agent_2"] == 1
        assert vc["agent_1"] == 1

    def test_vector_clock_with_none_initialization(self):
        """If vector_clock=None passed, it's treated as empty dict."""
        coordinator = StateCoordinator()

        coordinator.set("conv_000", "key", "val", "agent_1", vector_clock=None)

        vc = coordinator.get_vector_clock("conv_000")
        assert vc["agent_1"] == 1


class TestLWWConflictResolution:
    """Test Last-Write-Wins conflict resolution."""

    def test_lww_higher_timestamp_wins(self):
        """Higher timestamp always wins in concurrent write."""
        coordinator = StateCoordinator()
        conversation_id = "conv_lww"

        vc1 = {"agent_1": 1}
        vc2 = {"agent_2": 1}

        # Agent 1 writes at T=100
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 100.0
            coordinator.set(
                conversation_id, "key", "value_from_agent_1", "agent_1", vc1
            )

        assert coordinator.get(conversation_id, "key") == "value_from_agent_1"

        # Agent 2 writes at T=101 (later)
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 101.0
            coordinator.set(
                conversation_id, "key", "value_from_agent_2", "agent_2", vc2
            )

        # Higher timestamp (agent_2's) should win
        assert coordinator.get(conversation_id, "key") == "value_from_agent_2"

    def test_lww_stale_write_ignored(self):
        """Stale write (lower timestamp) is ignored."""
        coordinator = StateCoordinator()
        conversation_id = "conv_stale"

        vc1 = {"agent_1": 1}
        vc2 = {"agent_2": 1}

        # Write with agent_1 at T=200
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 200.0
            coordinator.set(conversation_id, "key", "newer_value", "agent_1", vc1)

        # Try to write with agent_2 at T=199 (earlier)
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 199.0
            coordinator.set(
                conversation_id, "key", "older_value", "agent_2", vc2
            )

        # Original (newer) value should remain
        assert coordinator.get(conversation_id, "key") == "newer_value"

    def test_lww_same_timestamp_keeps_existing(self):
        """Same timestamp: keep existing value (deterministic choice)."""
        coordinator = StateCoordinator()
        conversation_id = "conv_tie"

        vc1 = {"agent_1": 1}
        vc2 = {"agent_2": 1}

        # First write at T=300
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 300.0
            coordinator.set(conversation_id, "key", "first_value", "agent_1", vc1)

        # Second write at same T=300
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 300.0
            coordinator.set(
                conversation_id, "key", "second_value", "agent_2", vc2
            )

        # First value should remain (existing entry kept on tie)
        assert coordinator.get(conversation_id, "key") == "first_value"


class TestSubscriptionCallbacks:
    """Test subscription callback firing."""

    def test_subscription_callback_fires_on_update(self):
        """Callback is called with (key, value) on state update."""
        coordinator = StateCoordinator()
        conversation_id = "conv_callback"

        callback = MagicMock()
        coordinator.subscribe(conversation_id, callback)

        coordinator.set(conversation_id, "key", "value", "agent_1", {})

        callback.assert_called_once_with("key", "value")

    def test_subscription_multiple_callbacks(self):
        """Multiple callbacks all fire."""
        coordinator = StateCoordinator()
        conversation_id = "conv_multi"

        callback1 = MagicMock()
        callback2 = MagicMock()
        coordinator.subscribe(conversation_id, callback1)
        coordinator.subscribe(conversation_id, callback2)

        coordinator.set(conversation_id, "key", "value", "agent_1", {})

        callback1.assert_called_once_with("key", "value")
        callback2.assert_called_once_with("key", "value")

    def test_subscription_callback_exception_handled(self):
        """Exception in callback doesn't prevent other callbacks."""
        coordinator = StateCoordinator()
        conversation_id = "conv_except"

        def failing_callback(key, value):
            raise RuntimeError("Callback failed")

        callback_good = MagicMock()
        coordinator.subscribe(conversation_id, failing_callback)
        coordinator.subscribe(conversation_id, callback_good)

        # Should not raise even though failing_callback raises
        coordinator.set(conversation_id, "key", "value", "agent_1", {})

        callback_good.assert_called_once_with("key", "value")

    def test_subscription_no_callback_on_lww_ignore(self):
        """When LWW ignores stale write, callback NOT fired."""
        coordinator = StateCoordinator()
        conversation_id = "conv_no_callback"

        callback = MagicMock()
        coordinator.subscribe(conversation_id, callback)

        # Write newer value at T=400
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 400.0
            coordinator.set(conversation_id, "key", "newer", "agent_1", {})
        callback.assert_called_once()

        callback.reset_mock()

        # Try to write older value at T=399
        with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value.timestamp.return_value = 399.0
            coordinator.set(conversation_id, "key", "older", "agent_2", {})

        # Callback should NOT be called (LWW ignored the stale write)
        callback.assert_not_called()


class TestConcurrentUpdates:
    """Test concurrent write safety."""

    def test_concurrent_5_agents_lww_winner(self):
        """5 agents write same key concurrently; highest timestamp wins."""
        coordinator = StateCoordinator()
        conversation_id = "conv_5agents"

        agents = [f"agent_{i}" for i in range(1, 6)]

        # All write with initial vector clock
        for i, agent in enumerate(agents):
            # Stagger timestamps: agent_1 at 500, agent_2 at 501, etc.
            with patch("openclaw_gateway.state_coordinator.datetime") as mock_dt:
                mock_dt.now.return_value.timestamp.return_value = 500.0 + i
                vc = {agent: 0}
                coordinator.set(
                    conversation_id,
                    "shared_key",
                    f"value_from_{agent}",
                    agent,
                    vc,
                )

        # Agent with highest timestamp (agent_5, at 504) should win
        assert coordinator.get(conversation_id, "shared_key") == "value_from_agent_5"

    def test_concurrent_5_agents_vector_clocks(self):
        """5 agents can track independent vector clock increments."""
        coordinator = StateCoordinator()
        conversation_id = "conv_vc_5"

        vc = {f"agent_{i}": 0 for i in range(1, 6)}

        # Each agent writes twice
        for i in range(1, 6):
            agent = f"agent_{i}"
            coordinator.set(conversation_id, f"{agent}_key1", "val1", agent, vc)
            coordinator.set(conversation_id, f"{agent}_key2", "val2", agent, vc)

        # Each agent should have clock=2
        for i in range(1, 6):
            assert vc[f"agent_{i}"] == 2

    def test_get_vector_clock_aggregates(self):
        """get_vector_clock returns max per agent."""
        coordinator = StateCoordinator()
        conversation_id = "conv_agg"

        # Simulate writes with different vector clocks
        # set() increments the agent's clock, so initial state is important
        vc1 = {"agent_1": 0, "agent_2": 0}
        coordinator.set(conversation_id, "key_1", "val_1", "agent_1", vc1)
        # vc1 is now {"agent_1": 1, "agent_2": 0}

        vc2 = {"agent_1": 1, "agent_2": 0}
        coordinator.set(conversation_id, "key_2", "val_2", "agent_2", vc2)
        # vc2 is now {"agent_1": 1, "agent_2": 1}

        aggregated = coordinator.get_vector_clock(conversation_id)
        # Should be max of each agent across all state entries
        # key_1 stored with vc {"agent_1": 1, "agent_2": 0}
        # key_2 stored with vc {"agent_1": 1, "agent_2": 1}
        # max: agent_1=1, agent_2=1
        assert aggregated["agent_1"] == 1
        assert aggregated["agent_2"] == 1


class TestStateLookup:
    """Test state lookup performance."""

    def test_state_lookup_latency_under_5ms(self):
        """State lookup is O(1) and completes in < 5ms."""
        coordinator = StateCoordinator()
        conversation_id = "conv_perf"

        # Pre-populate with many entries
        for i in range(1000):
            coordinator.set(
                conversation_id,
                f"key_{i}",
                f"value_{i}",
                "agent_1",
                {},
            )

        # Measure lookup latency
        start = time.perf_counter()
        for i in range(100):
            value = coordinator.get(conversation_id, f"key_{i}")
            assert value is not None
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000
        assert avg_latency_ms < 5, f"Lookup latency {avg_latency_ms}ms exceeds 5ms"

    def test_get_nonexistent_key_returns_none(self):
        """Get for non-existent key returns None."""
        coordinator = StateCoordinator()

        result = coordinator.get("nonexistent_conv", "nonexistent_key")
        assert result is None

    def test_get_nonexistent_conversation_returns_none(self):
        """Get for non-existent conversation returns None."""
        coordinator = StateCoordinator()
        coordinator.set("conv_1", "key", "val", "agent_1", {})

        result = coordinator.get("conv_2", "key")
        assert result is None

    def test_get_vector_clock_nonexistent_conversation(self):
        """get_vector_clock for non-existent conversation returns empty dict."""
        coordinator = StateCoordinator()

        result = coordinator.get_vector_clock("nonexistent_conv")
        assert result == {}


class TestWALRecovery:
    """Test WAL recovery of state."""

    @pytest.mark.asyncio
    async def test_wal_recovery_restores_state(self):
        """WAL recovery restores state and vector clocks exactly."""
        coordinator = StateCoordinator()
        conversation_id = "conv_wal"

        # Simulate WAL entries as they would be read from file
        wal_entries = [
            {
                "component": "state_coordinator",
                "conversation_id": conversation_id,
                "entry_id": 1,
                "operation_type": "set",
                "data": {
                    "key": "key_1",
                    "value": "value_1",
                    "agent_id": "agent_1",
                    "vector_clock": {"agent_1": 1},
                    "timestamp_utc": 100.0,
                },
            },
            {
                "component": "state_coordinator",
                "conversation_id": conversation_id,
                "entry_id": 2,
                "operation_type": "set",
                "data": {
                    "key": "key_2",
                    "value": "value_2",
                    "agent_id": "agent_2",
                    "vector_clock": {"agent_1": 1, "agent_2": 1},
                    "timestamp_utc": 101.0,
                },
            },
            {
                "component": "state_coordinator",
                "conversation_id": conversation_id,
                "entry_id": 3,
                "operation_type": "set",
                "data": {
                    "key": "key_3",
                    "value": "value_3",
                    "agent_id": "agent_1",
                    "vector_clock": {"agent_1": 2, "agent_2": 1},
                    "timestamp_utc": 102.0,
                },
            },
        ]

        # Recover from WAL
        await coordinator.recover_from_wal(wal_entries)

        # Verify state reconstructed
        assert coordinator.get(conversation_id, "key_1") == "value_1"
        assert coordinator.get(conversation_id, "key_2") == "value_2"
        assert coordinator.get(conversation_id, "key_3") == "value_3"

        # Verify vector clock reconstructed exactly
        vc = coordinator.get_vector_clock(conversation_id)
        assert vc["agent_1"] == 2
        assert vc["agent_2"] == 1

    @pytest.mark.asyncio
    async def test_wal_recovery_ignores_other_components(self):
        """WAL recovery filters by component, ignores other entries."""
        coordinator = StateCoordinator()

        wal_entries = [
            {
                "component": "context_manager",
                "conversation_id": "conv_x",
                "entry_id": 1,
                "operation_type": "store",
                "data": {"key": "data"},
            },
            {
                "component": "state_coordinator",
                "conversation_id": "conv_y",
                "entry_id": 2,
                "operation_type": "set",
                "data": {
                    "key": "state_key",
                    "value": "state_val",
                    "agent_id": "agent_1",
                    "vector_clock": {"agent_1": 1},
                },
            },
        ]

        await coordinator.recover_from_wal(wal_entries)

        # Only state_coordinator entry should be recovered
        assert coordinator.get("conv_y", "state_key") == "state_val"
        assert coordinator.get("conv_x", "key") is None

    @pytest.mark.asyncio
    async def test_wal_recovery_zero_loss(self):
        """Recovery from 10 state updates restores all exactly."""
        coordinator = StateCoordinator()
        conversation_id = "conv_zeroloss"

        # Create 10 WAL entries
        wal_entries = []
        vc = {"agent_1": 0}
        for i in range(1, 11):
            coordinator.set(
                conversation_id,
                f"key_{i}",
                f"value_{i}",
                "agent_1",
                vc,
            )
            wal_entries.append(
                {
                    "component": "state_coordinator",
                    "conversation_id": conversation_id,
                    "entry_id": i,
                    "operation_type": "set",
                    "data": {
                        "key": f"key_{i}",
                        "value": f"value_{i}",
                        "agent_id": "agent_1",
                        "vector_clock": {"agent_1": i},
                        "timestamp_utc": 100.0 + i,
                    },
                }
            )

        # Simulate restart: create new coordinator and recover
        new_coordinator = StateCoordinator()
        await new_coordinator.recover_from_wal(wal_entries)

        # Verify all 10 keys recovered exactly
        for i in range(1, 11):
            assert new_coordinator.get(conversation_id, f"key_{i}") == f"value_{i}"

        # Verify vector clock recovered exactly
        vc_recovered = new_coordinator.get_vector_clock(conversation_id)
        assert vc_recovered["agent_1"] == 10

    @pytest.mark.asyncio
    async def test_wal_recovery_updates_next_wal_id(self):
        """WAL recovery updates _next_wal_id to continue from last entry."""
        coordinator = StateCoordinator()

        wal_entries = [
            {
                "component": "state_coordinator",
                "conversation_id": "conv_id",
                "entry_id": 100,
                "operation_type": "set",
                "data": {
                    "key": "k",
                    "value": "v",
                    "agent_id": "a",
                    "vector_clock": {},
                },
            }
        ]

        await coordinator.recover_from_wal(wal_entries)
        assert coordinator._next_wal_id == 101


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_vector_clock_initialization(self):
        """Set with empty vector clock initializes agent to 1."""
        coordinator = StateCoordinator()
        vc = {}

        coordinator.set("conv", "key", "val", "agent_1", vc)
        assert vc["agent_1"] == 1

    def test_multiple_conversations_isolated(self):
        """State in different conversations is isolated."""
        coordinator = StateCoordinator()

        coordinator.set("conv_1", "key", "value_1", "agent_1", {})
        coordinator.set("conv_2", "key", "value_2", "agent_1", {})

        assert coordinator.get("conv_1", "key") == "value_1"
        assert coordinator.get("conv_2", "key") == "value_2"

    def test_overwrite_same_key_same_agent(self):
        """Overwriting same key from same agent increments clock."""
        coordinator = StateCoordinator()
        vc = {"agent_1": 0}

        coordinator.set("conv", "key", "value_1", "agent_1", vc)
        assert vc["agent_1"] == 1
        assert coordinator.get("conv", "key") == "value_1"

        coordinator.set("conv", "key", "value_2", "agent_1", vc)
        assert vc["agent_1"] == 2
        assert coordinator.get("conv", "key") == "value_2"

    def test_vector_clock_not_mutated_by_copy(self):
        """Internal copy of vector clock prevents external mutation."""
        coordinator = StateCoordinator()
        vc = {"agent_1": 0}

        coordinator.set("conv", "key", "val", "agent_1", vc)

        # Mutate external clock
        vc["agent_2"] = 999

        # Internal state should have had a copy at time of set
        vc_internal = coordinator.get_vector_clock("conv")
        assert "agent_2" not in vc_internal
