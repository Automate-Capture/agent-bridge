"""Tests for protocol negotiation manager.

This module tests the ProtocolNegotiationManager and AgentCapabilities classes,
including capability registration, capability matrix population, protocol selection
logic (preferred, common, fallback), latency benchmarks, and edge cases.
"""

import pytest
import logging
import time
from unittest.mock import patch
from openclaw_gateway.negotiation import ProtocolNegotiationManager, AgentCapabilities


class TestAgentCapabilities:
    """Tests for AgentCapabilities dataclass."""

    def test_agent_capabilities_creation(self):
        """Test creating an AgentCapabilities instance."""
        caps = AgentCapabilities(
            agent_id="test_agent",
            supported_protocols=["langchain", "event_stream"]
        )
        assert caps.agent_id == "test_agent"
        assert caps.supported_protocols == ["langchain", "event_stream"]

    def test_agent_capabilities_with_single_protocol(self):
        """Test AgentCapabilities with single protocol."""
        caps = AgentCapabilities(
            agent_id="single_proto_agent",
            supported_protocols=["langchain"]
        )
        assert len(caps.supported_protocols) == 1
        assert "langchain" == caps.supported_protocols[0]

    def test_agent_capabilities_with_multiple_protocols(self):
        """Test AgentCapabilities with multiple protocols."""
        protocols = ["langchain", "event_stream", "custom_proto"]
        caps = AgentCapabilities(
            agent_id="multi_agent",
            supported_protocols=protocols
        )
        assert len(caps.supported_protocols) == 3
        assert set(caps.supported_protocols) == set(protocols)


class TestProtocolNegotiationManagerRegistration:
    """Tests for agent registration and capability matrix management."""

    def test_manager_initialization(self):
        """Test ProtocolNegotiationManager initialization."""
        manager = ProtocolNegotiationManager()
        assert manager._capabilities == {}
        assert manager._capability_matrix == {}

    def test_register_single_agent(self):
        """Test registering a single agent with capabilities."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_1",
            supported_protocols=["langchain", "event_stream"]
        )
        manager.register_capabilities(caps)

        assert "agent_1" in manager._capabilities
        assert manager._capabilities["agent_1"] == caps
        assert "agent_1" in manager._capability_matrix

    def test_capability_matrix_populated_correctly(self):
        """Test that capability matrix is correctly populated on registration."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_1",
            supported_protocols=["langchain", "event_stream"]
        )
        manager.register_capabilities(caps)

        # Verify matrix entries
        assert manager._capability_matrix["agent_1"]["langchain"] == 1.0
        assert manager._capability_matrix["agent_1"]["event_stream"] == 1.0
        assert len(manager._capability_matrix["agent_1"]) == 2

    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        manager = ProtocolNegotiationManager()

        caps1 = AgentCapabilities(
            agent_id="agent_1",
            supported_protocols=["langchain"]
        )
        caps2 = AgentCapabilities(
            agent_id="agent_2",
            supported_protocols=["event_stream"]
        )

        manager.register_capabilities(caps1)
        manager.register_capabilities(caps2)

        assert len(manager._capabilities) == 2
        assert len(manager._capability_matrix) == 2
        assert "agent_1" in manager._capabilities
        assert "agent_2" in manager._capabilities

    def test_capability_matrix_with_multiple_protocols(self):
        """Test capability matrix with agents having multiple protocols."""
        manager = ProtocolNegotiationManager()

        caps = AgentCapabilities(
            agent_id="multi_agent",
            supported_protocols=["langchain", "event_stream", "custom"]
        )
        manager.register_capabilities(caps)

        matrix = manager._capability_matrix["multi_agent"]
        assert matrix["langchain"] == 1.0
        assert matrix["event_stream"] == 1.0
        assert matrix["custom"] == 1.0
        assert len(matrix) == 3

    @pytest.mark.parametrize("protocols", [
        ["langchain"],
        ["langchain", "event_stream"],
        ["proto1", "proto2", "proto3", "proto4"],
    ])
    def test_register_agent_parametrized_protocols(self, protocols):
        """Test registration with various protocol combinations."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="test_agent",
            supported_protocols=protocols
        )
        manager.register_capabilities(caps)

        matrix = manager._capability_matrix["test_agent"]
        assert len(matrix) == len(protocols)
        for proto in protocols:
            assert matrix[proto] == 1.0


class TestProtocolSelection:
    """Tests for protocol selection logic."""

    def test_select_protocol_preferred_available(self):
        """Test protocol selection when preferred protocol is available in target."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain", "event_stream"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        selected = manager.select_protocol("source", "target", preferred_protocol="langchain")
        assert selected == "langchain"

    def test_select_protocol_preferred_unavailable_falls_to_common(self):
        """Test protocol selection when preferred is unavailable but common exists."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain", "event_stream"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream", "custom"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        # Request langchain but it's not in target's list
        selected = manager.select_protocol("source", "target", preferred_protocol="langchain")
        assert selected == "event_stream"  # Common protocol

    def test_select_protocol_no_preferred_selects_common(self):
        """Test protocol selection without preferred protocol selects common."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain", "event_stream", "custom"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream", "custom"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        # No preferred, should select from common
        selected = manager.select_protocol("source", "target")
        assert selected in ["event_stream", "custom"]

    def test_select_protocol_no_common_uses_fallback(self):
        """Test protocol selection with no common protocol uses fallback."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream", "custom"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        # No common protocol, should use fallback (first in target's list)
        selected = manager.select_protocol("source", "target")
        assert selected == "event_stream"

    def test_select_protocol_fallback_logs_warning(self):
        """Test that fallback protocol selection logs a warning."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        with patch('openclaw_gateway.negotiation.logger') as mock_logger:
            selected = manager.select_protocol("source", "target")
            assert selected == "event_stream"
            mock_logger.warning.assert_called()

    def test_select_protocol_unregistered_target_raises_error(self):
        """Test that selecting protocol for unregistered target raises ValueError."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain"]
        )
        manager.register_capabilities(source_caps)

        with pytest.raises(ValueError, match="No capabilities registered for target agent"):
            manager.select_protocol("source", "unregistered_target")

    def test_select_protocol_unregistered_source_uses_fallback(self):
        """Test that unregistered source agent uses fallback protocol."""
        manager = ProtocolNegotiationManager()

        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"]
        )
        manager.register_capabilities(target_caps)

        # Source not registered, should still work using fallback
        selected = manager.select_protocol("unregistered_source", "target")
        assert selected == "langchain"  # First in target's list


class TestProtocolSelectionWithMultipleFeatures:
    """Tests for multi-feature agent support."""

    def test_multi_feature_agent_registration(self):
        """Test agent with multiple features advertises all protocols."""
        manager = ProtocolNegotiationManager()

        caps = AgentCapabilities(
            agent_id="multi_feature_agent",
            supported_protocols=["streaming", "task_decomposition", "batch_processing"]
        )
        manager.register_capabilities(caps)

        matrix = manager._capability_matrix["multi_feature_agent"]
        assert len(matrix) == 3
        assert all(score == 1.0 for score in matrix.values())

    @pytest.mark.parametrize("num_protocols", [1, 2, 3, 4, 5])
    def test_multi_feature_agent_parametrized(self, num_protocols):
        """Test agents with varying number of protocols."""
        manager = ProtocolNegotiationManager()

        protocols = [f"proto_{i}" for i in range(num_protocols)]
        caps = AgentCapabilities(
            agent_id="test_agent",
            supported_protocols=protocols
        )
        manager.register_capabilities(caps)

        assert len(manager._capability_matrix["test_agent"]) == num_protocols

    def test_multi_agent_with_overlapping_protocols(self):
        """Test multiple agents with overlapping protocol support."""
        manager = ProtocolNegotiationManager()

        caps1 = AgentCapabilities(
            agent_id="agent_1",
            supported_protocols=["langchain", "event_stream", "custom"]
        )
        caps2 = AgentCapabilities(
            agent_id="agent_2",
            supported_protocols=["event_stream", "custom", "rpc"]
        )

        manager.register_capabilities(caps1)
        manager.register_capabilities(caps2)

        # Should select from common protocols
        selected = manager.select_protocol("agent_1", "agent_2")
        assert selected in ["event_stream", "custom"]


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_select_protocol_with_none_preferred(self):
        """Test protocol selection with explicit None preferred_protocol."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        selected = manager.select_protocol("source", "target", preferred_protocol=None)
        assert selected == "langchain"  # Common protocol

    def test_capability_matrix_scoring_is_binary(self):
        """Test that capability scores are binary (1.0 or 0.0)."""
        manager = ProtocolNegotiationManager()

        caps = AgentCapabilities(
            agent_id="test",
            supported_protocols=["proto1", "proto2"]
        )
        manager.register_capabilities(caps)

        matrix = manager._capability_matrix["test"]
        for score in matrix.values():
            assert score in [0.0, 1.0, 1], "Score should be 1.0 or 0.0"

    def test_select_protocol_single_protocol_agent(self):
        """Test protocol selection with single-protocol agents."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        selected = manager.select_protocol("source", "target")
        assert selected == "langchain"

    def test_register_agent_twice_overwrites(self):
        """Test that registering an agent twice overwrites the previous registration."""
        manager = ProtocolNegotiationManager()

        caps1 = AgentCapabilities(
            agent_id="agent",
            supported_protocols=["proto1"]
        )
        caps2 = AgentCapabilities(
            agent_id="agent",
            supported_protocols=["proto2", "proto3"]
        )

        manager.register_capabilities(caps1)
        manager.register_capabilities(caps2)

        # Should have the second registration
        assert manager._capability_matrix["agent"] == {"proto2": 1.0, "proto3": 1.0}

    def test_empty_protocol_list_edge_case(self):
        """Test handling of agents with empty protocol lists."""
        manager = ProtocolNegotiationManager()

        caps = AgentCapabilities(
            agent_id="empty_agent",
            supported_protocols=[]
        )
        manager.register_capabilities(caps)

        # Should handle gracefully
        assert manager._capability_matrix["empty_agent"] == {}

        # Selecting protocol with empty target should fail or use fallback gracefully
        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["proto1"]
        )
        manager.register_capabilities(source_caps)

        with pytest.raises((ValueError, IndexError)):
            # This should fail because target has no protocols to fallback to
            manager.select_protocol("source", "empty_agent")


class TestLatencyBenchmark:
    """Latency benchmarking tests for protocol selection."""

    def test_protocol_selection_latency(self):
        """Benchmark protocol selection latency (< 50ms)."""
        manager = ProtocolNegotiationManager()

        # Setup: Register multiple agents with multiple protocols
        for i in range(10):
            caps = AgentCapabilities(
                agent_id=f"agent_{i}",
                supported_protocols=[f"proto_{j}" for j in range(5)]
            )
            manager.register_capabilities(caps)

        # Measure latency for protocol selection
        start = time.perf_counter()
        for _ in range(100):
            manager.select_protocol("agent_0", "agent_5", preferred_protocol="proto_2")
        elapsed = time.perf_counter() - start

        # Calculate average time per selection
        avg_time_ms = (elapsed / 100) * 1000

        # Assert < 50ms per selection
        assert avg_time_ms < 50, f"Average selection latency {avg_time_ms:.2f}ms exceeds 50ms"

    def test_protocol_selection_with_fallback_latency(self):
        """Benchmark protocol selection latency when fallback is used (< 50ms)."""
        manager = ProtocolNegotiationManager()

        # Setup agents with no common protocols
        source_caps = AgentCapabilities(
            agent_id="source",
            supported_protocols=["proto_1", "proto_2", "proto_3"]
        )
        target_caps = AgentCapabilities(
            agent_id="target",
            supported_protocols=["proto_4", "proto_5", "proto_6"]
        )

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        # Measure latency for fallback selection
        start = time.perf_counter()
        for _ in range(100):
            manager.select_protocol("source", "target")
        elapsed = time.perf_counter() - start

        # Calculate average time per selection
        avg_time_ms = (elapsed / 100) * 1000

        # Assert < 50ms per selection
        assert avg_time_ms < 50, f"Average fallback selection latency {avg_time_ms:.2f}ms exceeds 50ms"

    def test_multiple_registration_and_selection_cycle_latency(self):
        """Benchmark a full cycle of registrations and selections (< 50ms)."""
        # Measure latency for 10 cycles of registration and selection
        times = []
        for _ in range(10):
            manager = ProtocolNegotiationManager()

            # Register 10 agents
            for i in range(10):
                caps = AgentCapabilities(
                    agent_id=f"agent_{i}",
                    supported_protocols=[f"proto_{j}" for j in range(5)]
                )
                manager.register_capabilities(caps)

            # Perform selections
            start = time.perf_counter()
            for i in range(10):
                manager.select_protocol(
                    f"agent_{i}",
                    f"agent_{(i + 1) % 10}",
                    preferred_protocol="proto_2"
                )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Calculate average time per cycle
        avg_cycle_time_ms = (sum(times) / len(times)) * 1000
        avg_selection_ms = avg_cycle_time_ms / 10

        # Assert < 50ms per selection
        assert avg_selection_ms < 50, f"Average selection latency {avg_selection_ms:.2f}ms exceeds 50ms"


class TestLoggingOutput:
    """Tests for logging behavior."""

    def test_registration_logs_info(self):
        """Test that agent registration logs info message."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="test_agent",
            supported_protocols=["proto1", "proto2"]
        )

        with patch('openclaw_gateway.negotiation.logger') as mock_logger:
            manager.register_capabilities(caps)
            mock_logger.info.assert_called()

    def test_preferred_protocol_selection_logs_info(self):
        """Test that preferred protocol selection logs info message."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(agent_id="source", supported_protocols=["proto1"])
        target_caps = AgentCapabilities(agent_id="target", supported_protocols=["proto1"])

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        with patch('openclaw_gateway.negotiation.logger') as mock_logger:
            manager.select_protocol("source", "target", preferred_protocol="proto1")
            mock_logger.info.assert_called()

    def test_common_protocol_selection_logs_info(self):
        """Test that common protocol selection logs info message."""
        manager = ProtocolNegotiationManager()

        source_caps = AgentCapabilities(agent_id="source", supported_protocols=["proto1"])
        target_caps = AgentCapabilities(agent_id="target", supported_protocols=["proto1"])

        manager.register_capabilities(source_caps)
        manager.register_capabilities(target_caps)

        with patch('openclaw_gateway.negotiation.logger') as mock_logger:
            manager.select_protocol("source", "target")
            mock_logger.info.assert_called()
