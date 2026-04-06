"""Unit tests for protocol negotiation manager.

Tests cover:
- Capability matrix updates after agent registration
- Protocol selection with preferred protocols
- Common protocol selection from multiple options
- Fallback to first advertised protocol when no common protocol exists
- Latency benchmarking (< 50ms per selection)
- Multiple protocols per agent
- Error handling for unregistered agents
- Edge cases (single protocol, multiple agents, various scenarios)
"""

import time
import pytest
from openclaw_gateway.negotiation import (
    AgentCapabilities,
    ProtocolNegotiationManager,
)


class TestAgentCapabilitiesDataclass:
    """Test AgentCapabilities dataclass creation and properties."""

    def test_create_agent_capabilities_single_protocol(self):
        """Create AgentCapabilities with single protocol."""
        caps = AgentCapabilities(
            agent_id="agent_a",
            supported_protocols=["langchain"],
        )
        assert caps.agent_id == "agent_a"
        assert caps.supported_protocols == ["langchain"]

    def test_create_agent_capabilities_multiple_protocols(self):
        """Create AgentCapabilities with multiple protocols."""
        caps = AgentCapabilities(
            agent_id="agent_b",
            supported_protocols=["langchain", "event_stream", "autogpt"],
        )
        assert caps.agent_id == "agent_b"
        assert len(caps.supported_protocols) == 3
        assert "langchain" in caps.supported_protocols
        assert "event_stream" in caps.supported_protocols
        assert "autogpt" in caps.supported_protocols


class TestProtocolNegotiationManagerInit:
    """Test ProtocolNegotiationManager initialization."""

    def test_manager_initialization(self):
        """Manager initializes with empty capability structures."""
        manager = ProtocolNegotiationManager()
        # Empty capability matrix and store
        assert manager._capability_matrix == {}
        assert manager._capabilities == {}


class TestRegisterCapabilities:
    """Test capability registration and matrix updates."""

    def test_register_agent_updates_matrix(self):
        """Registering agent updates capability matrix with 1.0 scores."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_a",
            supported_protocols=["langchain", "event_stream"],
        )
        manager.register_capabilities(caps)

        # Check matrix has been updated
        assert "agent_a" in manager._capability_matrix
        assert manager._capability_matrix["agent_a"]["langchain"] == 1.0
        assert manager._capability_matrix["agent_a"]["event_stream"] == 1.0

    def test_register_multiple_agents(self):
        """Registering multiple agents creates separate matrix entries."""
        manager = ProtocolNegotiationManager()
        caps_a = AgentCapabilities(
            agent_id="agent_a", supported_protocols=["langchain"]
        )
        caps_b = AgentCapabilities(
            agent_id="agent_b", supported_protocols=["event_stream"]
        )
        manager.register_capabilities(caps_a)
        manager.register_capabilities(caps_b)

        assert "agent_a" in manager._capability_matrix
        assert "agent_b" in manager._capability_matrix
        assert manager._capability_matrix["agent_a"]["langchain"] == 1.0
        assert manager._capability_matrix["agent_b"]["event_stream"] == 1.0

    def test_register_agent_with_many_protocols(self):
        """Agent advertising multiple protocols has all in matrix."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_multi",
            supported_protocols=["langchain", "event_stream", "autogpt", "grpc"],
        )
        manager.register_capabilities(caps)

        # All 4 protocols should be in matrix with score 1.0
        assert len(manager._capability_matrix["agent_multi"]) == 4
        for protocol in ["langchain", "event_stream", "autogpt", "grpc"]:
            assert manager._capability_matrix["agent_multi"][protocol] == 1.0

    def test_register_overwrites_previous_capabilities(self):
        """Re-registering an agent overwrites its capabilities."""
        manager = ProtocolNegotiationManager()
        caps1 = AgentCapabilities(
            agent_id="agent_a", supported_protocols=["langchain"]
        )
        caps2 = AgentCapabilities(
            agent_id="agent_a",
            supported_protocols=["event_stream", "autogpt"],
        )
        manager.register_capabilities(caps1)
        manager.register_capabilities(caps2)

        # Should have latest capabilities only
        matrix = manager._capability_matrix["agent_a"]
        assert "langchain" not in matrix  # Old protocol removed
        assert matrix["event_stream"] == 1.0
        assert matrix["autogpt"] == 1.0


class TestGetCompatibilityScore:
    """Test compatibility score retrieval."""

    def test_get_compatibility_score_supported_protocol(self):
        """Getting score for supported protocol returns 1.0."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_a", supported_protocols=["langchain"]
        )
        manager.register_capabilities(caps)

        score = manager.get_compatibility_score("agent_a", "langchain")
        assert score == 1.0

    def test_get_compatibility_score_unsupported_protocol(self):
        """Getting score for unsupported protocol returns 0.0."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_a", supported_protocols=["langchain"]
        )
        manager.register_capabilities(caps)

        score = manager.get_compatibility_score("agent_a", "event_stream")
        assert score == 0.0

    def test_get_compatibility_score_unregistered_agent(self):
        """Getting score for unregistered agent returns 0.0."""
        manager = ProtocolNegotiationManager()

        score = manager.get_compatibility_score("unknown_agent", "langchain")
        assert score == 0.0


class TestSelectProtocolPreferred:
    """Test protocol selection with preferred protocol."""

    def test_select_protocol_preferred_available(self):
        """Preferred protocol is selected if target supports it."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["langchain", "event_stream"]
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        # Prefer langchain
        selected = manager.select_protocol("source", "target", preferred_protocol="langchain")
        assert selected == "langchain"

    def test_select_protocol_preferred_unavailable(self):
        """If preferred unavailable, fallback to common or target's first."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["langchain", "event_stream"]
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        # Prefer langchain, but target doesn't support it
        # Should fallback to common protocol event_stream
        selected = manager.select_protocol("source", "target", preferred_protocol="langchain")
        assert selected == "event_stream"

    def test_select_protocol_preferred_not_in_target(self):
        """Preferred protocol not supported by target is ignored."""
        manager = ProtocolNegotiationManager()
        caps_target = AgentCapabilities(
            agent_id="target", supported_protocols=["event_stream"]
        )
        manager.register_capabilities(caps_target)

        # Prefer langchain, target supports event_stream
        selected = manager.select_protocol("source", "target", preferred_protocol="langchain")
        assert selected == "event_stream"


class TestSelectProtocolCommon:
    """Test protocol selection with common protocols."""

    def test_select_protocol_common_single(self):
        """With single common protocol, it is selected."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["langchain", "autogpt"]
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        selected = manager.select_protocol("source", "target")
        assert selected == "langchain"

    def test_select_protocol_common_multiple(self):
        """With multiple common protocols, highest scored is selected."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain", "event_stream", "autogpt"],
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        # Both common, MVP has equal scores (1.0), but max() will pick one
        selected = manager.select_protocol("source", "target")
        assert selected in ["langchain", "event_stream"]


class TestSelectProtocolFallback:
    """Test protocol selection fallback behavior."""

    def test_select_protocol_fallback_no_common(self):
        """Falls back to target's first protocol when no common protocol."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["langchain"]
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream", "autogpt"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        selected = manager.select_protocol("source", "target")
        assert selected == "event_stream"  # target's first protocol

    def test_select_protocol_fallback_single_protocol(self):
        """Edge case: target advertises only one protocol, fallback uses it."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["langchain"]
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        selected = manager.select_protocol("source", "target")
        assert selected == "event_stream"

    def test_select_protocol_source_not_registered(self):
        """Source not registered: fallback still works."""
        manager = ProtocolNegotiationManager()
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["event_stream", "autogpt"],
        )
        manager.register_capabilities(caps_target)

        # Source not registered, should fallback to target's first
        selected = manager.select_protocol("unregistered_source", "target")
        assert selected == "event_stream"


class TestSelectProtocolErrors:
    """Test error handling in protocol selection."""

    def test_select_protocol_unregistered_target_raises_error(self):
        """ValueError raised if target not registered."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["langchain"]
        )
        manager.register_capabilities(caps_source)

        with pytest.raises(ValueError) as exc_info:
            manager.select_protocol("source", "unregistered_target")

        assert "No capabilities registered" in str(exc_info.value)
        assert "unregistered_target" in str(exc_info.value)


class TestLatencyBenchmark:
    """Test negotiation latency requirements."""

    def test_latency_benchmark_single_selection(self):
        """Single protocol selection completes in < 50ms."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source",
            supported_protocols=["langchain", "event_stream"],
        )
        caps_target = AgentCapabilities(
            agent_id="target",
            supported_protocols=["langchain", "event_stream"],
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        start = time.perf_counter()
        manager.select_protocol("source", "target", preferred_protocol="langchain")
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 50, f"Selection took {elapsed:.2f}ms, expected < 50ms"

    def test_latency_benchmark_100_selections(self):
        """100 protocol selections complete in average < 50ms each."""
        manager = ProtocolNegotiationManager()

        # Register 50 agents with 5 protocols each
        for i in range(50):
            caps = AgentCapabilities(
                agent_id=f"agent_{i}",
                supported_protocols=[
                    f"protocol_{j}" for j in range(5)
                ],
            )
            manager.register_capabilities(caps)

        # Time 100 selections
        start = time.perf_counter()
        for i in range(100):
            source = f"agent_{i % 50}"
            target = f"agent_{(i + 1) % 50}"
            manager.select_protocol(
                source, target, preferred_protocol=f"protocol_0"
            )
        total_elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        avg_elapsed = total_elapsed / 100

        assert avg_elapsed < 50, (
            f"Average selection took {avg_elapsed:.2f}ms "
            f"(total: {total_elapsed:.2f}ms for 100 selections), "
            f"expected < 50ms per selection"
        )

    def test_latency_large_capability_matrix(self):
        """Protocol selection is fast even with large capability matrix."""
        manager = ProtocolNegotiationManager()

        # Create matrix with 100 agents, 10 protocols each
        for i in range(100):
            protocols = [f"protocol_{j}" for j in range(10)]
            caps = AgentCapabilities(
                agent_id=f"agent_{i}",
                supported_protocols=protocols,
            )
            manager.register_capabilities(caps)

        # Select from one of the agents
        start = time.perf_counter()
        selected = manager.select_protocol("agent_0", "agent_50")
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        assert elapsed < 50, (
            f"Selection with 100×10 matrix took {elapsed:.2f}ms, "
            f"expected < 50ms"
        )


@pytest.mark.parametrize(
    "source_protocols,target_protocols,expected_selection",
    [
        # Single common protocol
        (["langchain"], ["langchain"], "langchain"),
        # Multiple common, all equal score
        (
            ["langchain", "event_stream"],
            ["langchain", "event_stream"],
            ("langchain", "event_stream"),
        ),
        # No common protocols
        (["langchain"], ["event_stream"], "event_stream"),
        # Multiple targets
        (
            ["langchain", "event_stream"],
            ["event_stream", "autogpt"],
            "event_stream",
        ),
    ],
)
def test_select_protocol_parametrized(
    source_protocols, target_protocols, expected_selection
):
    """Parametrized tests for protocol selection across various scenarios."""
    manager = ProtocolNegotiationManager()
    caps_source = AgentCapabilities(
        agent_id="source", supported_protocols=source_protocols
    )
    caps_target = AgentCapabilities(
        agent_id="target", supported_protocols=target_protocols
    )
    manager.register_capabilities(caps_source)
    manager.register_capabilities(caps_target)

    selected = manager.select_protocol("source", "target")

    if isinstance(expected_selection, tuple):
        assert selected in expected_selection
    else:
        assert selected == expected_selection


@pytest.mark.parametrize(
    "protocols,preferred,expected",
    [
        (["langchain", "event_stream"], "langchain", "langchain"),
        (["langchain", "event_stream"], "event_stream", "event_stream"),
        (["event_stream"], "langchain", "event_stream"),
    ],
)
def test_select_protocol_with_preferred_parametrized(
    protocols, preferred, expected
):
    """Parametrized tests for protocol selection with preferred protocol."""
    manager = ProtocolNegotiationManager()
    caps = AgentCapabilities(agent_id="target", supported_protocols=protocols)
    manager.register_capabilities(caps)

    selected = manager.select_protocol(
        "source", "target", preferred_protocol=preferred
    )
    assert selected == expected


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_protocol_list_registration(self):
        """Agent with no protocols can be registered (edge case)."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(agent_id="agent_empty", supported_protocols=[])
        manager.register_capabilities(caps)

        assert "agent_empty" in manager._capability_matrix
        assert manager._capability_matrix["agent_empty"] == {}

    def test_protocol_selection_with_empty_target_protocols(self):
        """Selecting protocol with target having no protocols raises error."""
        manager = ProtocolNegotiationManager()
        caps_target = AgentCapabilities(
            agent_id="target", supported_protocols=[]
        )
        manager.register_capabilities(caps_target)

        # This should raise IndexError because target's first protocol doesn't exist
        with pytest.raises(IndexError):
            manager.select_protocol("source", "target")

    def test_same_source_target(self):
        """Selecting protocol with source = target works."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_a", supported_protocols=["langchain"]
        )
        manager.register_capabilities(caps)

        selected = manager.select_protocol("agent_a", "agent_a")
        assert selected == "langchain"

    def test_protocol_names_case_sensitive(self):
        """Protocol names are case-sensitive."""
        manager = ProtocolNegotiationManager()
        caps_source = AgentCapabilities(
            agent_id="source", supported_protocols=["LangChain"]
        )
        caps_target = AgentCapabilities(
            agent_id="target", supported_protocols=["langchain"]
        )
        manager.register_capabilities(caps_source)
        manager.register_capabilities(caps_target)

        # No common protocol (case mismatch)
        selected = manager.select_protocol("source", "target")
        assert selected == "langchain"  # Fallback to target's first

    def test_duplicate_protocols_in_list(self):
        """Duplicate protocols in supported_protocols list."""
        manager = ProtocolNegotiationManager()
        caps = AgentCapabilities(
            agent_id="agent_dup",
            supported_protocols=["langchain", "langchain", "event_stream"],
        )
        manager.register_capabilities(caps)

        # Both langchain occurrences should be in matrix
        assert manager._capability_matrix["agent_dup"]["langchain"] == 1.0
        assert manager._capability_matrix["agent_dup"]["event_stream"] == 1.0
