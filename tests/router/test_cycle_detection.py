"""
Tests for cycle detection using Tarjan's SCC algorithm.

Tests cover:
- 3-node cycle detection: A→B→C→A
- 5-node cycle detection: A→B→C→D→E→A
- Self-loop detection: A→A
- No-cycle scenarios: linear chains
- Disconnected components
- Large graphs (10+ agents)
"""

import pytest
import time
from openclaw_gateway.router import ConversationRouter


class TestTarjanCycleDetection:
    """Test Tarjan's SCC algorithm for cycle detection."""

    def test_three_node_cycle_detected(self):
        """Detect simple 3-node cycle: A→B→C→A."""
        router = ConversationRouter()

        # Build cycle: A→B→C→A
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")

        # Adding C→A should be detected as cycle
        assert router.check_cycle("conv1", "C", "A") is True

    def test_three_node_cycle_rejected(self):
        """Reject circular delegation A→B→C→A."""
        router = ConversationRouter()

        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")

        # Try to add the cycle-closing edge
        has_cycle = router.check_cycle("conv1", "C", "A")
        assert has_cycle is True

    def test_five_node_cycle_detected(self):
        """Detect 5-node cycle: A→B→C→D→E→A."""
        router = ConversationRouter()

        # Build chain: A→B→C→D→E
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")
        router.add_delegation("conv1", "C", "D")
        router.add_delegation("conv1", "D", "E")

        # Adding E→A should create cycle
        assert router.check_cycle("conv1", "E", "A") is True

    def test_self_loop_detected(self):
        """Detect self-loop: A→A."""
        router = ConversationRouter()

        # Self-loop should be detected
        assert router.check_cycle("conv1", "A", "A") is True

    def test_no_cycle_linear_chain(self):
        """Non-cyclic linear chain: A→B→C→D."""
        router = ConversationRouter()

        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")
        router.add_delegation("conv1", "C", "D")

        # Adding D→A would create cycle A→B→C→D→A
        assert router.check_cycle("conv1", "D", "A") is True
        # Adding D→new_agent should not create cycle
        assert router.check_cycle("conv1", "D", "new_agent") is False

    def test_non_cyclic_delegation_response(self):
        """Non-cyclic delegation: A→B→C→A_response succeeds."""
        router = ConversationRouter()

        # Build delegation chain
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")

        # Response from C back to A (different agent) should not create cycle
        # A_response is a different agent, so no cycle
        assert router.check_cycle("conv1", "C", "A_response") is False

    def test_disconnected_components_no_cycle(self):
        """Disconnected components without cycles."""
        router = ConversationRouter()

        # Component 1: A→B
        router.add_delegation("conv1", "A", "B")

        # Component 2: C→D
        router.add_delegation("conv1", "C", "D")

        # No cycle between disconnected components
        assert router.check_cycle("conv1", "B", "C") is False
        assert router.check_cycle("conv1", "D", "A") is False

    def test_disconnected_with_cycle(self):
        """Disconnected component with internal cycle."""
        router = ConversationRouter()

        # Component 1: A→B→C→A (cycle)
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")

        # Component 2: D→E (no cycle)
        router.add_delegation("conv1", "D", "E")

        # Closing cycle in component 1
        assert router.check_cycle("conv1", "C", "A") is True

        # E→D would create E↔D cycle (bidirectional)
        assert router.check_cycle("conv1", "E", "D") is True
        # But E→new_agent should not create cycle
        assert router.check_cycle("conv1", "E", "new_agent") is False

    def test_cycle_detection_under_100ms(self):
        """Cycle detection completes within 100ms."""
        router = ConversationRouter()

        # Build a 10-agent chain
        for i in range(9):
            agent_from = f"agent_{i}"
            agent_to = f"agent_{i+1}"
            router.add_delegation("conv1", agent_from, agent_to)

        # Measure cycle detection time
        start = time.time()
        has_cycle = router.check_cycle("conv1", "agent_9", "agent_0")
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert has_cycle is True
        assert elapsed < 100, f"Cycle detection took {elapsed}ms, expected < 100ms"

    def test_cycle_detection_under_50ms_10_agents(self):
        """Cycle detection completes within 50ms for 10-agent graph."""
        router = ConversationRouter()

        # Build a 10-agent fully connected graph
        agents = [f"agent_{i}" for i in range(10)]
        for i in range(len(agents)):
            for j in range(i + 1, min(i + 3, len(agents))):
                router.add_delegation("conv1", agents[i], agents[j])

        # Measure cycle detection time
        start = time.time()
        has_cycle = router.check_cycle("conv1", agents[-1], agents[0])
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert elapsed < 50, f"Cycle detection took {elapsed}ms, expected < 50ms"

    def test_empty_graph(self):
        """Empty graph has no cycles."""
        router = ConversationRouter()

        assert router.check_cycle("conv1", "A", "B") is False

    def test_single_agent_self_delegation(self):
        """Single agent cannot delegate to itself."""
        router = ConversationRouter()

        assert router.check_cycle("conv1", "agent_1", "agent_1") is True

    def test_tarjan_scc_multiple_sccs(self):
        """Tarjan finds multiple SCCs in graph."""
        router = ConversationRouter()

        # Component 1: A↔B (bidirectional = cycle)
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "A")

        # Component 2: C→D (one-directional, no cycle)
        router.add_delegation("conv1", "C", "D")

        sccs = router._tarjan_scc()

        # Should find at least 2 SCCs: one containing {A,B}, others singletons
        scc_list = [scc for scc in sccs if len(scc) > 1]
        assert len(scc_list) >= 1
        assert {"A", "B"} in sccs

    def test_check_cycle_without_modifying_graph(self):
        """check_cycle doesn't permanently modify graph."""
        router = ConversationRouter()

        router.add_delegation("conv1", "A", "B")

        # Check for cycle (temporarily adds edge)
        router.check_cycle("conv1", "B", "A")

        # Graph should still be in original state
        assert "B" in router._dependency_graph["A"]
        assert "A" not in router._dependency_graph["B"]

    def test_multiple_conversations_independent(self):
        """Cycles in different conversations are tracked in shared graph."""
        router = ConversationRouter()

        # Conv1: A→B→C
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")

        # Conv2: X→Y→Z
        router.add_delegation("conv2", "X", "Y")
        router.add_delegation("conv2", "Y", "Z")

        # Cycle in conv1
        assert router.check_cycle("conv1", "C", "A") is True

        # Z→X would create cycle in the shared graph (X→Y→Z→X)
        assert router.check_cycle("conv2", "Z", "X") is True
        # But C→X wouldn't create cycle (different components)
        assert router.check_cycle("conv1", "C", "X") is False

    @pytest.fixture
    def complex_graph(self):
        """Generate a complex graph with multiple cycles for testing."""
        router = ConversationRouter()

        # Create a graph: A→B→C→A (cycle 1)
        #                B→D→C (cycle 1 path)
        #                E→F→G (no cycle)
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")
        router.add_delegation("conv1", "B", "D")
        router.add_delegation("conv1", "D", "C")
        router.add_delegation("conv1", "E", "F")
        router.add_delegation("conv1", "F", "G")

        return router

    def test_complex_graph_cycle_detection(self, complex_graph):
        """Detect cycles in complex graph with multiple paths."""
        router = complex_graph

        # C→A should detect cycle (A→B→C→A)
        assert router.check_cycle("conv1", "C", "A") is True

        # G→E would create cycle (E→F→G→E), so this should be True
        assert router.check_cycle("conv1", "G", "E") is True
        # But G→new_agent should not create cycle
        assert router.check_cycle("conv1", "G", "new_agent") is False
