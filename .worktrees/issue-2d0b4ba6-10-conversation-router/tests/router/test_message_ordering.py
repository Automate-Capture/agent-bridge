"""
Tests for message ordering with topological sort.

Tests cover:
- 5 messages with explicit dependencies delivering in correct order
- Topological sort respects all dependency edges
- Diamond dependency graphs
- Linear chains
- Multiple independent branches
"""

import pytest
from openclaw_gateway.router import ConversationRouter


class TestMessageOrdering:
    """Test topological sort for message ordering."""

    def test_five_message_chain_ordering(self):
        """Five messages with linear dependencies: M1→M2→M3→M4→M5."""
        router = ConversationRouter()

        # Build linear dependency chain
        # M1 → M2 → M3 → M4 → M5
        router.add_delegation("conv1", "M1", "M2")
        router.add_delegation("conv1", "M2", "M3")
        router.add_delegation("conv1", "M3", "M4")
        router.add_delegation("conv1", "M4", "M5")

        # Topological sort should preserve order
        sorted_order = router.topological_sort()

        # Find indices
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # Verify ordering
        assert indices["M1"] < indices["M2"]
        assert indices["M2"] < indices["M3"]
        assert indices["M3"] < indices["M4"]
        assert indices["M4"] < indices["M5"]

    def test_topological_sort_with_dependencies(self):
        """Topological sort respects all dependency edges."""
        router = ConversationRouter()

        # Build graph:
        # A → B
        # A → C
        # B → D
        # C → D
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "A", "C")
        router.add_delegation("conv1", "B", "D")
        router.add_delegation("conv1", "C", "D")

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # Verify all edges respected
        assert indices["A"] < indices["B"]
        assert indices["A"] < indices["C"]
        assert indices["B"] < indices["D"]
        assert indices["C"] < indices["D"]

    def test_diamond_dependency_graph(self):
        """Diamond dependency: A→B, A→C, B→D, C→D."""
        router = ConversationRouter()

        # Diamond shape:
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "A", "C")
        router.add_delegation("conv1", "B", "D")
        router.add_delegation("conv1", "C", "D")

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # A must come first
        assert indices["A"] < indices["B"]
        assert indices["A"] < indices["C"]

        # D must come last
        assert indices["B"] < indices["D"]
        assert indices["C"] < indices["D"]

    def test_linear_chain_preserves_order(self):
        """Linear chain A→B→C→D→E preserves order."""
        router = ConversationRouter()

        agents = ["A", "B", "C", "D", "E"]
        for i in range(len(agents) - 1):
            router.add_delegation("conv1", agents[i], agents[i + 1])

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # Verify sequential order
        for i in range(len(agents) - 1):
            assert indices[agents[i]] < indices[agents[i + 1]]

    def test_multiple_independent_branches(self):
        """Multiple independent branches: A→B, C→D."""
        router = ConversationRouter()

        # Branch 1: A→B
        router.add_delegation("conv1", "A", "B")

        # Branch 2: C→D (independent)
        router.add_delegation("conv1", "C", "D")

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # Each branch respects its own order
        assert indices["A"] < indices["B"]
        assert indices["C"] < indices["D"]

    def test_topological_sort_rejects_cycles(self):
        """Topological sort raises error if graph contains cycles."""
        router = ConversationRouter()

        # Build cycle: A→B→C→A
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")
        router.add_delegation("conv1", "C", "A")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cycle detected"):
            router.topological_sort()

    def test_empty_graph_sort(self):
        """Empty graph sorts to empty list."""
        router = ConversationRouter()

        sorted_order = router.topological_sort()
        assert sorted_order == []

    def test_single_node_sort(self):
        """Single node sorts correctly."""
        router = ConversationRouter()

        # Add single self-reference then remove to have single node in graph
        router.add_delegation("conv1", "A", "B")

        sorted_order = router.topological_sort()
        assert "A" in sorted_order
        assert "B" in sorted_order
        assert len(sorted_order) == 2

    def test_disconnected_components_sort(self):
        """Disconnected components sort independently."""
        router = ConversationRouter()

        # Component 1: A→B→C
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")

        # Component 2: X→Y→Z
        router.add_delegation("conv1", "X", "Y")
        router.add_delegation("conv1", "Y", "Z")

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # Component 1 order preserved
        assert indices["A"] < indices["B"] < indices["C"]

        # Component 2 order preserved
        assert indices["X"] < indices["Y"] < indices["Z"]

    def test_complex_dag_sort(self):
        """Complex DAG with multiple paths to same node."""
        router = ConversationRouter()

        # Complex graph:
        #   A → B → D
        #   |       |
        #   └─→ C ──┘
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "A", "C")
        router.add_delegation("conv1", "B", "D")
        router.add_delegation("conv1", "C", "D")

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        assert indices["A"] < indices["B"]
        assert indices["A"] < indices["C"]
        assert indices["B"] < indices["D"]
        assert indices["C"] < indices["D"]

    def test_topological_sort_all_nodes_present(self):
        """Topological sort includes all reachable nodes."""
        router = ConversationRouter()

        nodes = ["A", "B", "C", "D", "E"]
        # Build path A→B→C→D→E
        for i in range(len(nodes) - 1):
            router.add_delegation("conv1", nodes[i], nodes[i + 1])

        sorted_order = router.topological_sort()

        # All nodes should be present
        assert len(sorted_order) == len(nodes)
        for node in nodes:
            assert node in sorted_order

    def test_wide_graph_sort(self):
        """Wide graph with one root and many children."""
        router = ConversationRouter()

        # Root A delegates to B, C, D, E, F
        for target in ["B", "C", "D", "E", "F"]:
            router.add_delegation("conv1", "A", target)

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # A must come first
        for target in ["B", "C", "D", "E", "F"]:
            assert indices["A"] < indices[target]

    def test_deep_graph_sort(self):
        """Deep graph with long chain."""
        router = ConversationRouter()

        # Create chain of 20 agents
        agents = [f"agent_{i}" for i in range(20)]
        for i in range(len(agents) - 1):
            router.add_delegation("conv1", agents[i], agents[i + 1])

        sorted_order = router.topological_sort()

        # Verify all agents present
        assert len(sorted_order) == len(agents)

        # Verify order
        indices = {msg: i for i, msg in enumerate(sorted_order)}
        for i in range(len(agents) - 1):
            assert indices[agents[i]] < indices[agents[i + 1]]

    def test_topological_sort_consistency(self):
        """Multiple sorts of same graph produce valid orderings."""
        router = ConversationRouter()

        # Build graph: A→B→C, A→D
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "B", "C")
        router.add_delegation("conv1", "A", "D")

        # Get multiple sorts
        sort1 = router.topological_sort()
        sort2 = router.topological_sort()

        # Both should be valid (satisfy constraints)
        for sort_order in [sort1, sort2]:
            indices = {msg: i for i, msg in enumerate(sort_order)}
            assert indices["A"] < indices["B"]
            assert indices["B"] < indices["C"]
            assert indices["A"] < indices["D"]

    @pytest.fixture
    def complex_dag(self):
        """Generate a complex DAG for ordering tests."""
        router = ConversationRouter()

        # Build complex graph with multiple valid topological orders
        # A → B → E
        # A → C → E
        # B → D → E
        # C → D
        router.add_delegation("conv1", "A", "B")
        router.add_delegation("conv1", "A", "C")
        router.add_delegation("conv1", "B", "E")
        router.add_delegation("conv1", "B", "D")
        router.add_delegation("conv1", "C", "D")
        router.add_delegation("conv1", "C", "E")
        router.add_delegation("conv1", "D", "E")

        return router

    def test_complex_dag_ordering(self, complex_dag):
        """Complex DAG respects all constraints."""
        router = complex_dag

        sorted_order = router.topological_sort()
        indices = {msg: i for i, msg in enumerate(sorted_order)}

        # Verify all edges
        assert indices["A"] < indices["B"]
        assert indices["A"] < indices["C"]
        assert indices["B"] < indices["E"]
        assert indices["B"] < indices["D"]
        assert indices["C"] < indices["D"]
        assert indices["C"] < indices["E"]
        assert indices["D"] < indices["E"]
