"""
Tests for fallback routing when primary route fails.

Tests cover:
- Primary route fails, message reroutes via fallback
- Fallback routing traversal order
- Multiple fallbacks in sequence
- Fallback exhaustion
- No protocol renegotiation on fallback
- Route configuration and retrieval
"""

import pytest
from openclaw_gateway.router import ConversationRouter, RouteConfig, RouteResult


class TestFallbackRouting:
    """Test fallback routing functionality."""

    def test_route_registration_and_retrieval(self):
        """Register route and retrieve it."""
        router = ConversationRouter()

        router.register_route(
            conversation_id="test_conv",
            primary_agent="primary_agent",
            fallback_agents=["fallback_1", "fallback_2"],
            protocol="langchain",
            timeout_ms=30000
        )

        # Retrieve and verify
        route = router.get_route("test_conv", "analyze")

        assert route.primary == "primary_agent"
        assert route.fallbacks == ["fallback_1", "fallback_2"]
        assert route.protocol == "langchain"
        assert route.timeout_ms == 30000
        assert route.conversation_id == "test_conv"

    def test_route_not_found_raises_error(self):
        """Get route for unregistered conversation raises ValueError."""
        router = ConversationRouter()

        with pytest.raises(ValueError, match="No route configured"):
            router.get_route("nonexistent_conv", "analyze")

    def test_next_fallback_from_primary(self):
        """Get next fallback after primary agent fails."""
        route = RouteResult(
            primary="primary",
            fallbacks=["fallback_1", "fallback_2", "fallback_3"],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="conv1"
        )

        # Primary failed, get first fallback
        next_agent = route.next_fallback("primary")
        assert next_agent == "fallback_1"

    def test_next_fallback_sequence(self):
        """Traverse fallbacks in sequence."""
        route = RouteResult(
            primary="A",
            fallbacks=["B", "C", "D"],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="conv1"
        )

        # A fails → B
        assert route.next_fallback("A") == "B"

        # B fails → C
        assert route.next_fallback("B") == "C"

        # C fails → D
        assert route.next_fallback("C") == "D"

        # D fails → None (exhausted)
        assert route.next_fallback("D") is None

    def test_next_fallback_exhaustion(self):
        """Fallback exhaustion returns None."""
        route = RouteResult(
            primary="primary",
            fallbacks=["fallback_1"],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="conv1"
        )

        # Primary failed → fallback_1
        assert route.next_fallback("primary") == "fallback_1"

        # fallback_1 failed → None (no more fallbacks)
        assert route.next_fallback("fallback_1") is None

    def test_next_fallback_no_fallbacks(self):
        """No fallbacks available returns None."""
        route = RouteResult(
            primary="primary",
            fallbacks=[],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="conv1"
        )

        # Primary failed but no fallbacks
        assert route.next_fallback("primary") is None

    def test_next_fallback_unknown_agent(self):
        """Unknown failed agent returns None."""
        route = RouteResult(
            primary="primary",
            fallbacks=["fallback_1", "fallback_2"],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="conv1"
        )

        # Unknown agent fails
        assert route.next_fallback("unknown_agent") is None

    def test_fallback_protocol_preserved(self):
        """Fallback routing preserves protocol."""
        router = ConversationRouter()

        router.register_route(
            conversation_id="conv1",
            primary_agent="primary",
            fallback_agents=["fallback_1", "fallback_2"],
            protocol="event_stream",
            timeout_ms=30000
        )

        route = router.get_route("conv1", "analyze")

        # Protocol should remain consistent for all fallbacks
        assert route.protocol == "event_stream"

        # Even when routing to fallback, protocol is same
        next_fb = route.next_fallback("primary")
        assert next_fb == "fallback_1"
        # Protocol doesn't change (implied by same RouteResult)
        assert route.protocol == "event_stream"

    def test_fallback_timeout_preserved(self):
        """Fallback routing preserves timeout setting."""
        router = ConversationRouter()

        timeout = 45000
        router.register_route(
            conversation_id="conv1",
            primary_agent="primary",
            fallback_agents=["fallback_1"],
            protocol="langchain",
            timeout_ms=timeout
        )

        route = router.get_route("conv1", "analyze")

        # Timeout should be same for fallbacks
        assert route.timeout_ms == timeout

    def test_primary_route_fails_fallback_succeeds(self):
        """Primary route fails, message reroutes via fallback without protocol renegotiation."""
        router = ConversationRouter()

        # Register route with fallback
        router.register_route(
            conversation_id="test_conv",
            primary_agent="primary_agent",
            fallback_agents=["fallback_agent"],
            protocol="langchain",
            timeout_ms=30000
        )

        # Get initial route
        route = router.get_route("test_conv", "analyze")
        assert route.primary == "primary_agent"
        assert route.protocol == "langchain"

        # Primary fails, get fallback
        fallback = route.next_fallback(route.primary)
        assert fallback == "fallback_agent"

        # Protocol remains same (no renegotiation)
        assert route.protocol == "langchain"

    def test_multiple_conversations_independent_routes(self):
        """Multiple conversations have independent routes."""
        router = ConversationRouter()

        # Register routes for different conversations
        router.register_route(
            conversation_id="conv1",
            primary_agent="agent_A",
            fallback_agents=["agent_B"],
            protocol="langchain",
            timeout_ms=30000
        )

        router.register_route(
            conversation_id="conv2",
            primary_agent="agent_X",
            fallback_agents=["agent_Y", "agent_Z"],
            protocol="event_stream",
            timeout_ms=45000
        )

        # Get routes
        route1 = router.get_route("conv1", "analyze")
        route2 = router.get_route("conv2", "analyze")

        # Verify independence
        assert route1.primary == "agent_A"
        assert route2.primary == "agent_X"

        assert route1.protocol == "langchain"
        assert route2.protocol == "event_stream"

        assert route1.timeout_ms == 30000
        assert route2.timeout_ms == 45000

    def test_fallback_ordering_matters(self):
        """Fallback agents tried in registered order."""
        router = ConversationRouter()

        # Register with specific fallback order
        router.register_route(
            conversation_id="conv1",
            primary_agent="primary",
            fallback_agents=["first_choice", "second_choice", "last_choice"],
            protocol="langchain",
            timeout_ms=30000
        )

        route = router.get_route("conv1", "analyze")

        # Verify order
        assert route.next_fallback("primary") == "first_choice"
        assert route.next_fallback("first_choice") == "second_choice"
        assert route.next_fallback("second_choice") == "last_choice"
        assert route.next_fallback("last_choice") is None

    def test_route_config_all_agents_method(self):
        """RouteConfig.all_agents() returns correct list."""
        config = RouteConfig(
            primary_agent="primary",
            fallback_agents=["fallback_1", "fallback_2"],
            protocol="langchain",
            timeout_ms=30000
        )

        all_agents = config.all_agents()

        assert all_agents == ["primary", "fallback_1", "fallback_2"]

    def test_route_config_empty_fallbacks(self):
        """RouteConfig with empty fallbacks."""
        config = RouteConfig(
            primary_agent="primary",
            fallback_agents=[],
            protocol="langchain",
            timeout_ms=30000
        )

        assert config.all_agents() == ["primary"]

    def test_route_result_conversation_id_tracking(self):
        """RouteResult tracks conversation ID for routing."""
        route = RouteResult(
            primary="agent_1",
            fallbacks=["agent_2"],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="test_conv_123"
        )

        assert route.conversation_id == "test_conv_123"

    def test_fallback_agent_availability_tracking(self):
        """Track which agents are available in fallback sequence."""
        route = RouteResult(
            primary="primary",
            fallbacks=["fallback_1", "fallback_2", "fallback_3"],
            protocol="langchain",
            timeout_ms=30000,
            conversation_id="conv1"
        )

        # Simulate trying primary, then fallbacks
        available = [route.primary] + route.fallbacks
        assert len(available) == 4

        # Try agents in order
        current = route.primary
        attempts = [current]

        while True:
            next_agent = route.next_fallback(current)
            if next_agent is None:
                break
            attempts.append(next_agent)
            current = next_agent

        # Should try all 4 agents
        assert len(attempts) == 4
        assert set(attempts) == set(available)

    def test_default_protocol_langchain(self):
        """Default protocol is langchain."""
        router = ConversationRouter()

        router.register_route(
            conversation_id="conv1",
            primary_agent="agent_1",
            fallback_agents=["agent_2"]
            # protocol not specified, should default
        )

        route = router.get_route("conv1", "analyze")
        assert route.protocol == "langchain"

    def test_default_timeout_30000ms(self):
        """Default timeout is 30000ms."""
        router = ConversationRouter()

        router.register_route(
            conversation_id="conv1",
            primary_agent="agent_1",
            fallback_agents=["agent_2"]
            # timeout_ms not specified, should default
        )

        route = router.get_route("conv1", "analyze")
        assert route.timeout_ms == 30000

    def test_custom_timeout_settings(self):
        """Custom timeout settings are preserved."""
        router = ConversationRouter()

        custom_timeout = 60000
        router.register_route(
            conversation_id="conv1",
            primary_agent="agent_1",
            fallback_agents=["agent_2"],
            timeout_ms=custom_timeout
        )

        route = router.get_route("conv1", "analyze")
        assert route.timeout_ms == custom_timeout

    @pytest.fixture
    def multi_tier_fallback(self):
        """Generate route with multiple fallback tiers."""
        router = ConversationRouter()

        router.register_route(
            conversation_id="conv_multi",
            primary_agent="primary_tier_1",
            fallback_agents=[
                "secondary_tier_2_opt_1",
                "secondary_tier_2_opt_2",
                "tertiary_tier_3_opt_1"
            ],
            protocol="langchain",
            timeout_ms=30000
        )

        return router

    def test_multi_tier_fallback_traversal(self, multi_tier_fallback):
        """Traverse multi-tier fallback hierarchy."""
        router = multi_tier_fallback

        route = router.get_route("conv_multi", "analyze")

        # Traverse all tiers
        tier1 = route.primary
        assert tier1 == "primary_tier_1"

        tier2_opt1 = route.next_fallback(tier1)
        assert tier2_opt1 == "secondary_tier_2_opt_1"

        tier2_opt2 = route.next_fallback(tier2_opt1)
        assert tier2_opt2 == "secondary_tier_2_opt_2"

        tier3 = route.next_fallback(tier2_opt2)
        assert tier3 == "tertiary_tier_3_opt_1"

        # Should be exhausted
        assert route.next_fallback(tier3) is None
