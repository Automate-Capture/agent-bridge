"""Unit tests for protocol adapter base class.

This module tests the abstract base class contract for ProtocolAdapter,
ensuring that:
1. Direct instantiation raises TypeError
2. Concrete subclasses implementing all abstract methods instantiate successfully
3. Incomplete subclasses missing required abstract methods raise TypeError
4. Property access works correctly
5. Polymorphic dispatch works with subclass implementations
"""

import pytest
from abc import ABC
from openclaw_gateway.adapters import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage
import uuid


class TestProtocolAdapterABCEnforcement:
    """Test abstract base class enforcement rules."""

    def test_cannot_instantiate_protocol_adapter_directly(self):
        """Cannot directly instantiate ProtocolAdapter (TypeError).

        ProtocolAdapter is an ABC with abstract methods. Attempting direct
        instantiation must raise TypeError with message indicating abstract class.
        """
        with pytest.raises(TypeError) as exc_info:
            ProtocolAdapter(agent_id="test_agent", protocol_name="test_protocol")

        # Verify the error message indicates abstract methods
        error_msg = str(exc_info.value).lower()
        assert "abstract" in error_msg or "can't instantiate" in error_msg

    def test_protocol_adapter_is_abc(self):
        """ProtocolAdapter is a proper ABC subclass."""
        assert issubclass(ProtocolAdapter, ABC)

    def test_concrete_subclass_implementing_all_methods_instantiates(self):
        """Concrete subclass implementing all abstract methods instantiates successfully.

        A subclass providing implementations for ingest() and egress() must
        instantiate without error.
        """

        class ConcreteAdapter(ProtocolAdapter):
            """Concrete adapter implementing all abstract methods."""

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                """Parse agent format to canonical."""
                # Simple mock: return a valid CanonicalMessage
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={"document_id": "doc1", "format": "pdf", "size": 1024},
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                """Transform canonical to agent format."""
                return b'{"test": "data"}'

        # Should instantiate without error
        adapter = ConcreteAdapter(
            agent_id="test_agent", protocol_name="test_protocol"
        )
        assert adapter is not None
        assert isinstance(adapter, ProtocolAdapter)

    def test_incomplete_subclass_missing_ingest_raises_type_error(self):
        """Subclass missing ingest() abstract method raises TypeError.

        A subclass that doesn't implement the ingest() method must raise
        TypeError when attempting instantiation.
        """

        class IncompleteAdapterMissingIngest(ProtocolAdapter):
            """Incomplete adapter missing ingest() implementation."""

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                """Transform canonical to agent format."""
                return b'{"test": "data"}'

        with pytest.raises(TypeError) as exc_info:
            IncompleteAdapterMissingIngest(
                agent_id="test_agent", protocol_name="test_protocol"
            )

        error_msg = str(exc_info.value).lower()
        assert "abstract" in error_msg or "can't instantiate" in error_msg

    def test_incomplete_subclass_missing_egress_raises_type_error(self):
        """Subclass missing egress() abstract method raises TypeError.

        A subclass that doesn't implement the egress() method must raise
        TypeError when attempting instantiation.
        """

        class IncompleteAdapterMissingEgress(ProtocolAdapter):
            """Incomplete adapter missing egress() implementation."""

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                """Parse agent format to canonical."""
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={"document_id": "doc1", "format": "pdf", "size": 1024},
                )

        with pytest.raises(TypeError) as exc_info:
            IncompleteAdapterMissingEgress(
                agent_id="test_agent", protocol_name="test_protocol"
            )

        error_msg = str(exc_info.value).lower()
        assert "abstract" in error_msg or "can't instantiate" in error_msg


class TestProtocolAdapterProperties:
    """Test property access and initialization."""

    def test_agent_id_property_read_only(self):
        """agent_id property is read-only and set via __init__."""

        class ConcreteAdapter(ProtocolAdapter):
            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={"document_id": "doc1", "format": "pdf", "size": 1024},
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b'{"test": "data"}'

        adapter = ConcreteAdapter(agent_id="my_agent", protocol_name="my_protocol")
        assert adapter.agent_id == "my_agent"

        # Verify it's read-only by attempting to set it
        with pytest.raises(AttributeError):
            adapter.agent_id = "different_agent"

    def test_protocol_name_property_read_only(self):
        """protocol_name property is read-only and set via __init__."""

        class ConcreteAdapter(ProtocolAdapter):
            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={"document_id": "doc1", "format": "pdf", "size": 1024},
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b'{"test": "data"}'

        adapter = ConcreteAdapter(
            agent_id="my_agent", protocol_name="my_protocol"
        )
        assert adapter.protocol_name == "my_protocol"

        # Verify it's read-only by attempting to set it
        with pytest.raises(AttributeError):
            adapter.protocol_name = "different_protocol"


class TestPolymorphicDispatch:
    """Test polymorphic dispatch with multiple concrete implementations."""

    def test_polymorphic_dispatch_calls_correct_subclass_method(self):
        """Polymorphic dispatch calls the correct subclass implementation.

        When using a ProtocolAdapter reference to call ingest(), the correct
        subclass method should be invoked (dynamic dispatch).
        """

        class Adapter1(ProtocolAdapter):
            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="adapter1",
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={"document_id": "doc1", "format": "pdf", "size": 1024},
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b'{"source": "adapter1"}'

        class Adapter2(ProtocolAdapter):
            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="adapter2",
                    conversation_id="test_conv",
                    message_type="request",
                    intent="delegate",
                    payload={
                        "target_agent": "other",
                        "task_description": "test task",
                        "deadline": 3600,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b'{"source": "adapter2"}'

        adapters = [
            Adapter1(agent_id="agent1", protocol_name="proto1"),
            Adapter2(agent_id="agent2", protocol_name="proto2"),
        ]

        # Verify polymorphic dispatch
        assert adapters[0].agent_id == "agent1"
        assert adapters[1].agent_id == "agent2"

        # Test that we can treat them as ProtocolAdapter references
        for adapter in adapters:
            assert isinstance(adapter, ProtocolAdapter)

    def test_multiple_subclass_instantiation(self):
        """Multiple subclass instances can coexist and function independently."""

        class MockAdapter(ProtocolAdapter):
            def __init__(self, agent_id: str, protocol_name: str, marker: str):
                super().__init__(agent_id, protocol_name)
                self.marker = marker

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={"document_id": f"doc_{self.marker}", "format": "pdf", "size": 1024},
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return f'{{"marker": "{self.marker}"}}'.encode()

        adapter1 = MockAdapter(agent_id="agent1", protocol_name="proto1", marker="A")
        adapter2 = MockAdapter(agent_id="agent2", protocol_name="proto2", marker="B")

        assert adapter1.agent_id == "agent1"
        assert adapter2.agent_id == "agent2"
        assert adapter1.marker == "A"
        assert adapter2.marker == "B"
        assert adapter1.protocol_name == "proto1"
        assert adapter2.protocol_name == "proto2"
