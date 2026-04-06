"""
Unit tests for ProtocolAdapter abstract base class.

Tests verify that:
1. ProtocolAdapter ABC cannot be instantiated directly
2. Concrete subclass implementing all methods can be instantiated
3. Concrete subclass missing abstract methods raises TypeError on instantiation
4. Abstract properties and methods are properly enforced
"""

import pytest
from abc import ABC
from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage
import uuid


class TestProtocolAdapterABCContract:
    """Test that ProtocolAdapter is a proper ABC with correct enforcement."""

    def test_protocol_adapter_is_abc(self):
        """Verify ProtocolAdapter is a subclass of ABC."""
        assert issubclass(ProtocolAdapter, ABC)

    def test_cannot_instantiate_abstract_class_directly(self):
        """Test that ProtocolAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ProtocolAdapter(agent_id="test_agent", protocol_name="test_protocol")

    def test_concrete_implementation_with_all_methods_succeeds(self):
        """Test that a concrete subclass implementing all abstract methods can be instantiated."""

        class ConcreteAdapter(ProtocolAdapter):
            """Concrete implementation with all required methods."""

            @property
            def agent_id(self) -> str:
                """Return the agent_id."""
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                """Minimal ingest implementation for testing."""
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                """Minimal egress implementation for testing."""
                return b"test_output"

        # Should not raise TypeError
        adapter = ConcreteAdapter(
            agent_id="test_agent", protocol_name="test_protocol"
        )
        assert adapter is not None
        assert adapter.agent_id == "test_agent"
        assert adapter.protocol_name == "test_protocol"

    def test_incomplete_subclass_missing_ingest_raises_error(self):
        """Test that a subclass missing the ingest method raises TypeError."""

        class IncompleteAdapter(ProtocolAdapter):
            """Missing ingest method."""

            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter(agent_id="test", protocol_name="test")

    def test_incomplete_subclass_missing_egress_raises_error(self):
        """Test that a subclass missing the egress method raises TypeError."""

        class IncompleteAdapter(ProtocolAdapter):
            """Missing egress method."""

            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter(agent_id="test", protocol_name="test")

    def test_incomplete_subclass_missing_agent_id_property_raises_error(self):
        """Test that a subclass missing the agent_id property raises TypeError."""

        class IncompleteAdapter(ProtocolAdapter):
            """Missing agent_id property."""

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="test_agent",
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter(agent_id="test", protocol_name="test")


class TestProtocolAdapterProperties:
    """Test that ProtocolAdapter properties work correctly."""

    def test_agent_id_property_accessible(self):
        """Test that agent_id property is accessible."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        adapter = TestAdapter(agent_id="my_agent", protocol_name="my_protocol")
        assert adapter.agent_id == "my_agent"

    def test_protocol_name_property_accessible(self):
        """Test that protocol_name property is accessible."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        adapter = TestAdapter(agent_id="my_agent", protocol_name="my_protocol")
        assert adapter.protocol_name == "my_protocol"

    def test_protocol_name_immutable(self):
        """Test that protocol_name is read-only (immutable)."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        adapter = TestAdapter(agent_id="my_agent", protocol_name="my_protocol")

        # Attempting to set protocol_name should raise AttributeError
        with pytest.raises(AttributeError):
            adapter.protocol_name = "new_protocol"


class TestProtocolAdapterInitialization:
    """Test ProtocolAdapter initialization validation."""

    def test_init_with_empty_agent_id_raises_error(self):
        """Test that initializing with empty agent_id raises ValueError."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="test",
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            TestAdapter(agent_id="", protocol_name="test_protocol")

    def test_init_with_empty_protocol_name_raises_error(self):
        """Test that initializing with empty protocol_name raises ValueError."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id="test",
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        with pytest.raises(ValueError, match="protocol_name cannot be empty"):
            TestAdapter(agent_id="test_agent", protocol_name="")


class TestMethodSignatures:
    """Test that method signatures match the architecture specification."""

    def test_ingest_is_async(self):
        """Test that ingest method is async."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        adapter = TestAdapter(agent_id="test", protocol_name="test")
        import inspect

        # Verify ingest is a coroutine function
        assert inspect.iscoroutinefunction(adapter.ingest)

    def test_egress_is_async(self):
        """Test that egress method is async."""

        class TestAdapter(ProtocolAdapter):
            @property
            def agent_id(self) -> str:
                return self._agent_id

            async def ingest(self, raw_message: bytes) -> CanonicalMessage:
                return CanonicalMessage(
                    message_id=str(uuid.uuid4()),
                    source_agent_id=self.agent_id,
                    conversation_id="test_conv",
                    message_type="request",
                    intent="analyze",
                    payload={
                        "document_id": "doc1",
                        "format": "text",
                        "size": 100,
                    },
                )

            async def egress(self, canonical: CanonicalMessage) -> bytes:
                return b"test"

        adapter = TestAdapter(agent_id="test", protocol_name="test")
        import inspect

        # Verify egress is a coroutine function
        assert inspect.iscoroutinefunction(adapter.egress)
