"""Tests for openclaw-gateway error exception hierarchy.

Tests verify:
1. Each error type instantiation and string representation
2. Error inheritance chain
3. Exception handling and catching by parent class
"""

import pytest

from openclaw_gateway.errors import (
    AdapterError,
    CycleDetectedError,
    DuplicateMessageError,
    OpenclawError,
    RoutingError,
    TimeoutError,
    ValidationError,
)


class TestErrorInstantiation:
    """Test that each error type can be instantiated with a message."""

    def test_openclaw_error_instantiation(self):
        """Test OpenclawError can be instantiated with a message."""
        error = OpenclawError("Test message")
        assert str(error) == "Test message"

    def test_adapter_error_instantiation(self):
        """Test AdapterError can be instantiated with a message."""
        error = AdapterError("Adapter failed")
        assert str(error) == "Adapter failed"

    def test_routing_error_instantiation(self):
        """Test RoutingError can be instantiated with a message."""
        error = RoutingError("Routing failed")
        assert str(error) == "Routing failed"

    def test_cycle_detected_error_instantiation(self):
        """Test CycleDetectedError can be instantiated with a message."""
        error = CycleDetectedError("Cycle detected in A->B->A")
        assert str(error) == "Cycle detected in A->B->A"

    def test_timeout_error_instantiation(self):
        """Test TimeoutError can be instantiated with a message."""
        error = TimeoutError("Request timed out after 30s")
        assert str(error) == "Request timed out after 30s"

    def test_validation_error_instantiation(self):
        """Test ValidationError can be instantiated with a message."""
        error = ValidationError("Validation failed: missing field X")
        assert str(error) == "Validation failed: missing field X"

    def test_duplicate_message_error_instantiation(self):
        """Test DuplicateMessageError can be instantiated with a message."""
        error = DuplicateMessageError("Message ID already processed")
        assert str(error) == "Message ID already processed"


class TestErrorInheritance:
    """Test the error inheritance chain."""

    def test_openclaw_error_is_exception(self):
        """Test that OpenclawError inherits from Exception."""
        assert issubclass(OpenclawError, Exception)

    def test_adapter_error_inherits_from_openclaw_error(self):
        """Test that AdapterError inherits from OpenclawError."""
        assert issubclass(AdapterError, OpenclawError)

    def test_routing_error_inherits_from_openclaw_error(self):
        """Test that RoutingError inherits from OpenclawError."""
        assert issubclass(RoutingError, OpenclawError)

    def test_cycle_detected_error_inherits_from_routing_error(self):
        """Test that CycleDetectedError inherits from RoutingError."""
        assert issubclass(CycleDetectedError, RoutingError)

    def test_cycle_detected_error_inherits_from_openclaw_error(self):
        """Test that CycleDetectedError transitively inherits from OpenclawError."""
        assert issubclass(CycleDetectedError, OpenclawError)

    def test_timeout_error_inherits_from_openclaw_error(self):
        """Test that TimeoutError inherits from OpenclawError."""
        assert issubclass(TimeoutError, OpenclawError)

    def test_validation_error_inherits_from_openclaw_error(self):
        """Test that ValidationError inherits from OpenclawError."""
        assert issubclass(ValidationError, OpenclawError)

    def test_duplicate_message_error_inherits_from_openclaw_error(self):
        """Test that DuplicateMessageError inherits from OpenclawError."""
        assert issubclass(DuplicateMessageError, OpenclawError)


class TestErrorExceptionHandling:
    """Test exception handling and catching by parent class."""

    def test_catch_adapter_error_by_type(self):
        """Test catching AdapterError by its own type."""
        with pytest.raises(AdapterError):
            raise AdapterError("Test")

    def test_catch_adapter_error_by_openclaw_error(self):
        """Test catching AdapterError by OpenclawError parent."""
        with pytest.raises(OpenclawError):
            raise AdapterError("Test")

    def test_catch_routing_error_by_type(self):
        """Test catching RoutingError by its own type."""
        with pytest.raises(RoutingError):
            raise RoutingError("Test")

    def test_catch_routing_error_by_openclaw_error(self):
        """Test catching RoutingError by OpenclawError parent."""
        with pytest.raises(OpenclawError):
            raise RoutingError("Test")

    def test_catch_cycle_detected_error_by_type(self):
        """Test catching CycleDetectedError by its own type."""
        with pytest.raises(CycleDetectedError):
            raise CycleDetectedError("Test")

    def test_catch_cycle_detected_error_by_routing_error(self):
        """Test catching CycleDetectedError by RoutingError parent."""
        with pytest.raises(RoutingError):
            raise CycleDetectedError("Test")

    def test_catch_cycle_detected_error_by_openclaw_error(self):
        """Test catching CycleDetectedError by OpenclawError parent."""
        with pytest.raises(OpenclawError):
            raise CycleDetectedError("Test")

    def test_catch_timeout_error_by_type(self):
        """Test catching TimeoutError by its own type."""
        with pytest.raises(TimeoutError):
            raise TimeoutError("Test")

    def test_catch_timeout_error_by_openclaw_error(self):
        """Test catching TimeoutError by OpenclawError parent."""
        with pytest.raises(OpenclawError):
            raise TimeoutError("Test")

    def test_catch_validation_error_by_type(self):
        """Test catching ValidationError by its own type."""
        with pytest.raises(ValidationError):
            raise ValidationError("Test")

    def test_catch_validation_error_by_openclaw_error(self):
        """Test catching ValidationError by OpenclawError parent."""
        with pytest.raises(OpenclawError):
            raise ValidationError("Test")

    def test_catch_duplicate_message_error_by_type(self):
        """Test catching DuplicateMessageError by its own type."""
        with pytest.raises(DuplicateMessageError):
            raise DuplicateMessageError("Test")

    def test_catch_duplicate_message_error_by_openclaw_error(self):
        """Test catching DuplicateMessageError by OpenclawError parent."""
        with pytest.raises(OpenclawError):
            raise DuplicateMessageError("Test")


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize(
    "error_class,expected_message",
    [
        (OpenclawError, "Base error"),
        (AdapterError, "Adapter transformation failed"),
        (RoutingError, "Routing configuration error"),
        (CycleDetectedError, "Circular delegation detected"),
        (TimeoutError, "Request timeout"),
        (ValidationError, "Validation check failed"),
        (DuplicateMessageError, "Duplicate message detected"),
    ],
)
def test_error_string_representation(error_class, expected_message):
    """Parametrized test for error instantiation and string representation."""
    error = error_class(expected_message)
    assert str(error) == expected_message
    assert isinstance(error, OpenclawError)


@pytest.mark.parametrize(
    "error_class,parent_classes",
    [
        (OpenclawError, [Exception]),
        (AdapterError, [OpenclawError, Exception]),
        (RoutingError, [OpenclawError, Exception]),
        (CycleDetectedError, [RoutingError, OpenclawError, Exception]),
        (TimeoutError, [OpenclawError, Exception]),
        (ValidationError, [OpenclawError, Exception]),
        (DuplicateMessageError, [OpenclawError, Exception]),
    ],
)
def test_error_inheritance_chain(error_class, parent_classes):
    """Parametrized test for error inheritance chain verification."""
    for parent_class in parent_classes:
        assert issubclass(error_class, parent_class)
