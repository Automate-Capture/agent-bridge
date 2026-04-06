"""Comprehensive tests for the error exception hierarchy.

Tests cover:
- Instantiation of all error types with messages
- String representation (str(error) returns message)
- Inheritance chain verification
- Catching errors by parent class
- Specific type catching
- Edge cases (empty messages, unicode, error chaining)
"""

import pytest

from openclaw_gateway.errors import (
    OpenclawError,
    AdapterError,
    RoutingError,
    CycleDetectedError,
    TimeoutError,
    ValidationError,
    DuplicateMessageError,
)


# Test data: (error_class, error_name, parent_class)
ERROR_CLASSES = [
    (OpenclawError, "OpenclawError", Exception),
    (AdapterError, "AdapterError", OpenclawError),
    (RoutingError, "RoutingError", OpenclawError),
    (CycleDetectedError, "CycleDetectedError", RoutingError),
    (TimeoutError, "TimeoutError", OpenclawError),
    (ValidationError, "ValidationError", OpenclawError),
    (DuplicateMessageError, "DuplicateMessageError", OpenclawError),
]


class TestErrorInstantiation:
    """Test instantiation of all error classes."""

    @pytest.mark.parametrize("error_class,name,parent", ERROR_CLASSES)
    def test_error_instantiation_with_message(self, error_class, name, parent):
        """Verify each error can be instantiated with a message."""
        message = f"Test {name} error"
        error = error_class(message)
        assert isinstance(error, error_class)
        assert str(error) == message

    @pytest.mark.parametrize("error_class,name,parent", ERROR_CLASSES)
    def test_error_string_representation(self, error_class, name, parent):
        """Verify error message is retrievable via str()."""
        message = "Custom error message"
        error = error_class(message)
        assert str(error) == message

    @pytest.mark.parametrize("error_class,name,parent", ERROR_CLASSES)
    def test_error_repr(self, error_class, name, parent):
        """Verify error repr contains class name."""
        message = "Test error"
        error = error_class(message)
        assert name in repr(error)

    def test_empty_error_message(self):
        """Verify errors handle empty messages gracefully."""
        error = AdapterError("")
        assert str(error) == ""
        assert isinstance(error, OpenclawError)

    def test_unicode_error_message(self):
        """Verify errors handle unicode characters in messages."""
        message = "Error with unicode: 日本語 🚀 Ñoño"
        error = ValidationError(message)
        assert str(error) == message

    def test_multiline_error_message(self):
        """Verify errors handle multiline messages."""
        message = "Error with\nmultiple\nlines"
        error = DuplicateMessageError(message)
        assert str(error) == message


class TestErrorInheritance:
    """Test inheritance chain and type relationships."""

    @pytest.mark.parametrize("error_class,name,parent", ERROR_CLASSES)
    def test_inheritance_chain(self, error_class, name, parent):
        """Verify each error inherits from expected parent."""
        error = error_class("test")
        assert isinstance(error, parent)

    def test_cycle_detected_error_inherits_from_routing_error(self):
        """Verify CycleDetectedError specifically inherits from RoutingError."""
        error = CycleDetectedError("Cycle detected")
        assert isinstance(error, RoutingError)
        assert isinstance(error, OpenclawError)
        # Verify it's a direct child of RoutingError
        assert CycleDetectedError.__bases__ == (RoutingError,)

    def test_adapter_error_inherits_from_openclaw_error(self):
        """Verify AdapterError directly inherits from OpenclawError."""
        assert AdapterError.__bases__ == (OpenclawError,)

    def test_routing_error_inherits_from_openclaw_error(self):
        """Verify RoutingError directly inherits from OpenclawError."""
        assert RoutingError.__bases__ == (OpenclawError,)

    def test_timeout_error_inherits_from_openclaw_error(self):
        """Verify TimeoutError directly inherits from OpenclawError."""
        assert TimeoutError.__bases__ == (OpenclawError,)

    def test_validation_error_inherits_from_openclaw_error(self):
        """Verify ValidationError directly inherits from OpenclawError."""
        assert ValidationError.__bases__ == (OpenclawError,)

    def test_duplicate_message_error_inherits_from_openclaw_error(self):
        """Verify DuplicateMessageError directly inherits from OpenclawError."""
        assert DuplicateMessageError.__bases__ == (OpenclawError,)

    def test_openclaw_error_inherits_from_exception(self):
        """Verify OpenclawError inherits from Exception."""
        assert OpenclawError.__bases__ == (Exception,)


class TestErrorCatching:
    """Test error catching by parent and sibling classes."""

    @pytest.mark.parametrize("error_class,name,parent", ERROR_CLASSES)
    def test_all_errors_catchable_as_openclaw_error(self, error_class, name, parent):
        """Verify all specialized errors can be caught as OpenclawError."""
        with pytest.raises(OpenclawError):
            raise error_class(f"{name} test")

    def test_catch_specific_adapter_error(self):
        """Verify AdapterError can be caught specifically."""
        with pytest.raises(AdapterError):
            raise AdapterError("Adapter failed")

    def test_catch_routing_error(self):
        """Verify RoutingError can be caught specifically."""
        with pytest.raises(RoutingError):
            raise RoutingError("Routing failed")

    def test_catch_cycle_detected_error_as_routing_error(self):
        """Verify CycleDetectedError can be caught as RoutingError."""
        with pytest.raises(RoutingError):
            raise CycleDetectedError("Cycle detected")

    def test_cycle_detected_error_not_caught_as_adapter_error(self):
        """Verify CycleDetectedError is not caught as unrelated error type."""
        with pytest.raises(CycleDetectedError):
            try:
                raise CycleDetectedError("Cycle")
            except AdapterError:
                pytest.fail("Should not catch CycleDetectedError as AdapterError")

    def test_multiple_specialized_errors_parent_catching(self):
        """Verify all specialized error types caught by OpenclawError."""
        errors_to_test = [
            AdapterError("adapter"),
            RoutingError("routing"),
            CycleDetectedError("cycle"),
            TimeoutError("timeout"),
            ValidationError("validation"),
            DuplicateMessageError("duplicate"),
        ]

        for error in errors_to_test:
            with pytest.raises(OpenclawError):
                raise error


class TestErrorRaisingAndCatching:
    """Test raising and catching errors in realistic scenarios."""

    def test_raise_and_catch_adapter_error(self):
        """Test raising and catching AdapterError."""
        message = "Failed to parse message"
        try:
            raise AdapterError(message)
        except AdapterError as e:
            assert str(e) == message
            assert isinstance(e, OpenclawError)

    def test_raise_cycle_detected_and_catch_as_routing_error(self):
        """Test raising CycleDetectedError and catching as RoutingError."""
        message = "Circular dependency A->B->C->A"
        try:
            raise CycleDetectedError(message)
        except RoutingError as e:
            assert str(e) == message
            assert isinstance(e, CycleDetectedError)

    def test_error_type_verification_after_catching(self):
        """Test verifying error type after catching by parent."""
        try:
            raise CycleDetectedError("Cycle")
        except OpenclawError as e:
            # Verify we can determine the actual error type
            assert isinstance(e, CycleDetectedError)
            assert isinstance(e, RoutingError)
            assert not isinstance(e, AdapterError)

    def test_error_message_preservation_through_chain(self):
        """Test that error messages are preserved through exception chain."""
        original_message = "Original error message"
        try:
            raise ValidationError(original_message)
        except OpenclawError as e:
            assert str(e) == original_message

    def test_error_chaining_with_cause(self):
        """Test error chaining with exception cause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as orig:
                raise AdapterError("Adapter failed") from orig
        except AdapterError as e:
            assert str(e) == "Adapter failed"
            assert isinstance(e.__cause__, ValueError)


class TestErrorComparisons:
    """Test comparisons and checks between error types."""

    def test_isinstance_checks_all_errors(self):
        """Verify isinstance checks work for all error types."""
        errors = {
            "OpenclawError": OpenclawError("base"),
            "AdapterError": AdapterError("adapter"),
            "RoutingError": RoutingError("routing"),
            "CycleDetectedError": CycleDetectedError("cycle"),
            "TimeoutError": TimeoutError("timeout"),
            "ValidationError": ValidationError("validation"),
            "DuplicateMessageError": DuplicateMessageError("duplicate"),
        }

        # All should be instances of OpenclawError
        for error_name, error in errors.items():
            assert isinstance(error, OpenclawError), f"{error_name} not instance of OpenclawError"

        # CycleDetectedError should also be RoutingError
        assert isinstance(errors["CycleDetectedError"], RoutingError)

    def test_error_type_names(self):
        """Verify error class names are correct."""
        assert OpenclawError.__name__ == "OpenclawError"
        assert AdapterError.__name__ == "AdapterError"
        assert RoutingError.__name__ == "RoutingError"
        assert CycleDetectedError.__name__ == "CycleDetectedError"
        assert TimeoutError.__name__ == "TimeoutError"
        assert ValidationError.__name__ == "ValidationError"
        assert DuplicateMessageError.__name__ == "DuplicateMessageError"


class TestErrorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_error_with_none_message(self):
        """Test error with None message."""
        # Python allows None as exception message
        error = AdapterError(None)
        assert error.args == (None,)

    def test_error_with_multiple_args(self):
        """Test error with multiple arguments."""
        error = ValidationError("Field", "missing", "in message")
        assert error.args == ("Field", "missing", "in message")

    def test_error_inheritance_mro(self):
        """Test Method Resolution Order (MRO) for error hierarchy."""
        cycle_mro = CycleDetectedError.__mro__
        expected_mro = (
            CycleDetectedError,
            RoutingError,
            OpenclawError,
            Exception,
            BaseException,
            object,
        )
        assert cycle_mro == expected_mro

    def test_error_comparison_different_instances(self):
        """Test that different error instances are not equal."""
        error1 = AdapterError("same message")
        error2 = AdapterError("same message")
        # Different instances, so not equal
        assert error1 is not error2

    def test_error_with_very_long_message(self):
        """Test error with very long message."""
        long_message = "x" * 10000
        error = RoutingError(long_message)
        assert str(error) == long_message
        assert len(str(error)) == 10000


class TestErrorImports:
    """Test that all errors can be imported correctly."""

    def test_import_all_errors(self):
        """Verify all error classes can be imported."""
        from openclaw_gateway.errors import (
            OpenclawError,
            AdapterError,
            RoutingError,
            CycleDetectedError,
            TimeoutError,
            ValidationError,
            DuplicateMessageError,
        )

        assert OpenclawError is not None
        assert AdapterError is not None
        assert RoutingError is not None
        assert CycleDetectedError is not None
        assert TimeoutError is not None
        assert ValidationError is not None
        assert DuplicateMessageError is not None

    def test_error_class_docstrings(self):
        """Verify all error classes have docstrings."""
        error_classes = [
            OpenclawError,
            AdapterError,
            RoutingError,
            CycleDetectedError,
            TimeoutError,
            ValidationError,
            DuplicateMessageError,
        ]

        for error_class in error_classes:
            assert error_class.__doc__ is not None
            assert len(error_class.__doc__) > 0
