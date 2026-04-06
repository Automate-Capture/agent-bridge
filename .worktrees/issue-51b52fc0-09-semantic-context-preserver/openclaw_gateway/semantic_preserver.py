"""Semantic context preserver for message validation and confidence scoring.

This module provides the SemanticContextPreserver class that validates semantic
intent preservation and critical field presence during message transformation.
It computes confidence scores (% of critical fields present) and rejects messages
below the 95% confidence threshold.

The semantic preserver ensures no data loss or semantic drift when messages cross
protocol boundaries by:
1. Extracting and comparing semantic intent (analyze, delegate, stream_result)
2. Validating all critical fields required for the intent are present
3. Computing confidence score as % of critical fields present
4. Enforcing threshold: message rejected if score < 0.95
"""

import logging
from typing import Optional
from openclaw_gateway.canonical_message import CanonicalMessage, CRITICAL_FIELDS_BY_INTENT
from openclaw_gateway.errors import ValidationError

# Configure module logger
logger = logging.getLogger(__name__)

# Confidence threshold: messages must have >= 95% of critical fields to pass
CONFIDENCE_THRESHOLD = 0.95


class SemanticContextPreserver:
    """Validates semantic intent preservation and critical field presence.

    This class ensures that messages preserve their semantic intent and critical
    fields during protocol transformation. It provides:
    - Intent extraction from messages
    - Intent preservation validation (original == transformed)
    - Critical field verification per intent type
    - Confidence scoring based on field presence %
    - Threshold enforcement (>= 0.95 required)

    All operations complete in < 5ms per message for < 100 fields.
    """

    def __init__(self):
        """Initialize the semantic context preserver."""
        self._critical_fields = CRITICAL_FIELDS_BY_INTENT

    def extract_intent(self, message: CanonicalMessage) -> str:
        """Extract semantic intent from a message.

        The intent indicates the semantic action the message represents:
        - "analyze": Document analysis request with document_id, format, size
        - "delegate": Task delegation with target_agent, task_description, deadline
        - "stream_result": Streaming result chunk with chunk_index, total_chunks, data

        Args:
            message: The CanonicalMessage to extract intent from

        Returns:
            The intent string: "analyze", "delegate", or "stream_result"

        Raises:
            ValidationError: If message.intent is invalid or missing
        """
        if not message or not hasattr(message, 'intent'):
            raise ValidationError("Message missing intent field")

        intent = message.intent
        if intent not in self._critical_fields:
            raise ValidationError(
                f"Invalid intent '{intent}'. "
                f"Valid intents: {set(self._critical_fields.keys())}"
            )

        return intent

    def compute_confidence(self, message: CanonicalMessage) -> float:
        """Compute confidence score as % of critical fields present.

        Confidence is calculated as:
            confidence = present_fields / total_critical_fields

        For example:
        - 10/10 fields → confidence = 1.0 (100%)
        - 9/10 fields → confidence = 0.9 (90%)
        - 9.5/10 fields → confidence = 0.95 (95%) - exact threshold

        Args:
            message: The CanonicalMessage to score

        Returns:
            Float between 0.0 and 1.0 representing confidence percentage

        Raises:
            ValidationError: If intent is invalid or payload is missing
        """
        if not message or not hasattr(message, 'intent'):
            raise ValidationError("Message missing intent field")

        intent = message.intent
        if intent not in self._critical_fields:
            raise ValidationError(
                f"Invalid intent '{intent}'. "
                f"Valid intents: {set(self._critical_fields.keys())}"
            )

        # Get critical fields for this intent
        required_fields = self._critical_fields[intent]
        if not required_fields:
            # No critical fields defined for this intent
            return 1.0

        # Count how many critical fields are present in payload
        payload = message.payload or {}
        present_count = sum(1 for field in required_fields if field in payload)

        # Calculate confidence as present/total
        total_count = len(required_fields)
        confidence = present_count / total_count if total_count > 0 else 1.0

        return confidence

    def validate(
        self,
        original: CanonicalMessage,
        transformed: CanonicalMessage
    ) -> bool:
        """Validate semantic intent preservation and critical fields.

        This method performs three validations:
        1. Intent preservation: original.intent == transformed.intent (string equality)
        2. Critical fields: All required fields for the intent are present
        3. Confidence threshold: Confidence score >= 0.95

        A message passes validation only if ALL three conditions are met.

        Args:
            original: The original CanonicalMessage
            transformed: The transformed CanonicalMessage after protocol translation

        Returns:
            True if validation passes (intent preserved, fields present, confidence >= 0.95)
            False if validation fails for any reason

        Raises:
            ValidationError: If messages are invalid (missing intent, etc.)

        Examples:
            >>> original = CanonicalMessage(
            ...     message_id="550e8400-e29b-41d4-a716-446655440000",
            ...     source_agent_id="agent_a",
            ...     conversation_id="conv_123",
            ...     message_type="request",
            ...     intent="analyze",
            ...     payload={"document_id": "doc_1", "format": "pdf", "size": 1024}
            ... )
            >>> transformed = original  # Same message (100% confidence)
            >>> preserver = SemanticContextPreserver()
            >>> preserver.validate(original, transformed)
            True

            >>> # With missing field (9/10 = 0.9 confidence, rejected)
            >>> incomplete = CanonicalMessage(
            ...     message_id="550e8400-e29b-41d4-a716-446655440001",
            ...     source_agent_id="agent_b",
            ...     conversation_id="conv_123",
            ...     message_type="request",
            ...     intent="analyze",
            ...     payload={"document_id": "doc_1", "format": "pdf"}  # Missing "size"
            ... )
            >>> preserver.validate(original, incomplete)
            False
        """
        try:
            # Validation 1: Both messages must have valid intent
            original_intent = self.extract_intent(original)
            transformed_intent = self.extract_intent(transformed)

            # Validation 2: Intent must be preserved (string equality)
            if original_intent != transformed_intent:
                logger.warning(
                    f"Intent mismatch: original='{original_intent}' "
                    f"transformed='{transformed_intent}'"
                )
                return False

            # Validation 3: All critical fields must be present in transformed message
            required_fields = self._critical_fields[transformed_intent]
            payload = transformed.payload or {}
            missing_fields = required_fields - set(payload.keys())

            if missing_fields:
                logger.warning(
                    f"Missing critical fields for intent '{transformed_intent}': "
                    f"{missing_fields}. Required: {required_fields}, Got: {set(payload.keys())}"
                )
                return False

            # Validation 4: Confidence score must be >= threshold
            confidence = self.compute_confidence(transformed)
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"Confidence score {confidence:.2f} below threshold "
                    f"{CONFIDENCE_THRESHOLD}. Fields: {len(payload)}/{len(required_fields)}"
                )
                return False

            return True

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            raise ValidationError(f"Validation failed: {e}")
