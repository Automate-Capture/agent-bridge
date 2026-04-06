"""Semantic context preservation validation module.

This module implements SemanticContextPreserver, which validates that message
intent and critical fields are preserved during protocol transformations.

Key responsibilities:
- Extract semantic intent from messages
- Validate transformed messages preserve original intent
- Verify critical fields are present based on intent type
- Compute confidence score (% of critical fields present)
- Reject messages with confidence < 0.95 (95% threshold)

Performance requirement: All operations < 5ms per message.
"""

import logging
from typing import Dict, Set

from openclaw_gateway.canonical_message import (
    CRITICAL_FIELDS_BY_INTENT,
    CanonicalMessage,
)

logger = logging.getLogger(__name__)


class SemanticContextPreserver:
    """
    Validates semantic consistency through message transformations.

    Ensures that when messages are transformed between different agent protocols,
    the core semantic intent and all critical fields are preserved.

    Responsibilities:
    - Extract intent from messages
    - Validate transformed message preserves original intent
    - Verify critical fields present based on intent
    - Compute confidence score (% of required fields present)
    - Enforce 95% confidence threshold for message acceptance

    Attributes:
        CONFIDENCE_THRESHOLD (float): Minimum confidence score for acceptance (0.95)
    """

    CONFIDENCE_THRESHOLD = 0.95

    def validate(
        self, original: CanonicalMessage, transformed: CanonicalMessage
    ) -> bool:
        """
        Validate that transformed message preserves semantics of original.

        Performs three checks:
        1. Intent preserved (string equality)
        2. Critical fields present (based on intent type)
        3. Confidence score >= 0.95 threshold

        Args:
            original: Original CanonicalMessage before transformation
            transformed: Transformed CanonicalMessage after protocol bridge

        Returns:
            True if all checks pass, False if any check fails

        Performance: < 5ms per validation
        """
        # Check 1: Intent preserved (string equality)
        if original.intent != transformed.intent:
            logger.warning(
                f"Intent mismatch: {original.intent} → {transformed.intent}"
            )
            return False

        # Check 2 & 3: Critical fields and confidence score
        confidence = self.compute_confidence(transformed)

        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Confidence {confidence:.2f} below threshold {self.CONFIDENCE_THRESHOLD} "
                f"for intent {original.intent}"
            )
            return False

        logger.debug(
            f"Validation passed: intent={original.intent}, confidence={confidence:.2f}"
        )
        return True

    def compute_confidence(self, message: CanonicalMessage) -> float:
        """
        Compute confidence score based on critical fields presence.

        Calculates the percentage of required fields that are present in the
        message payload based on the message's intent type.

        Formula: score = fields_present / fields_required

        Args:
            message: CanonicalMessage to evaluate

        Returns:
            Confidence score in [0.0, 1.0]:
            - 1.0 = all critical fields present
            - 0.95 = 19/20 fields present (boundary case - accepted)
            - 0.9 = 9/10 fields present (rejected)
            - 0.0 = no fields present

        Examples:
            >>> msg_full = CanonicalMessage(...)  # all critical fields
            >>> preserver.compute_confidence(msg_full)
            1.0

            >>> msg_partial = CanonicalMessage(...)  # 9/10 fields
            >>> preserver.compute_confidence(msg_partial)
            0.9

            >>> msg_boundary = CanonicalMessage(...)  # 19/20 fields
            >>> preserver.compute_confidence(msg_boundary)
            0.95
        """
        intent = self.extract_intent(message)
        required_fields = CRITICAL_FIELDS_BY_INTENT.get(intent, set())

        if not required_fields:
            # Unknown intent with no required fields = perfect score
            return 1.0

        payload_keys = set(message.payload.keys())
        present_count = len(payload_keys & required_fields)
        total_count = len(required_fields)

        if total_count == 0:
            return 1.0

        score = present_count / total_count
        logger.debug(
            f"Confidence score for intent '{intent}': {present_count}/{total_count} = {score:.2f}"
        )
        return score

    def extract_intent(self, message: CanonicalMessage) -> str:
        """
        Extract semantic intent from a message.

        Returns the message's intent field, which indicates the semantic
        purpose of the message. Valid intents are:
        - "analyze": Document analysis request
        - "delegate": Task delegation to another agent
        - "stream_result": Streaming result chunk delivery

        Args:
            message: CanonicalMessage to extract intent from

        Returns:
            Intent string from the message (one of: "analyze", "delegate", "stream_result")

        Examples:
            >>> msg = CanonicalMessage(intent="analyze", ...)
            >>> preserver.extract_intent(msg)
            'analyze'
        """
        return message.intent
