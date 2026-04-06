"""Canonical message format for all inter-agent communication.

This module defines the CanonicalMessage class, a Pydantic model that serves as
the single source of truth for all inter-agent message translation. It provides:

1. Validation of message structure and types
2. UUID v4 enforcement for message_id
3. Semantic intent literals (analyze, delegate, stream_result)
4. Payload validation per intent type
5. JSON serialization with ISO-8601 timestamp format
6. Metadata auto-population with defaults

The canonical format eliminates O(n²) custom translator code between agent types.
All timestamps are stored as unix floats internally but serialized to ISO-8601
JSON format for external consumption.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, field_serializer
from typing import Dict, Any, Literal, Set
from datetime import datetime, timezone
import uuid as uuid_module
import json as json_module


# Critical fields by intent (for semantic validation)
CRITICAL_FIELDS_BY_INTENT: Dict[str, Set[str]] = {
    "analyze": {"document_id", "format", "size"},
    "delegate": {"target_agent", "task_description", "deadline"},
    "stream_result": {"chunk_index", "total_chunks", "data"}
}


class CanonicalMessage(BaseModel):
    """
    Canonical message format for all inter-agent communication.

    All timestamps are unix floats internally, serialized to ISO-8601 for JSON.
    All message_ids are UUID-v4 format.
    Critical fields validated per intent type.

    Attributes:
        message_id: UUID-v4 format message identifier (validated)
        source_agent_id: Agent that originated this message
        conversation_id: Hierarchical conversation ID (e.g., root_123.subtask_1)
        message_type: Type of message (request, response, state_update)
        intent: Semantic intent of the message (analyze, delegate, stream_result)
        payload: Message-specific data, validated by intent type
        metadata: Protocol metadata (source, timestamp, chain, vector clock)
        timestamp_utc: Unix timestamp (float) of message creation
    """

    model_config = ConfigDict()

    message_id: str = Field(
        ...,
        description="UUID-v4 format message identifier"
    )
    source_agent_id: str = Field(
        ...,
        description="Agent that originated this message"
    )
    conversation_id: str = Field(
        ...,
        description="Hierarchical conversation ID (e.g., root_123.subtask_1)"
    )
    message_type: Literal["request", "response", "state_update"] = Field(
        ...,
        description="Type of message"
    )
    intent: Literal["analyze", "delegate", "stream_result"] = Field(
        ...,
        description="Semantic intent of the message"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message-specific data, validated by intent"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Protocol metadata (source, timestamp, chain, vector clock)"
    )
    timestamp_utc: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="Unix timestamp (float) of message creation"
    )

    @field_validator('message_id')
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        """Ensure message_id is UUID-v4 format.

        Args:
            v: The message_id string to validate

        Returns:
            The validated message_id string

        Raises:
            ValueError: If message_id is not a valid UUID v4
        """
        if not v:
            raise ValueError("message_id cannot be empty")

        try:
            # Parse the UUID
            parsed = uuid_module.UUID(v)
            # Check if it's version 4
            if parsed.version != 4:
                raise ValueError(
                    f"message_id must be UUID version 4, got version {parsed.version}"
                )
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid UUID-v4 message_id '{v}': {e}")

        return v

    @model_validator(mode='after')
    def validate_payload_by_intent(self) -> "CanonicalMessage":
        """Validate critical fields are present based on intent.

        Returns:
            The validated CanonicalMessage instance

        Raises:
            ValueError: If critical fields for the intent are missing
        """
        intent = self.intent

        # Only validate if intent is known
        if intent not in CRITICAL_FIELDS_BY_INTENT:
            return self

        required = CRITICAL_FIELDS_BY_INTENT[intent]
        payload_keys = set(self.payload.keys())
        missing = required - payload_keys

        if missing:
            raise ValueError(
                f"Missing critical fields for intent '{intent}': {missing}. "
                f"Required: {required}, Got: {payload_keys}"
            )

        return self

    @model_validator(mode='after')
    def ensure_metadata_fields(self) -> "CanonicalMessage":
        """Ensure required metadata fields are present with defaults.

        Returns:
            The CanonicalMessage with metadata defaults populated
        """
        # Initialize if None
        if self.metadata is None:
            self.metadata = {}

        # Set defaults if missing
        if 'protocol_source' not in self.metadata:
            self.metadata['protocol_source'] = 'unknown'

        if 'conversation_chain' not in self.metadata:
            self.metadata['conversation_chain'] = [self.source_agent_id]

        if 'vector_clock' not in self.metadata:
            self.metadata['vector_clock'] = {self.source_agent_id: 1}

        return self

    @field_serializer('timestamp_utc', when_used='json')
    def serialize_timestamp_utc(self, value: float) -> str:
        """Serialize unix timestamp to ISO-8601 format for JSON.

        Args:
            value: Unix timestamp float

        Returns:
            ISO-8601 formatted timestamp string with Z suffix
        """
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', '') + 'Z'

    @classmethod
    def model_validate_json(cls, json_data: str) -> "CanonicalMessage":
        """Deserialize JSON with ISO-8601 timestamp conversion.

        This method overrides the default behavior to handle ISO-8601 timestamp
        conversion back to unix float format.

        Args:
            json_data: JSON string with ISO-8601 timestamp_utc field

        Returns:
            CanonicalMessage instance with unix float timestamp_utc

        Raises:
            ValueError: If JSON is invalid or timestamp conversion fails
        """
        # Parse the JSON
        data = json_module.loads(json_data)

        # Convert ISO-8601 timestamp back to unix float if needed
        if 'timestamp_utc' in data:
            ts = data['timestamp_utc']
            # If it's a string in ISO-8601 format, convert to unix float
            if isinstance(ts, str):
                # Handle ISO-8601 format: "2026-04-05T12:34:56.789012Z"
                # Remove trailing Z and parse
                if ts.endswith('Z'):
                    ts = ts[:-1]
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00').split('+')[0])
                # If no timezone info, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                data['timestamp_utc'] = dt.timestamp()

        # Use Pydantic's model_validate for validation
        return cls.model_validate(data)

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON with ISO-8601 timestamp format.

        Args:
            **kwargs: Additional arguments to pass to model_dump_json

        Returns:
            JSON string with timestamp_utc in ISO-8601 format
        """
        # Use the parent class method which applies json_encoders via model_config
        return super().model_dump_json(**kwargs)
