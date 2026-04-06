"""
CanonicalMessage: Unified message format for all inter-agent communication.

This module defines the CanonicalMessage Pydantic model that serves as the
single source of truth for inter-agent communication, normalizing incompatible
agent protocols (LangChain, AutoGPT, event streaming) to a unified JSON format.

Key features:
- All timestamps are unix floats internally, serialized to ISO-8601 for JSON
- All message_ids are UUID-v4 format, validated at ingestion time
- Critical fields validated per intent type (analyze, delegate, stream_result)
- Automatic metadata field defaults for missing protocol metadata
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator, field_serializer, model_validator


# Critical fields by intent (for semantic validation)
CRITICAL_FIELDS_BY_INTENT: Dict[str, Set[str]] = {
    "analyze": {"document_id", "format", "size"},
    "delegate": {"target_agent", "task_description", "deadline"},
    "stream_result": {"chunk_index", "total_chunks", "data"},
}


class CanonicalMessage(BaseModel):
    """
    Canonical message format for all inter-agent communication.

    All timestamps are unix floats internally, serialized to ISO-8601 for JSON.
    All message_ids are UUID-v4 format.
    Critical fields validated per intent type.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_id": "123e4567-e89b-12d3-a456-426614174000",
                "source_agent_id": "langchain_analyzer",
                "conversation_id": "root_123",
                "message_type": "request",
                "intent": "analyze",
                "payload": {
                    "document_id": "doc_abc123",
                    "format": "pdf",
                    "size": 1048576,
                },
                "metadata": {
                    "protocol_source": "langchain",
                    "conversation_chain": ["langchain_analyzer"],
                    "vector_clock": {"langchain_analyzer": 1},
                },
                "timestamp_utc": "2026-04-05T12:34:56.789012Z",
            }
        }
    )

    message_id: str = Field(
        ...,
        description="UUID-v4 format message identifier",
    )
    source_agent_id: str = Field(
        ...,
        description="Agent that originated this message",
    )
    conversation_id: str = Field(
        ...,
        description="Hierarchical conversation ID (e.g., root_123.subtask_1)",
    )
    message_type: Literal["request", "response", "state_update"] = Field(
        ...,
        description="Type of message",
    )
    intent: Literal["analyze", "delegate", "stream_result"] = Field(
        ...,
        description="Semantic intent of the message",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message-specific data, validated by intent",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Protocol metadata (source, timestamp, chain, vector clock)",
    )
    timestamp_utc: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="Unix timestamp (float) of message creation",
    )

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        """Ensure message_id is UUID-v4 format."""
        try:
            # Parse UUID without version parameter to check actual version
            parsed = uuid.UUID(v)
            if parsed.version != 4:
                raise ValueError(
                    f"message_id must be UUID version 4, got version {parsed.version}"
                )
        except ValueError as e:
            raise ValueError(f"Invalid UUID-v4 message_id '{v}': {e}")
        return v

    @field_validator("payload")
    @classmethod
    def validate_critical_fields(
        cls, v: Dict[str, Any], info
    ) -> Dict[str, Any]:
        """Validate critical fields are present based on intent."""
        intent = info.data.get("intent")

        if intent == "analyze":
            required = {"document_id", "format", "size"}
        elif intent == "delegate":
            required = {"target_agent", "task_description", "deadline"}
        elif intent == "stream_result":
            required = {"chunk_index", "total_chunks", "data"}
        else:
            return v  # Unknown intent, skip validation

        missing = required - set(v.keys())
        if missing:
            raise ValueError(
                f"Missing critical fields for intent '{intent}': {missing}. "
                f"Required: {required}, Got: {set(v.keys())}"
            )

        return v

    @model_validator(mode="after")
    def ensure_metadata_fields(self) -> "CanonicalMessage":
        """Ensure required metadata fields are present."""
        # Handle None metadata by converting to empty dict
        if self.metadata is None:
            self.metadata = {}

        # Set defaults if missing
        if "protocol_source" not in self.metadata:
            self.metadata["protocol_source"] = "unknown"
        if "conversation_chain" not in self.metadata:
            self.metadata["conversation_chain"] = [self.source_agent_id]
        if "vector_clock" not in self.metadata:
            self.metadata["vector_clock"] = {self.source_agent_id: 1}

        return self

    @field_validator("timestamp_utc", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        """Parse ISO-8601 timestamp strings to unix float."""
        if isinstance(v, str):
            # Parse ISO-8601 timestamp string to datetime, then to unix float
            # Handle formats with and without Z suffix
            v_clean = v.rstrip("Z")
            try:
                # Try parsing with microseconds first
                if "." in v_clean:
                    dt = datetime.fromisoformat(v_clean)
                else:
                    dt = datetime.fromisoformat(v_clean)
                # Ensure it's timezone-aware (assume UTC if not)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError as e:
                raise ValueError(f"Invalid ISO-8601 timestamp format: {v}") from e
        return float(v)

    @field_serializer("timestamp_utc")
    def serialize_timestamp(self, value: float, _info) -> str:
        """Serialize unix timestamp to ISO-8601 format."""
        return datetime.fromtimestamp(value, timezone.utc).isoformat() + "Z"
