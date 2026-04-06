"""
EventStreamAdapter: Protocol adapter for event-driven streaming messages.

This module converts between event stream JSON format ({event, chunk_index, total_chunks, data})
and canonical format with streaming intent. Preserves chunk ordering and metadata through
all transformations.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class EventStreamAdapter(ProtocolAdapter):
    """
    Adapter for event-driven streaming messages.

    Converts event stream JSON chunks (with chunk_index, total_chunks metadata) to/from
    canonical messages with intent=stream_result, preserving strict chunk ordering.
    """

    def __init__(self, agent_id: str = "event_stream_agent"):
        """
        Initialize EventStreamAdapter.

        Args:
            agent_id: Unique identifier for this adapter instance. Defaults to "event_stream_agent".
        """
        super().__init__(agent_id=agent_id, protocol_name="event_stream")

    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """
        Parse event stream JSON → CanonicalMessage with intent=stream_result.

        Expected JSON format:
        {
            "event": "chunk",
            "chunk_index": 0,
            "total_chunks": 100,
            "data": "..."
        }

        Args:
            raw_message: Raw bytes from agent (expected UTF-8 JSON)

        Returns:
            CanonicalMessage with intent=stream_result and critical fields:
            - chunk_index: preserved from input
            - total_chunks: preserved from input
            - data: preserved from input

        Raises:
            ValueError: If JSON cannot be parsed or required fields are missing
        """
        try:
            # Parse JSON from bytes
            message_dict = json.loads(raw_message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to parse event stream JSON: {e}")

        # Extract required fields with defaults
        chunk_index = message_dict.get("chunk_index", 0)
        total_chunks = message_dict.get("total_chunks", 1)
        data = message_dict.get("data", "")

        # Validate types
        if not isinstance(chunk_index, int):
            raise ValueError(
                f"chunk_index must be an integer, got {type(chunk_index).__name__}"
            )
        if not isinstance(total_chunks, int):
            raise ValueError(
                f"total_chunks must be an integer, got {type(total_chunks).__name__}"
            )

        # Create canonical message
        canonical = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id=self.agent_id,
            conversation_id="stream_default",  # Will be overridden by router
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "data": data,
            },
            metadata={
                "protocol_source": "event_stream",
                "conversation_chain": [self.agent_id],
                "vector_clock": {self.agent_id: 1},
            },
        )

        return canonical

    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """
        Transform CanonicalMessage → event stream JSON format.

        Args:
            canonical: Valid CanonicalMessage instance with intent=stream_result

        Returns:
            bytes (UTF-8 JSON) in event stream format:
            {
                "event": "chunk",
                "chunk_index": <value>,
                "total_chunks": <value>,
                "data": <value>
            }

        Raises:
            ValueError: If critical fields are missing from payload
        """
        # Validate intent
        if canonical.intent != "stream_result":
            raise ValueError(
                f"EventStreamAdapter.egress() requires intent='stream_result', "
                f"got intent='{canonical.intent}'"
            )

        # Extract critical fields from payload
        payload = canonical.payload
        required_fields = {"chunk_index", "total_chunks", "data"}
        missing = required_fields - set(payload.keys())
        if missing:
            raise ValueError(
                f"Missing critical fields for stream_result: {missing}. "
                f"Payload: {payload}"
            )

        chunk_index = payload["chunk_index"]
        total_chunks = payload["total_chunks"]
        data = payload["data"]

        # Build event stream format
        event_dict: Dict[str, Any] = {
            "event": "chunk",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "data": data,
        }

        # Serialize to JSON bytes
        json_str = json.dumps(event_dict, ensure_ascii=False)
        return json_str.encode("utf-8")
