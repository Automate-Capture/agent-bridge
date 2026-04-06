"""
LangChain Protocol Adapter

Converts between LangChain tool_calls JSON format and the canonical message format.

LangChain Message Format (ingest):
{
    "tool_calls": [
        {
            "id": "call_abc123",
            "function": "analyze_document",
            "arguments": {"document_id": "...", "format": "pdf", "size": 1024}
        },
        ...
    ]
}

LangChain Message Format (egress):
{
    "tool_calls": [
        {
            "id": "<message_id from canonical>",
            "function": "<intent mapped to function name>",
            "arguments": <payload from canonical>
        },
        ...
    ]
}

This adapter:
- Extracts intent from function names (analyze_document → analyze)
- Preserves all tool_calls in metadata for round-trip losslessness
- Maps intents back to function names on egress
- Handles multiple tool_calls in a single message
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from openclaw_gateway.adapters.base import ProtocolAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


class LangChainAdapter(ProtocolAdapter):
    """
    Protocol adapter for LangChain tool_calls format.

    Implements ingest() to convert LangChain tool_calls → CanonicalMessage
    and egress() to convert CanonicalMessage → LangChain tool_calls format.
    """

    # Intent mapping: semantic intent → LangChain function name
    INTENT_TO_FUNCTION = {
        "analyze": "analyze_document",
        "delegate": "delegate_task",
        "stream_result": "stream_result",
    }

    # Reverse mapping: LangChain function name → semantic intent
    FUNCTION_TO_INTENT = {v: k for k, v in INTENT_TO_FUNCTION.items()}

    def __init__(self, agent_id: str = "langchain_agent") -> None:
        """
        Initialize LangChainAdapter.

        Args:
            agent_id: Unique identifier for this adapter instance (default: "langchain_agent")
        """
        super().__init__(agent_id=agent_id, protocol_name="langchain")

    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """
        Parse LangChain tool_calls → CanonicalMessage.

        Extracts semantic intent from function names and preserves all tool_calls
        in metadata for round-trip preservation.

        Args:
            raw_message: Raw bytes (UTF-8 JSON) containing tool_calls array

        Returns:
            CanonicalMessage with intent extracted from first tool_call's function name
            and all tool_calls preserved in metadata

        Raises:
            ValueError: If JSON is malformed, no tool_calls present, or parsing fails
        """
        try:
            payload = json.loads(raw_message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode LangChain message: {e}")

        # Extract tool_calls array
        tool_calls = payload.get("tool_calls", [])
        if not tool_calls:
            raise ValueError("No tool_calls in LangChain message")

        # Extract intent from first tool_call's function name
        first_tool_call = tool_calls[0]
        function_name = first_tool_call.get("function", "unknown")
        intent = self.FUNCTION_TO_INTENT.get(function_name, "delegate")

        # Build arguments from first tool_call
        arguments = first_tool_call.get("arguments", {})

        # Extract or generate conversation_id
        conversation_id = arguments.pop("conversation_id", f"conv_{uuid.uuid4().hex[:8]}")

        # Build canonical payload with arguments from first tool_call
        canonical_payload = arguments.copy()

        # Build canonical message
        canonical = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id=self.agent_id,
            conversation_id=conversation_id,
            message_type="request",
            intent=intent,
            payload=canonical_payload,
            metadata={
                "protocol_source": "langchain",
                "conversation_chain": [self.agent_id],
                "vector_clock": {self.agent_id: 1},
                "tool_calls": tool_calls,  # Preserve all tool_calls in metadata
            },
            timestamp_utc=datetime.now(timezone.utc).timestamp(),
        )

        return canonical

    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """
        Transform CanonicalMessage → LangChain tool_calls format.

        Maps canonical intent back to function name and reconstructs tool_calls.
        If original tool_calls are preserved in metadata, uses them exactly.
        Otherwise reconstructs a single tool_call from canonical payload.

        Args:
            canonical: Valid CanonicalMessage instance

        Returns:
            bytes (UTF-8 JSON) with tool_calls array suitable for LangChain

        Raises:
            ValueError: If transformation cannot be completed
        """
        # Check if original tool_calls are preserved in metadata
        if "tool_calls" in canonical.metadata and canonical.metadata["tool_calls"]:
            tool_calls = canonical.metadata["tool_calls"]
        else:
            # Reconstruct single tool_call from canonical
            function_name = self.INTENT_TO_FUNCTION.get(
                canonical.intent, "unknown_function"
            )
            tool_calls = [
                {
                    "id": canonical.message_id,
                    "function": function_name,
                    "arguments": canonical.payload,
                }
            ]

        # Build LangChain format
        langchain_msg = {"tool_calls": tool_calls}

        return json.dumps(langchain_msg).encode("utf-8")
