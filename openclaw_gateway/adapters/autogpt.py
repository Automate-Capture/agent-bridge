"""AutoGPT protocol adapter.

Converts between AutoGPT task hierarchy format and canonical message format.
Handles nested subtask structures and intent extraction based on task composition.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openclaw_gateway.canonical_message import CanonicalMessage
from openclaw_gateway.adapters.base import ProtocolAdapter


class AutoGPTAdapter(ProtocolAdapter):
    """
    Converts AutoGPT task hierarchy format ↔ canonical format.

    AutoGPT Message Format (ingest):
    {
        "task": {
            "id": "task_123",
            "description": "Analyze document",
            "subtasks": [
                {"id": "subtask_1", "description": "Extract entities"},
                {"id": "subtask_2", "description": "Generate embeddings"}
            ]
        }
    }

    AutoGPT Message Format (egress):
    {
        "task": {
            "id": "<message_id>",
            "description": "<task_description from payload>",
            "subtasks": [<list of nested task dicts>]
        }
    }

    Intent Mapping:
    - Task with subtasks → intent="delegate"
    - Task without subtasks → intent="analyze"
    """

    def __init__(self, agent_id: str = "autogpt_agent"):
        """
        Initialize AutoGPT adapter.

        Args:
            agent_id: Unique identifier for this adapter instance
        """
        super().__init__(agent_id=agent_id, protocol_name="autogpt")

    async def ingest(self, raw_message: bytes) -> CanonicalMessage:
        """
        Parse AutoGPT task hierarchy JSON → CanonicalMessage.

        Recursively flattens the task hierarchy into a flat structure stored in payload.
        Intent is determined by presence of subtasks:
        - If subtasks exist and non-empty → intent="delegate"
        - If no subtasks or empty list → intent="analyze"

        Args:
            raw_message: Raw bytes from AutoGPT agent (UTF-8 JSON)

        Returns:
            CanonicalMessage with intent extracted from task structure

        Raises:
            ValueError: If raw_message cannot be parsed, missing required fields
        """
        try:
            payload = json.loads(raw_message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode AutoGPT message: {e}")

        # Extract task from payload
        if "task" not in payload:
            raise ValueError("Missing required field 'task' in AutoGPT message")

        task = payload["task"]
        if not isinstance(task, dict):
            raise ValueError("Field 'task' must be a dictionary")

        # Extract basic task information
        task_id = task.get("id")
        if not task_id:
            raise ValueError("Missing required field 'task.id'")

        task_description = task.get("description", "")
        if not task_description:
            raise ValueError("Missing required field 'task.description'")

        # Recursively extract subtasks (flattened structure)
        subtasks = task.get("subtasks", [])
        if not isinstance(subtasks, list):
            raise ValueError("Field 'task.subtasks' must be a list")

        # Determine intent based on subtask presence
        has_subtasks = len(subtasks) > 0
        intent = "delegate" if has_subtasks else "analyze"

        # For delegate intent, build the payload with subtasks
        # For analyze intent, payload only contains task metadata
        if intent == "delegate":
            # Include full task hierarchy in payload for delegate tasks
            canonical_payload = {
                "task_id": task_id,
                "task_description": task_description,
                "target_agent": task.get("target_agent", self.agent_id),
                "deadline": task.get("deadline", datetime.now(timezone.utc).timestamp() + 3600),
                "subtasks": self._flatten_subtasks(subtasks),
            }
        else:
            # For analyze tasks, minimal payload with required fields
            canonical_payload = {
                "document_id": task_id,
                "format": task.get("format", "unknown"),
                "size": task.get("size", 0),
            }

        # Extract or generate conversation_id
        conversation_id = task.get("conversation_id", f"conv_{task_id}")

        # Build canonical message
        # Always store original task in metadata for perfect round-trip preservation
        metadata = {
            "protocol_source": "autogpt",
            "conversation_chain": [self.agent_id],
            "vector_clock": {self.agent_id: 1},
            "_autogpt_original_task": task,  # Store original for round-trip
        }

        canonical = CanonicalMessage(
            message_id=str(uuid.uuid4()),
            source_agent_id=self.agent_id,
            conversation_id=conversation_id,
            message_type="request",
            intent=intent,
            payload=canonical_payload,
            metadata=metadata,
            timestamp_utc=datetime.now(timezone.utc).timestamp(),
        )

        return canonical

    async def egress(self, canonical: CanonicalMessage) -> bytes:
        """
        Transform CanonicalMessage → AutoGPT task hierarchy JSON.

        Reconstructs the task hierarchy from canonical message payload.
        If original task is available in metadata, uses it to preserve structure.
        Otherwise, reconstructs from payload fields.

        Args:
            canonical: Valid CanonicalMessage instance

        Returns:
            bytes (UTF-8 JSON) with AutoGPT task hierarchy format

        Raises:
            ValueError: If required fields missing in canonical message
        """
        # Check if we have the original task in metadata for perfect round-trip
        if (
            canonical.metadata
            and "_autogpt_original_task" in canonical.metadata
        ):
            original_task = canonical.metadata["_autogpt_original_task"]
            autogpt_msg = {"task": original_task}
        else:
            # Reconstruct task from canonical payload
            if canonical.intent == "delegate":
                # Reconstruct from delegate payload
                subtasks_data = canonical.payload.get("subtasks", [])
                subtasks = self._reconstruct_subtasks(subtasks_data)

                task = {
                    "id": canonical.message_id,
                    "description": canonical.payload.get(
                        "task_description", ""
                    ),
                    "target_agent": canonical.payload.get(
                        "target_agent", self.agent_id
                    ),
                    "deadline": canonical.payload.get(
                        "deadline", datetime.now(timezone.utc).timestamp() + 3600
                    ),
                    "subtasks": subtasks,
                }
            else:
                # Reconstruct from analyze payload
                task = {
                    "id": canonical.message_id,
                    "description": canonical.payload.get("document_id", ""),
                    "format": canonical.payload.get("format", "unknown"),
                    "size": canonical.payload.get("size", 0),
                    "subtasks": [],  # Always include subtasks field, even if empty
                }

            autogpt_msg = {"task": task}

        return json.dumps(autogpt_msg).encode("utf-8")

    def _flatten_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recursively flatten nested subtask structure.

        Preserves all subtask properties and maintains hierarchy information.

        Args:
            subtasks: List of subtask dictionaries (may be nested)

        Returns:
            Flattened list of subtask dictionaries with preserved properties
        """
        flattened = []
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                continue

            # Create a copy to avoid modifying original
            flattened_subtask = dict(subtask)

            # Recursively flatten nested subtasks
            if "subtasks" in flattened_subtask:
                nested = flattened_subtask.pop("subtasks", [])
                if nested:
                    flattened_subtask["subtasks"] = self._flatten_subtasks(nested)
                else:
                    flattened_subtask["subtasks"] = []

            flattened.append(flattened_subtask)

        return flattened

    def _reconstruct_subtasks(
        self, subtasks_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct nested subtask structure from flattened representation.

        Handles both flat and nested subtask lists.

        Args:
            subtasks_data: List of subtask dictionaries (flat or nested)

        Returns:
            Reconstructed subtask list preserving original nesting
        """
        reconstructed = []
        for subtask_data in subtasks_data:
            if not isinstance(subtask_data, dict):
                continue

            # Create a copy to avoid modifying original
            reconstructed_subtask = dict(subtask_data)

            # Recursively reconstruct nested subtasks
            if "subtasks" in reconstructed_subtask:
                nested = reconstructed_subtask["subtasks"]
                if nested:
                    reconstructed_subtask["subtasks"] = (
                        self._reconstruct_subtasks(nested)
                    )

            reconstructed.append(reconstructed_subtask)

        return reconstructed
