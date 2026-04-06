"""
Write-Ahead Log (WAL) module for persistent gateway state.

This module provides unified WAL operations across all components:
- state_coordinator: state updates and vector clocks
- context_manager: conversation contexts and delegation chains
- deduplication: processed message tracking
- failure_recovery: operation logs for rollback

WAL format: JSON lines, one entry per line
Entry schema:
{
  "entry_id": int,
  "timestamp": float (unix),
  "component": str,
  "conversation_id": str,
  "operation_type": str,
  "data": dict
}
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WAL:
    """Write-Ahead Log for persistent state management."""

    def __init__(self, wal_path: str = "/var/lib/openclaw/gateway.wal"):
        """
        Initialize WAL.

        Args:
            wal_path: Path to WAL file
        """
        self._wal_path = wal_path
        self._next_entry_id = 1
        self._ensure_wal_dir()

    def _ensure_wal_dir(self) -> None:
        """Ensure WAL directory exists."""
        wal_dir = os.path.dirname(self._wal_path)
        if wal_dir and not os.path.exists(wal_dir):
            try:
                os.makedirs(wal_dir, mode=0o755, exist_ok=True)
            except OSError as e:
                logger.warning(f"Failed to create WAL directory {wal_dir}: {e}")

    def write(
        self,
        component: str,
        conversation_id: str,
        operation_type: str,
        data: Dict[str, Any],
    ) -> int:
        """
        Write entry to WAL.

        Args:
            component: Component name (state_coordinator, context_manager, etc.)
            conversation_id: Conversation identifier
            operation_type: Type of operation
            data: Operation-specific payload

        Returns:
            Entry ID for recovery reference
        """
        entry = {
            "entry_id": self._next_entry_id,
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "component": component,
            "conversation_id": conversation_id,
            "operation_type": operation_type,
            "data": data,
        }

        try:
            with open(self._wal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._next_entry_id += 1
            return entry["entry_id"]
        except OSError as e:
            logger.error(f"Failed to write WAL entry: {e}")
            raise

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all entries from WAL.

        Returns:
            List of WAL entries in order

        Handles:
            - Missing WAL file (returns empty list)
            - Malformed JSON lines (skips with warning, continues)
        """
        if not os.path.exists(self._wal_path):
            logger.info(f"WAL file {self._wal_path} not found")
            return []

        entries = []
        try:
            with open(self._wal_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                        # Track highest entry_id for next writes
                        if "entry_id" in entry:
                            self._next_entry_id = max(
                                self._next_entry_id, entry["entry_id"] + 1
                            )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed WAL entry at line {line_num}: {e}"
                        )
                        continue

            logger.info(f"Read {len(entries)} entries from WAL")
            return entries
        except OSError as e:
            logger.error(f"Failed to read WAL file: {e}")
            return []

    def clear(self) -> None:
        """
        Clear WAL file.

        Used for testing and explicit state resets.
        """
        try:
            if os.path.exists(self._wal_path):
                os.remove(self._wal_path)
            self._next_entry_id = 1
            logger.info(f"Cleared WAL file {self._wal_path}")
        except OSError as e:
            logger.error(f"Failed to clear WAL file: {e}")

    def get_entries_for_component(
        self, component: str
    ) -> List[Dict[str, Any]]:
        """
        Get all entries for a specific component.

        Args:
            component: Component name to filter by

        Returns:
            List of entries matching component
        """
        all_entries = self.read_all()
        return [e for e in all_entries if e.get("component") == component]
