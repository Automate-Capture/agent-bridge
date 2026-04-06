"""openclaw-gateway: Agent-to-agent message translation gateway."""

from . import adapters
from .canonical_message import CanonicalMessage

__version__ = "0.1.0"

__all__ = ["adapters", "CanonicalMessage"]
