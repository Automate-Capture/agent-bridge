"""openclaw-gateway: Agent-to-agent message translation gateway."""

from .context_manager import ConversationContextManager, ConversationContext
from .canonical_message import CanonicalMessage

__version__ = "0.1.0"

__all__ = [
    "ConversationContextManager",
    "ConversationContext",
    "CanonicalMessage",
]
