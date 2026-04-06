# AgentBridge API Reference

AgentBridge is a production-grade autonomous agent gateway system designed to enable dynamic protocol adaptation and semantic message routing between heterogeneous AI agent architectures. It facilitates distributed state coordination and robust failure recovery.

---

## Core Modules

### `openclaw_gateway.canonical_message`

This module defines the standardized, canonical message format used across the entire AgentBridge system, ensuring interoperability between different agent protocols.

**Key Classes/Functions:**

*   **`CanonicalMessage` (Class)**
    *   **Signature:** `CanonicalMessage(sender_id: str, recipient_id: str, message_type: str, payload: dict, metadata: dict = None)`
    *   **Description:** Represents the standardized message structure. It encapsulates all necessary information (sender, recipient, type, and content) in a format agnostic to the underlying agent implementation.
    *   **Example Usage:**
        ```python
        from openclaw_gateway.canonical_message import CanonicalMessage
        msg = CanonicalMessage(
            sender_id="AgentA",
            recipient_id="AgentB",
            message_type="TOOL_CALL",
            payload={"tool_name": "search", "args": {"query": "latest news"}},
            metadata={"timestamp": "2023-10-27T10:00:00Z"}
        )
        ```

### `openclaw_gateway.errors`

This module centralizes custom exceptions used throughout the gateway for standardized error handling, particularly during protocol translation and routing failures.

**Key Classes/Functions:**

*   **`GatewayError` (Base Exception)**
    *   **Signature:** `class GatewayError(Exception): ...`
    *   **Description:** The base exception class for all AgentBridge operational errors.
    *   **Example Usage:**
        ```python
        try:
            # ... gateway operation ...
        except GatewayError as e:
            print(f"A gateway-specific error occurred: {e}")
        ```
*   **`ProtocolMismatchError` (Exception)**
    *   **Signature:** `class ProtocolMismatchError(GatewayError): ...`
    *   **Description:** Raised when a message cannot be successfully translated between the source and target agent protocols.
    *   **Example Usage:**
        ```python
        raise ProtocolMismatchError("Cannot map LangChain ToolCall to AutoGPT Command structure.")
        ```

### `openclaw_gateway.adapters.base`

This module defines the abstract interface that all specific protocol adapters must implement. This enforces a contract for message translation and interaction with heterogeneous agent systems.

**Key Classes/Functions:**

*   **`BaseAdapter` (Abstract Class)**
    *   **Signature:** `class BaseAdapter(ABC): ...`
    *   **Description:** Defines the required methods for any protocol adapter: `to_canonical`, `from_canonical`, and potentially `send_message`.
    *   **Methods:**
        *   `to_canonical(native_message: Any) -> CanonicalMessage`: Converts a native agent message format into the system's canonical format.
        *   `from_canonical(canonical_message: CanonicalMessage) -> Any`: Converts a canonical message back into the format expected by the specific agent.
    *   **Example Usage (Conceptual):**
        ```python
        from openclaw_gateway.adapters.base import BaseAdapter
        class MyCustomAdapter(BaseAdapter):
            def to_canonical(self, native_msg):
                # Implementation to map native_msg to CanonicalMessage
                pass
        ```

### `openclaw_gateway.adapters.langchain`

This module provides the concrete implementation for interfacing with agents utilizing the LangChain framework. It handles the translation between LangChain-specific structures (e.g., `ToolCall` objects) and the canonical format.

**Key Classes/Functions:**

*   **`LangChainAdapter` (Class)**
    *   **Signature:** `LangChainAdapter(config: dict)`
    *   **Description:** Implements `BaseAdapter` specifically for LangChain agents. It manages the mapping logic for LangChain's internal message structures.
    *   **Methods:**
        *   `to_canonical(langchain_message: dict) -> CanonicalMessage`: Converts a LangChain message structure into `CanonicalMessage`.
        *   `from_canonical(canonical_message: CanonicalMessage) -> dict`: Converts a `CanonicalMessage` into a format consumable by a LangChain agent.
    *   **Example Usage:**
        ```python
        from openclaw_gateway.adapters.langchain import LangChainAdapter
        adapter = LangChainAdapter(config={"api_key": "..."})
        # Assume 'lc_tool_call' is a structure from a LangChain agent
        canonical = adapter.to_canonical(lc_tool_call)
        ```

### `openclaw_gateway.adapters` (General)

This directory houses various specialized adapters (e.g., `AutoGPTAdapter`, `VectorContextAdapter`) that implement the `BaseAdapter` interface for different agent architectures.

**Key Classes/Functions:**

*   **`AutoGPTAdapter` (Conceptual Class)**
    *   **Signature:** `AutoGPTAdapter(config: dict)`
    *   **Description:** Adapts messages between the AgentBridge canonical format and the specific command structures used by AutoGPT-like agents.
    *   **Example Usage:**
        ```python
        from openclaw_gateway.adapters.auto_gpt import AutoGPTAdapter
        adapter = AutoGPTAdapter()
        # ... use adapter methods ...
        ```

### `openclaw_gateway.MessageTranslationEngine` (Conceptual/Core Logic)

This module is the heart of the semantic routing and protocol adaptation. It utilizes AST transformation and bidirectional schema mapping tables to ensure accurate message conversion.

**Key Classes/Functions:**

*   **`MessageTranslationEngine` (Class)**
    *   **Signature:** `MessageTranslationEngine(schema_map: dict, ast_transformer: ASTTransformer)`
    *   **Description:** Manages the translation pipeline. It uses pre-defined bidirectional schema maps to guide AST transformations, converting complex, agent-specific message structures (like LangChain tool calls or vector context queries) into the canonical format and vice-versa.
    *   **Methods:**
        *   `translate_to_canonical(native_message: Any, source_adapter: BaseAdapter) -> CanonicalMessage`: Performs the translation from any native format to the standard.
        *   `translate_from_canonical(canonical_message: CanonicalMessage, target_adapter: BaseAdapter) -> Any`: Performs the translation from the standard format to the target agent's native format.
    *   **Example Usage:**
        ```python
        from openclaw_gateway.MessageTranslationEngine import MessageTranslationEngine
        # Assume schema_map and transformer are initialized
        engine = MessageTranslationEngine(schema_map, transformer)

        # Translate a complex LangChain structure to canonical
        canonical_msg = engine.translate_to_canonical(lc_input, langchain_adapter)

        # Translate canonical back to AutoGPT format
        auto_gpt_msg = engine.translate_from_canonical(canonical_msg, auto_gpt_adapter)
        ```

### `openclaw_gateway.cli`

This module provides the command-line interface for interacting with the gateway, useful for testing, debugging, and running simple routing tasks without a full service deployment.

**Key Classes/Functions:**

*   **`run_gateway_cli` (Function)**
    *   **Signature:** `run_gateway_cli(args: list)`
    *   **Description:** Entry point for the command-line interface. Allows users to simulate message routing, check adapter compatibility, or run diagnostic tests.
    *   **Example Usage:**
        ```bash
        # Example CLI invocation
        python -m openclaw_gateway.cli --route AgentA --target AgentB --message '{"type": "PING"}'
        ```

---

## System Overview Diagram (Conceptual Flow)

1.  **Agent $\rightarrow$ Gateway:** Agent sends native message $\rightarrow$ **Adapter** (`to_canonical`) $\rightarrow$ **`CanonicalMessage`**.
2.  **Gateway Core:** **`MessageTranslationEngine`** routes the `CanonicalMessage` based on routing rules.
3.  **Gateway $\rightarrow$ Agent:** **`CanonicalMessage`** $\rightarrow$ **`MessageTranslationEngine`** (`translate_from_canonical`) $\rightarrow$ **Adapter** (`from_canonical`) $\rightarrow$ Native Message $\rightarrow$ Agent.