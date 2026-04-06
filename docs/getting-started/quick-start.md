# 🚀 AgentBridge Quick Start Guide

AgentBridge is a powerful framework designed to build production-grade, autonomous agent gateway systems. It enables seamless, dynamic protocol adaptation and semantic message routing between diverse AI agent architectures, handling distributed state coordination and robust failure recovery.

This guide will get you up and running with the core components, focusing on the `openclaw_gateway` module.

## Prerequisites

Before you begin, ensure you have Python (3.8+) installed.

## Installation

Since AgentBridge is distributed via a package structure, the recommended way to install it for development or use is via pip, assuming you are in the root directory containing `setup.py`.

```bash
# Install the package in editable mode for development
pip install -e .
```

## Core Concepts

The heart of AgentBridge lies in its ability to act as a universal translator and router.

1.  **Canonical Message:** All communication flows through the `CanonicalMessage` format (`openclaw_gateway.canonical_message`). This standardized structure decouples the gateway from the specific protocols of the connected agents.
2.  **Adapters:** Specific implementations handle the translation between the canonical format and agent-specific formats (e.g., LangChain, AutoGPT). The `LangChainAdapter` (`openclaw_gateway.adapters.langchain`) is the primary implementation for LangChain integration.
3.  **Message Translation Engine:** This engine (conceptually implemented via AST transformation and schema mapping) ensures that a request formatted for Agent A can be correctly interpreted and executed by Agent B, regardless of their native syntax.

## 🛠️ Usage Examples

Here are a few examples demonstrating how to utilize the gateway components.

### Example 1: Basic LangChain Message Translation

This example shows how to initialize the LangChain adapter and translate a canonical request into a format suitable for a LangChain agent.

```python
from openclaw_gateway.canonical_message import CanonicalMessage
from openclaw_gateway.adapters.langchain import LangChainAdapter

# 1. Initialize the Adapter
langchain_adapter = LangChainAdapter()

# 2. Define a Canonical Request (e.g., a request to execute a specific tool)
canonical_request = CanonicalMessage(
    message_type="TOOL_CALL",
    payload={
        "tool_name": "weather_lookup",
        "parameters": {"city": "San Francisco"}
    }
)

# 3. Translate the message to the LangChain format
try:
    langchain_payload = langchain_adapter.to_agent_format(canonical_request)
    print("--- Successfully Translated to LangChain Format ---")
    print(langchain_payload)
except Exception as e:
    print(f"Translation Error: {e}")
```

### Example 2: Routing and Handling Errors

This demonstrates how the gateway structure handles routing and potential failures using the defined error mechanisms.

```python
from openclaw_gateway.canonical_message import CanonicalMessage
from openclaw_gateway.errors import AgentGatewayError

# Assume we have a routing mechanism that directs the message
def route_message(message: CanonicalMessage):
    if message.message_type == "STATE_UPDATE":
        print("Routing to State Coordinator...")
        # In a real system, this would call a state manager
        return True
    elif message.message_type == "INVALID_COMMAND":
        # Simulate a failure scenario
        raise AgentGatewayError("Protocol mismatch detected during routing.")
    else:
        print(f"Routing standard message type: {message.message_type}")
        return True

# Test Case 1: Successful Routing
success_msg = CanonicalMessage(message_type="STATE_UPDATE", payload={"key": "user_session", "value": "active"})
try:
    route_message(success_msg)
except AgentGatewayError as e:
    print(f"Caught expected error: {e}")

print("-" * 20)

# Test Case 2: Failure Handling
failure_msg = CanonicalMessage(message_type="INVALID_COMMAND", payload={})
try:
    route_message(failure_msg)
except AgentGatewayError as e:
    print(f"✅ Successfully caught gateway error: {e}")
```

### Example 3: Command Line Interface (CLI) Interaction

For quick testing or integration into a larger system, AgentBridge provides a CLI entry point (`openclaw_gateway.cli`).

*(Note: This requires running the script from the command line, not directly in a Python interpreter.)*

**To run this example, save the following as `run_gateway.py` and execute it:**

```python
# run_gateway.py (Conceptual usage via CLI)
# This simulates invoking the CLI functionality defined in openclaw_gateway.cli
# In a real scenario, you would run: python -m openclaw_gateway.cli --command process --input "..."

import argparse
from openclaw_gateway.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentBridge Gateway CLI")
    parser.add_argument("--command", required=True, choices=['process', 'status'])
    parser.add_argument("--input", help="Input message payload for processing")
    args = parser.parse_args()
    
    # The main function handles parsing and execution based on the command
    main(args)
```

**Execution:**

```bash
# Simulate processing an input message
python run_gateway.py --command process --input '{"type": "QUERY", "data": "What is the weather?"}'
```

## 📚 Module Reference

| Module/Class | Purpose | Key Functionality |
| :--- | :--- | :--- |
| `openclaw_gateway.canonical_message` | Defines the universal data structure for all inter-agent communication. | `CanonicalMessage` class |
| `openclaw_gateway.adapters.langchain` | Implements the translation logic specifically for LangChain agent inputs/outputs. | `LangChainAdapter.to_agent_format()` |
| `openclaw_gateway.errors` | Provides custom exceptions for predictable failure handling across the gateway. | `AgentGatewayError` |
| `openclaw_gateway.cli` | Provides a command-line interface for testing and simple gateway operations. | `main()` function |