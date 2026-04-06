# openclaw-gateway

A lightweight agent-to-agent message translation gateway that eliminates manual protocol bridge code when autonomous agents must collaborate.

## Features

- **Protocol Translation**: Seamlessly translate between LangChain, AutoGPT, and event-streaming agents
- **Canonical Message Format**: Single unified format eliminates O(n²) translator explosion
- **Context Preservation**: Maintain conversation context through multi-hop agent delegations
- **Automatic Failure Recovery**: Rollback and reroute on agent failures
- **Zero-Loss Semantics**: Preserve message intent and critical fields across protocol boundaries
- **Distributed State Coordination**: Vector clocks + Last-Write-Wins for conflict resolution
- **Cycle Detection**: Automatically detect and reject circular delegation attempts
- **CLI & Programmatic API**: Use as a standalone service or embed in your application

## Installation

### From Source

```bash
git clone https://github.com/example/openclaw-gateway.git
cd openclaw-gateway
pip install -e .
```

### From PyPI

```bash
pip install openclaw-gateway
```

### Verify Installation

```bash
openclaw-gateway --version
```

## Quick Start: 3-Agent Setup

This example sets up three heterogeneous agents (LangChain, AutoGPT, spaCy) communicating through the gateway.

### Step 1: Install and Start Gateway

```bash
# Install
pip install openclaw-gateway

# Start gateway server on port 8080
openclaw-gateway start --port 8080 --log-level INFO
```

### Step 2: Register Agents

In another terminal, register the three agents:

```bash
# Register LangChain agent (HTTP on port 5000)
openclaw-gateway register \
  --agent-id "langchain_analyzer" \
  --protocol "langchain" \
  --endpoint "http://localhost:5000" \
  --capabilities '{"streaming": false, "task_decomposition": true}'

# Register AutoGPT agent (HTTP on port 5001)
openclaw-gateway register \
  --agent-id "autogpt_planner" \
  --protocol "autogpt" \
  --endpoint "http://localhost:5001" \
  --capabilities '{"streaming": false, "task_decomposition": true}'

# Register spaCy agent (WebSocket on port 5002)
openclaw-gateway register \
  --agent-id "spacy_extractor" \
  --protocol "event_stream" \
  --endpoint "ws://localhost:5002" \
  --capabilities '{"streaming": true, "task_decomposition": false}'
```

### Step 3: Submit a Workflow

Create a workflow file (`workflow.json`):

```json
{
  "conversation_id": "conv_123",
  "source_agent": "langchain_analyzer",
  "message": {
    "message_id": "msg_001",
    "source_agent_id": "langchain_analyzer",
    "conversation_id": "conv_123",
    "message_type": "request",
    "intent": "analyze",
    "payload": {
      "document": "research_paper.pdf",
      "format": "pdf"
    },
    "metadata": {
      "protocol_source": "langchain",
      "timestamp_utc": "2024-01-01T00:00:00Z"
    }
  }
}
```

Submit the workflow:

```bash
openclaw-gateway submit --workflow-file workflow.json --gateway-url http://localhost:8080
```

## Basic API Example

Use openclaw-gateway programmatically in your Python code:

```python
import asyncio
from openclaw_gateway import Gateway, LangChainAdapter, AutoGPTAdapter, EventStreamAdapter

async def main():
    # Create and start gateway
    gateway = Gateway(port=8080, context_limit=1000)

    # Register adapters for different agent types
    gateway.register_adapter("langchain", LangChainAdapter(
        endpoint="http://localhost:5000",
        protocol="langchain",
        features=["task_decomposition"]
    ))

    gateway.register_adapter("autogpt", AutoGPTAdapter(
        endpoint="http://localhost:5001",
        features=["task_decomposition", "streaming"]
    ))

    gateway.register_adapter("spacy", EventStreamAdapter(
        endpoint="ws://localhost:5002",
        features=["streaming"]
    ))

    # Start the gateway
    await gateway.start()

    # Submit a message from LangChain agent to AutoGPT agent
    from openclaw_gateway import CanonicalMessage

    message = CanonicalMessage(
        message_id="msg_001",
        source_agent_id="langchain_analyzer",
        conversation_id="conv_123",
        message_type="request",
        intent="delegate",
        payload={
            "document": "research_paper.pdf",
            "format": "pdf"
        },
        metadata={
            "protocol_source": "langchain",
            "timestamp_utc": "2024-01-01T00:00:00Z"
        }
    )

    # Route the message through the gateway
    response = await gateway.route_message(
        source_agent="langchain_analyzer",
        message=message,
        conversation_id="conv_123"
    )

    print(f"Response: {response}")

    # Subscribe to conversation updates
    async def on_state_change(context):
        print(f"Conversation {context.conversation_id} state updated")

    gateway.subscribe_conversation("conv_123", on_state_change)

    # Shutdown
    await gateway.stop()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## CLI Commands

### `openclaw-gateway --version`
Display the version of openclaw-gateway.

### `openclaw-gateway register`
Register a new agent with the gateway.

Options:
- `--agent-id`: Unique identifier for the agent
- `--protocol`: Protocol type (langchain, autogpt, event_stream)
- `--endpoint`: Agent endpoint URL
- `--config`: Path to configuration file (default: /etc/openclaw/config.yaml)
- `--capabilities`: JSON string of agent capabilities

### `openclaw-gateway start`
Start the openclaw-gateway server.

Options:
- `--port`: Port to listen on (default: 8080)
- `--config`: Path to configuration file
- `--log-level`: Logging level (default: INFO)

### `openclaw-gateway status`
Check the status of a conversation.

Options:
- `--conversation-id`: Conversation ID to check

### `openclaw-gateway submit`
Submit a workflow to the gateway.

Options:
- `--workflow-file`: Path to workflow file
- `--gateway-url`: Gateway URL (default: http://localhost:8080)

## Project Structure

```
openclaw-gateway/
├── openclaw_gateway/
│   ├── __init__.py              # Public API exports
│   ├── cli.py                   # CLI commands
│   ├── gateway.py               # Main Gateway orchestrator
│   ├── models.py                # Canonical message model
│   ├── context_manager.py       # Conversation context management
│   ├── router.py                # Message routing with cycle detection
│   ├── state_coordinator.py     # State coordination with vector clocks
│   ├── semantic_preserver.py    # Intent and field validation
│   ├── failure_recovery.py      # Rollback and reroute logic
│   ├── deduplication.py         # Message deduplication
│   └── adapters/
│       ├── __init__.py
│       ├── base.py              # Protocol adapter interface
│       ├── langchain.py         # LangChain protocol adapter
│       ├── autogpt.py           # AutoGPT protocol adapter
│       └── event_stream.py      # Event-streaming protocol adapter
├── tests/
│   ├── test_package.py          # Package and CLI tests
│   ├── adapters/                # Adapter tests
│   ├── integration/             # Integration tests
│   └── ...
├── setup.py                     # Package setup configuration
├── pyproject.toml               # Build system configuration
├── requirements.txt             # Dependency list
└── README.md                    # This file
```

## Development

### Install for Development

```bash
git clone https://github.com/example/openclaw-gateway.git
cd openclaw-gateway
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Format Code

```bash
black openclaw_gateway/ tests/
```

### Type Check

```bash
mypy openclaw_gateway/ --strict
```

## Performance Requirements

- **Message latency**: P95 < 500ms end-to-end
- **State coordination**: < 100ms per update
- **Context lookup**: < 10ms
- **Cycle detection**: < 50ms for 10-agent graph
- **Protocol negotiation**: < 50ms per handshake

## Architecture Highlights

### Single Canonical Format
Eliminates O(n²) translator explosion by using one intermediate format as a hub.

### Vector Clocks + Last-Write-Wins
Deterministically resolve concurrent state updates without a centralized clock.

### Write-Ahead Logging
Survive gateway crashes with zero message loss via append-only persistence.

### Tarjan's SCC Algorithm
Detect circular delegations in < 50ms for typical agent graphs.

### Operation Logs for Rollback
Roll back state changes byte-for-byte on agent failure, then reroute to backup.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md and submit pull requests to the main repository.
