<p align="center">
  <img src="assets/hero.jpg" alt="AgentBridge" width="900">
</p>

<h1 align="center">AgentBridge</h1>

<p align="center">
  <strong>Autonomous gateway for dynamic protocol adaptation and semantic routing between AI agents.</strong>
</p>

<p align="center">
  <a href="https://github.com/Lumi-node/agent-bridge"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License Badge"></a>
  <a href="https://github.com/Lumi-node/agent-bridge"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version Badge"></a>
  <a href="https://github.com/Lumi-node/agent-bridge"><img src="https://img.shields.io/badge/Tests-84%20files-green.svg" alt="Test Count Badge"></a>
</p>

---

AgentBridge is a production-grade autonomous gateway designed to solve the critical interoperability challenge in heterogeneous AI ecosystems. It enables seamless, dynamic protocol adaptation and semantic message routing between diverse AI agent architectures.

This system implements a robust MessageTranslationEngine utilizing Abstract Syntax Tree transformation and bidirectional schema mapping. It converts agent-specific formats (like LangChain tool calls or AutoGPT structures) into a canonical intermediate representation, ensuring agents can communicate regardless of their native protocol.

---

## Quick Start

First, install the package via pip:

```bash
pip install agent_bridge
```

Then, initialize the gateway and process a message:

```python
from openclaw_gateway.canonical_message import CanonicalMessage
from openclaw_gateway.adapters.langchain import LangChainAdapter
from openclaw_gateway.cli import AgentBridgeCLI

# Initialize the adapter for a specific agent type
langchain_adapter = LangChainAdapter()

# Create a canonical message object
canonical_msg = CanonicalMessage(
    source_agent="LangChain",
    payload={"action": "query", "parameters": {"topic": "weather"}}
)

# Translate the canonical message to the target agent's format (e.g., AutoGPT)
target_format_message = langchain_adapter.translate_to_target(canonical_msg)

print(f"Translated Message: {target_format_message}")
```

## What Can You Do?

### Dynamic Protocol Adaptation
AgentBridge handles the complex translation layer between disparate agent communication standards. It uses recursive descent parsing and type-safe serialization to map complex structures reliably.

```python
# Example of schema mapping in action
from openclaw_gateway.adapters.base import BaseAdapter

class CustomAgentAdapter(BaseAdapter):
    def translate_to_target(self, canonical_msg: CanonicalMessage) -> dict:
        # Custom logic to map canonical structure to proprietary format
        return {"custom_field": canonical_msg.payload.get("action")}
```

### Semantic Routing
Messages are routed not just based on destination, but on their *meaning*. The canonical representation allows the gateway to inspect the intent of a message before forwarding it to the most appropriate downstream agent.

```python
# Routing decision based on message content
if canonical_msg.payload.get("action") == "query":
    print("Routing to Information Retrieval Agent.")
elif canonical_msg.payload.get("action") == "execute":
    print("Routing to Action Executor Agent.")
```

## Architecture

The system is structured around a central **Canonical Message** format. Adapters (e.g., `LangChainAdapter`) sit at the edges, responsible for translating between the external agent protocol and the internal canonical format. The core logic resides in the `MessageTranslationEngine`, which manages the bidirectional schema mapping tables.

```mermaid
graph TD
    A[External Agent 1 (LangChain)] -->|Protocol A| B(Adapter Layer);
    C[External Agent 2 (AutoGPT)] -->|Protocol B| B;
    B --> D{MessageTranslationEngine};
    D --> E[Canonical Message Representation];
    E --> F{Semantic Router};
    F --> G[Target Agent];
```

## API Reference

**`openclaw_gateway.canonical_message.CanonicalMessage`**
Represents the standardized, intermediate message format.
*Signature:* `CanonicalMessage(source_agent: str, payload: dict)`
*Example:* `CanonicalMessage("LangChain", {"action": "query", "parameters": {"topic": "weather"}})`

**`openclaw_gateway.adapters.base.BaseAdapter`**
Abstract base class defining the interface for all protocol translators.
*Methods:* `translate_to_target(canonical_msg: CanonicalMessage) -> Any`, `translate_from_source(target_msg: Any) -> CanonicalMessage`

**`openclaw_gateway.cli.AgentBridgeCLI`**
Command-Line Interface for running the gateway daemon.
*Usage:* `agent_bridge run --config /path/to/config.yaml`

## Research Background

This work draws inspiration from research in distributed systems and heterogeneous computing, specifically concerning middleware design for complex, evolving agentic workflows. The concept of a canonical message bus is rooted in established patterns for microservice communication, adapted here for the unique challenges of AI agent state and protocol variance.

## Testing

The project maintains comprehensive test coverage, with **84 test files** ensuring the stability of the translation and routing logic across various adapter implementations.

## Contributing

We welcome contributions! Please refer to the contribution guidelines in the repository for details on submitting pull requests, reporting bugs, and suggesting features.

## Citation

This project is inspired by foundational work in multi-agent systems and protocol negotiation. Further reading on agent interoperability is recommended.

## License
The AgentBridge project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.