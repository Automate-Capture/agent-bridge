"""CLI interface for openclaw-gateway.

This module provides the command-line interface for the openclaw-gateway package,
enabling users to register agents, start the gateway server, submit workflows,
and monitor conversation status.

Entry point: openclaw-gateway command
"""

import typer
import sys
from typing import Optional
from openclaw_gateway import __version__


def version_callback(value: bool) -> None:
    """Callback for --version flag."""
    if value:
        typer.echo(f"openclaw-gateway {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="openclaw-gateway: Agent-to-agent message translation gateway",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Main entry point for openclaw-gateway CLI."""
    pass


@app.command()
def register(
    agent_id: str = typer.Option(..., help="Unique identifier for the agent"),
    protocol: str = typer.Option(..., help="Protocol type (langchain, autogpt, event_stream)"),
    endpoint: str = typer.Option(..., help="Agent endpoint URL or WebSocket address"),
    config: Optional[str] = typer.Option(
        None, help="Path to configuration file"
    ),
) -> None:
    """Register an agent with the gateway.

    Example:
        openclaw-gateway register \\
            --agent-id langchain_analyzer \\
            --protocol langchain \\
            --endpoint http://localhost:5000 \\
            --config /etc/openclaw/config.yaml
    """
    typer.echo(f"Registering agent: {agent_id}")
    typer.echo(f"  Protocol: {protocol}")
    typer.echo(f"  Endpoint: {endpoint}")
    if config:
        typer.echo(f"  Config: {config}")
    typer.echo("Agent registration not yet implemented")


@app.command()
def start(
    port: int = typer.Option(8080, help="Port to run the gateway on"),
    config: Optional[str] = typer.Option(None, help="Path to configuration file"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
) -> None:
    """Start the gateway server.

    Example:
        openclaw-gateway start --port 8080 --config /etc/openclaw/config.yaml
    """
    typer.echo(f"Starting gateway on port {port}")
    if config:
        typer.echo(f"  Config: {config}")
    typer.echo(f"  Log level: {log_level}")
    typer.echo("Gateway startup not yet implemented")


@app.command()
def submit(
    workflow_file: str = typer.Option(..., help="Path to workflow JSON file"),
    gateway_url: Optional[str] = typer.Option(
        None, help="Gateway URL (default: http://localhost:8080)"
    ),
) -> None:
    """Submit a workflow to the gateway.

    Example:
        openclaw-gateway submit \\
            --workflow-file workflow.json \\
            --gateway-url http://localhost:8080
    """
    typer.echo(f"Submitting workflow from: {workflow_file}")
    gateway = gateway_url or "http://localhost:8080"
    typer.echo(f"  Gateway: {gateway}")
    typer.echo("Workflow submission not yet implemented")


@app.command()
def status(
    conversation_id: str = typer.Option(..., help="Conversation ID to check"),
    gateway_url: Optional[str] = typer.Option(
        None, help="Gateway URL (default: http://localhost:8080)"
    ),
) -> None:
    """Check the status of a conversation.

    Example:
        openclaw-gateway status --conversation-id conv_123
    """
    typer.echo(f"Status for conversation: {conversation_id}")
    gateway = gateway_url or "http://localhost:8080"
    typer.echo(f"  Gateway: {gateway}")
    typer.echo("Status check not yet implemented")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
