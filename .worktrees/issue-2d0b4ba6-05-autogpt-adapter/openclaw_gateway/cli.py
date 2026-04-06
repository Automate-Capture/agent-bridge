"""
openclaw_gateway.cli: Command-line interface for openclaw-gateway.

Provides CLI commands for registering agents, starting the gateway, submitting workflows,
and monitoring conversation status.
"""

import typer
from typing import Optional
from openclaw_gateway import __version__


def version_callback(value: bool) -> None:
    """Handle --version flag."""
    if value:
        typer.echo(f"openclaw-gateway {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="openclaw-gateway: Lightweight agent-to-agent message translation gateway"
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Main callback for CLI."""
    pass


@app.command()
def version() -> None:
    """Display the version of openclaw-gateway."""
    typer.echo(f"openclaw-gateway {__version__}")


@app.command()
def register(
    agent_id: str = typer.Option(..., help="Unique identifier for the agent"),
    protocol: str = typer.Option(..., help="Protocol type (langchain, autogpt, event_stream)"),
    endpoint: str = typer.Option(..., help="Agent endpoint URL"),
    config: str = typer.Option(
        "/etc/openclaw/config.yaml", help="Path to configuration file"
    ),
    capabilities: Optional[str] = typer.Option(
        None, help="JSON string of agent capabilities"
    ),
) -> None:
    """Register a new agent with the gateway."""
    typer.echo(
        f"Registering agent {agent_id} with protocol {protocol} at {endpoint}"
    )
    # TODO: Implement agent registration logic (will be in issue-15)


@app.command()
def start(
    port: int = typer.Option(8080, help="Port to listen on"),
    config: str = typer.Option(
        "/etc/openclaw/config.yaml", help="Path to configuration file"
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    """Start the openclaw-gateway server."""
    typer.echo(f"Starting gateway on port {port} with log level {log_level}")
    # TODO: Implement server startup logic (will be in issue-15)


@app.command()
def status(
    conversation_id: str = typer.Option(..., help="Conversation ID to check"),
) -> None:
    """Check the status of a conversation."""
    typer.echo(f"Checking status of conversation {conversation_id}")
    # TODO: Implement status checking logic (will be in issue-15)


@app.command()
def submit(
    workflow_file: str = typer.Option(..., help="Path to workflow file"),
    gateway_url: str = typer.Option(
        "http://localhost:8080", help="Gateway URL"
    ),
) -> None:
    """Submit a workflow to the gateway."""
    typer.echo(f"Submitting workflow from {workflow_file} to {gateway_url}")
    # TODO: Implement workflow submission logic (will be in issue-15)


if __name__ == "__main__":
    app()
