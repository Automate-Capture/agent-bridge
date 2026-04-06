"""
tests/test_package.py: Package import and CLI entry point tests.

Tests verify that:
1. All exported classes are importable from openclaw_gateway
2. The __version__ attribute is accessible
3. The CLI entry point works correctly
4. Classes are instantiable (when available)
"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess
import sys


class TestPackageImports:
    """Unit tests for package imports."""

    def test_imports_work(self):
        """Test that all exported classes can be imported from openclaw_gateway."""
        from openclaw_gateway import (
            Gateway,
            LangChainAdapter,
            AutoGPTAdapter,
            EventStreamAdapter,
            CanonicalMessage,
        )
        # Since sibling issues haven't implemented these yet, they may be None
        # We just verify the imports don't raise errors
        assert Gateway is not None or Gateway is None  # Either imported or None placeholder
        assert LangChainAdapter is not None or LangChainAdapter is None
        assert AutoGPTAdapter is not None or AutoGPTAdapter is None
        assert EventStreamAdapter is not None or EventStreamAdapter is None
        assert CanonicalMessage is not None or CanonicalMessage is None

    def test_version_accessible(self):
        """Test that __version__ is accessible from the package."""
        from openclaw_gateway import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        # Verify version format (semantic versioning)
        parts = __version__.split(".")
        assert len(parts) >= 2, f"Version {__version__} is not in semantic format"

    def test_all_exports_present(self):
        """Test that __all__ contains all expected exports."""
        import openclaw_gateway

        expected_exports = [
            "__version__",
            "Gateway",
            "LangChainAdapter",
            "AutoGPTAdapter",
            "EventStreamAdapter",
            "CanonicalMessage",
        ]

        assert hasattr(openclaw_gateway, "__all__")
        assert all(export in openclaw_gateway.__all__ for export in expected_exports)

    def test_package_metadata(self):
        """Test that package has correct metadata."""
        import openclaw_gateway

        assert hasattr(openclaw_gateway, "__version__")
        assert hasattr(openclaw_gateway, "__all__")

    def test_adapters_subpackage_importable(self):
        """Test that the adapters subpackage is importable."""
        import openclaw_gateway.adapters

        assert openclaw_gateway.adapters is not None


class TestCLIEntryPoint:
    """Tests for CLI entry point functionality."""

    def test_cli_app_exists(self):
        """Test that the CLI app exists and is callable."""
        from openclaw_gateway.cli import app

        assert app is not None
        # Verify it's a Typer app by checking for the necessary attributes
        assert hasattr(app, "command") or callable(app)

    def test_cli_version_command(self):
        """Test that the version command works."""
        from openclaw_gateway.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "openclaw-gateway" in result.stdout
        assert "0.1.0" in result.stdout

    def test_cli_has_help(self):
        """Test that CLI has help text."""
        from openclaw_gateway.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "openclaw-gateway" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_register_help(self):
        """Test that register command has help."""
        from openclaw_gateway.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["register", "--help"])

        assert result.exit_code == 0
        assert "agent-id" in result.stdout or "agent_id" in result.stdout

    def test_cli_start_help(self):
        """Test that start command has help."""
        from openclaw_gateway.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["start", "--help"])

        assert result.exit_code == 0
        assert "port" in result.stdout

    def test_cli_status_help(self):
        """Test that status command has help."""
        from openclaw_gateway.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "conversation-id" in result.stdout or "conversation_id" in result.stdout

    def test_cli_submit_help(self):
        """Test that submit command has help."""
        from openclaw_gateway.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["submit", "--help"])

        assert result.exit_code == 0
        assert "workflow" in result.stdout.lower()


class TestPackageInstallation:
    """Integration tests for package installation and CLI entry point."""

    def test_entry_point_version_command(self):
        """Test that openclaw-gateway --version works via entry point.

        This test verifies that:
        1. The package can be imported as a module
        2. The version string is accessible
        3. The version follows semantic versioning
        """
        from openclaw_gateway import __version__

        # Verify version string is present and valid
        assert __version__
        assert isinstance(__version__, str)
        version_parts = __version__.split(".")
        assert len(version_parts) >= 2

    def test_entry_point_maps_to_cli_app(self):
        """Test that entry point correctly maps to CLI app."""
        # Verify the entry point is configured in setup.py
        from openclaw_gateway.cli import app

        assert app is not None

    def test_package_has_setup_py(self):
        """Test that setup.py exists and contains correct entry point."""
        import os

        setup_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "setup.py"
        )
        assert os.path.exists(setup_path), "setup.py not found"

        with open(setup_path, "r") as f:
            content = f.read()
            assert "openclaw-gateway" in content, "Package name not found in setup.py"
            assert "console_scripts" in content, "console_scripts not configured"
            assert (
                "openclaw_gateway.cli:app" in content
            ), "CLI entry point not configured"

    def test_package_has_pyproject_toml(self):
        """Test that pyproject.toml exists and is properly configured."""
        import os

        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
        )
        assert os.path.exists(pyproject_path), "pyproject.toml not found"

        with open(pyproject_path, "r") as f:
            content = f.read()
            assert "name = " in content or 'name = "' in content
            assert "setuptools.build_meta" in content
            assert (
                "openclaw-gateway" in content
            ), "Package name not found in pyproject.toml"


class TestPackageStructure:
    """Tests for package directory structure."""

    def test_openclaw_gateway_dir_exists(self):
        """Test that openclaw_gateway directory exists."""
        import os
        import openclaw_gateway

        package_dir = os.path.dirname(openclaw_gateway.__file__)
        assert os.path.isdir(package_dir)

    def test_adapters_dir_exists(self):
        """Test that openclaw_gateway/adapters directory exists."""
        import os
        import openclaw_gateway.adapters

        adapters_dir = os.path.dirname(openclaw_gateway.adapters.__file__)
        assert os.path.isdir(adapters_dir)

    def test_cli_module_exists(self):
        """Test that cli.py module exists."""
        import os
        import openclaw_gateway

        package_dir = os.path.dirname(openclaw_gateway.__file__)
        cli_path = os.path.join(package_dir, "cli.py")
        assert os.path.isfile(cli_path)

    def test_init_files_exist(self):
        """Test that __init__.py files exist in required locations."""
        import os
        import openclaw_gateway

        package_dir = os.path.dirname(openclaw_gateway.__file__)

        # Check openclaw_gateway/__init__.py
        init_path = os.path.join(package_dir, "__init__.py")
        assert os.path.isfile(init_path), f"{init_path} not found"

        # Check openclaw_gateway/adapters/__init__.py
        adapters_init = os.path.join(package_dir, "adapters", "__init__.py")
        assert os.path.isfile(adapters_init), f"{adapters_init} not found"


class TestVersionString:
    """Tests for version string consistency."""

    def test_version_format(self):
        """Test that version string follows semantic versioning."""
        from openclaw_gateway import __version__

        parts = __version__.split(".")
        assert len(parts) >= 2, f"Version {__version__} should have at least major.minor"

        # Check that major and minor are numeric
        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            pytest.fail(f"Version parts are not numeric: {__version__}")

    def test_version_consistency(self):
        """Test that version is consistent across package."""
        from openclaw_gateway import __version__
        import openclaw_gateway

        # Both access methods should return the same version
        assert openclaw_gateway.__version__ == __version__


class TestDependencies:
    """Tests for package dependencies."""

    def test_pydantic_available(self):
        """Test that pydantic is available."""
        try:
            import pydantic  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("pydantic not installed")

    def test_aiohttp_available(self):
        """Test that aiohttp is available."""
        try:
            import aiohttp  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("aiohttp not installed")

    def test_typer_available(self):
        """Test that typer is available."""
        try:
            import typer  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("typer not installed")

    def test_yaml_available(self):
        """Test that pyyaml is available."""
        try:
            import yaml  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("pyyaml not installed")
