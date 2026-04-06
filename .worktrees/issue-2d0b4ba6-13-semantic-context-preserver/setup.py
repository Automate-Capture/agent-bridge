"""
Setup configuration for openclaw-gateway package.

This script defines package metadata, dependencies, and entry points for
the openclaw-gateway distribution.
"""

from setuptools import setup, find_packages
import re

# Read version from __init__.py without importing the package
with open("openclaw_gateway/__init__.py", "r") as f:
    version_match = re.search(r'^__version__ = ["\']([^"\']+)["\']', f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string")

setup(
    name="openclaw-gateway",
    version=version,
    description="Lightweight agent-to-agent message translation gateway",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Research Radar Team",
    author_email="team@example.com",
    url="https://github.com/example/openclaw-gateway",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "typer>=0.9.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "openclaw-gateway=openclaw_gateway.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
