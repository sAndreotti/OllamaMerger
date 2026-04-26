"""Interact with the local Ollama instance."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OllamaResult:
    """Result of an Ollama command."""

    success: bool
    output: str


def create_model(model_name: str, model_dir: Path) -> OllamaResult:
    """Run `ollama create` to import a model from a Modelfile.

    Args:
        model_name: Name for the Ollama model (e.g. "username/my-model").
        model_dir: Directory containing the Modelfile and model weights.
    """
    return _run_ollama(["ollama", "create", model_name, "-f", "Modelfile"], cwd=model_dir)


def push_model(model_name: str) -> OllamaResult:
    """Run `ollama push` to upload a model to the Ollama registry.

    The model name must include your Ollama username (e.g. "username/my-model").
    You must be logged in via `ollama login` before pushing.

    Args:
        model_name: Full model name with username (e.g. "username/my-model").
    """
    return _run_ollama(["ollama", "push", model_name])


def _run_ollama(cmd: list[str], cwd: Path | None = None) -> OllamaResult:
    """Execute an Ollama CLI command."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return OllamaResult(
            success=result.returncode == 0,
            output=result.stdout if result.returncode == 0 else result.stderr,
        )
    except FileNotFoundError:
        return OllamaResult(
            success=False,
            output="Ollama is not installed or not found in PATH. Install from https://ollama.com",
        )
    except subprocess.TimeoutExpired:
        return OllamaResult(
            success=False,
            output="Command timed out (10 min limit).",
        )
