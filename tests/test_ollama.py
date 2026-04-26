"""Tests for Ollama CLI integration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from ollama_merger.core.ollama import create_model, push_model


class TestCreateModel:
    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_create_success(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="success\n", stderr=""
        )
        result = create_model("my-model", tmp_path)
        assert result.success is True
        assert "success" in result.output
        mock_run.assert_called_once_with(
            ["ollama", "create", "my-model", "-f", "Modelfile"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
        )

    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_create_failure(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error: bad modelfile"
        )
        result = create_model("my-model", tmp_path)
        assert result.success is False
        assert "bad modelfile" in result.output

    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_create_ollama_not_found(self, mock_run, tmp_path):
        mock_run.side_effect = FileNotFoundError()
        result = create_model("my-model", tmp_path)
        assert result.success is False
        assert "not installed" in result.output

    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_create_timeout(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=600)
        result = create_model("my-model", tmp_path)
        assert result.success is False
        assert "timed out" in result.output


class TestPushModel:
    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_push_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="pushing manifest\n", stderr=""
        )
        result = push_model("username/my-model")
        assert result.success is True
        mock_run.assert_called_once_with(
            ["ollama", "push", "username/my-model"],
            cwd=None,
            capture_output=True,
            text=True,
            timeout=600,
        )

    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_push_failure(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="unauthorized"
        )
        result = push_model("username/my-model")
        assert result.success is False
        assert "unauthorized" in result.output

    @patch("ollama_merger.core.ollama.subprocess.run")
    def test_push_ollama_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = push_model("username/my-model")
        assert result.success is False
        assert "not installed" in result.output
