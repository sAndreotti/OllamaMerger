"""Tests for CLI interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ollama_merger.cli import _format_size, app

runner = CliRunner()


class TestConvertCommand:
    @patch("ollama_merger.cli.write_modelfile")
    @patch("ollama_merger.cli.download_model")
    def test_convert_basic(self, mock_download, mock_write, tmp_path):
        mock_download.return_value = tmp_path / "Mistral-7B-v0.1"
        mock_write.return_value = tmp_path / "Mistral-7B-v0.1" / "Modelfile"

        result = runner.invoke(
            app, ["convert", "mistralai/Mistral-7B-v0.1", "-o", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Mistral-7B-v0.1" in result.output
        mock_download.assert_called_once()

    @patch("ollama_merger.cli.write_modelfile")
    @patch("ollama_merger.cli.download_model")
    def test_convert_with_revision(self, mock_download, mock_write, tmp_path):
        mock_download.return_value = tmp_path / "model"
        mock_write.return_value = tmp_path / "model" / "Modelfile"

        result = runner.invoke(
            app,
            ["convert", "mistralai/Mistral-7B-v0.1", "-r", "v2", "-o", str(tmp_path)],
        )
        assert result.exit_code == 0
        mock_download.assert_called_once_with(
            "mistralai/Mistral-7B-v0.1", tmp_path, revision="v2"
        )

    @patch("ollama_merger.cli.write_modelfile")
    @patch("ollama_merger.cli.download_model")
    def test_convert_with_system_prompt(self, mock_download, mock_write, tmp_path):
        mock_download.return_value = tmp_path / "model"
        mock_write.return_value = tmp_path / "model" / "Modelfile"

        result = runner.invoke(
            app,
            [
                "convert",
                "mistralai/Mistral-7B-v0.1",
                "-s",
                "You are helpful.",
                "-o",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        mock_write.assert_called_once_with(tmp_path / "model", system_prompt="You are helpful.")

    def test_convert_invalid_url(self):
        result = runner.invoke(app, ["convert", "not-valid"])
        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("ollama_merger.cli.download_model")
    def test_convert_download_failure(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        result = runner.invoke(app, ["convert", "org/model"])
        assert result.exit_code == 1
        assert "Download failed" in result.output


class TestConvertWithPush:
    @patch("ollama_merger.cli.create_model")
    @patch("ollama_merger.cli.write_modelfile")
    @patch("ollama_merger.cli.download_model")
    def test_convert_with_create(self, mock_download, mock_write, mock_create, tmp_path):
        from ollama_merger.core.ollama import OllamaResult

        mock_download.return_value = tmp_path / "model"
        mock_write.return_value = tmp_path / "model" / "Modelfile"
        mock_create.return_value = OllamaResult(success=True, output="success")

        result = runner.invoke(app, ["convert", "org/model", "-o", str(tmp_path), "--create"])
        assert result.exit_code == 0
        assert "created successfully" in result.output
        mock_create.assert_called_once_with("model", tmp_path / "model")

    @patch("ollama_merger.cli.push_model")
    @patch("ollama_merger.cli.create_model")
    @patch("ollama_merger.cli.write_modelfile")
    @patch("ollama_merger.cli.download_model")
    def test_convert_with_push(self, mock_download, mock_write, mock_create, mock_push, tmp_path):
        from ollama_merger.core.ollama import OllamaResult

        mock_download.return_value = tmp_path / "model"
        mock_write.return_value = tmp_path / "model" / "Modelfile"
        mock_create.return_value = OllamaResult(success=True, output="")
        mock_push.return_value = OllamaResult(success=True, output="")

        result = runner.invoke(
            app,
            ["convert", "org/model", "-o", str(tmp_path), "--push", "--name", "user/my-model"],
        )
        assert result.exit_code == 0
        assert "pushed successfully" in result.output

    @patch("ollama_merger.cli.write_modelfile")
    @patch("ollama_merger.cli.download_model")
    def test_convert_push_requires_username(self, mock_download, mock_write, tmp_path):
        from ollama_merger.core.ollama import OllamaResult

        mock_download.return_value = tmp_path / "model"
        mock_write.return_value = tmp_path / "model" / "Modelfile"

        with patch("ollama_merger.cli.create_model") as mock_create:
            mock_create.return_value = OllamaResult(success=True, output="")
            result = runner.invoke(app, ["convert", "org/model", "-o", str(tmp_path), "--push"])

        assert result.exit_code == 1
        assert "username" in result.output.lower()


class TestPushCommand:
    @patch("ollama_merger.cli.push_model")
    def test_push_success(self, mock_push):
        from ollama_merger.core.ollama import OllamaResult

        mock_push.return_value = OllamaResult(success=True, output="done")
        result = runner.invoke(app, ["push-cmd", "user/my-model"])
        assert result.exit_code == 0
        assert "pushed successfully" in result.output

    def test_push_requires_username(self):
        result = runner.invoke(app, ["push-cmd", "my-model"])
        assert result.exit_code == 1
        assert "username" in result.output.lower()

    @patch("ollama_merger.cli.push_model")
    def test_push_failure(self, mock_push):
        from ollama_merger.core.ollama import OllamaResult

        mock_push.return_value = OllamaResult(success=False, output="unauthorized")
        result = runner.invoke(app, ["push-cmd", "user/my-model"])
        assert result.exit_code == 1
        assert "Push failed" in result.output


class TestListCommand:
    def test_list_invalid_url(self):
        result = runner.invoke(app, ["list", "not-valid"])
        assert result.exit_code == 1


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_size(3 * 1024**3) == "3.0 GB"
