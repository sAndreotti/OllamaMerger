"""Tests for HuggingFace model downloader."""

from pathlib import Path
from unittest.mock import patch

from ollama_merger.core.downloader import ALLOW_PATTERNS, download_model


class TestDownloadModel:
    @patch("ollama_merger.core.downloader.snapshot_download")
    def test_download_creates_correct_subdir(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path / "Mistral-7B-v0.1")

        result = download_model("mistralai/Mistral-7B-v0.1", tmp_path)

        assert result == tmp_path / "Mistral-7B-v0.1"

    @patch.dict("os.environ", {}, clear=True)
    @patch("ollama_merger.core.downloader.snapshot_download")
    def test_download_passes_correct_patterns(self, mock_download, tmp_path):
        download_model("mistralai/Mistral-7B-v0.1", tmp_path)

        mock_download.assert_called_once_with(
            repo_id="mistralai/Mistral-7B-v0.1",
            local_dir=str(tmp_path / "Mistral-7B-v0.1"),
            revision=None,
            allow_patterns=ALLOW_PATTERNS,
            token=None,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("ollama_merger.core.downloader.snapshot_download")
    def test_download_with_revision(self, mock_download, tmp_path):
        download_model("mistralai/Mistral-7B-v0.1", tmp_path, revision="v2")

        mock_download.assert_called_once_with(
            repo_id="mistralai/Mistral-7B-v0.1",
            local_dir=str(tmp_path / "Mistral-7B-v0.1"),
            revision="v2",
            allow_patterns=ALLOW_PATTERNS,
            token=None,
        )

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    @patch("ollama_merger.core.downloader.snapshot_download")
    def test_download_passes_hf_token(self, mock_download, tmp_path):
        download_model("org/model", tmp_path)

        mock_download.assert_called_once_with(
            repo_id="org/model",
            local_dir=str(tmp_path / "model"),
            revision=None,
            allow_patterns=ALLOW_PATTERNS,
            token="hf_test_token_123",
        )

    def test_allow_patterns_includes_safetensors(self):
        assert "*.safetensors" in ALLOW_PATTERNS

    def test_allow_patterns_includes_metadata(self):
        assert "config.json" in ALLOW_PATTERNS
        assert "tokenizer_config.json" in ALLOW_PATTERNS
        assert "tokenizer.json" in ALLOW_PATTERNS
