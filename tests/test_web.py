"""Tests for the web interface."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ollama_merger.web.app import app

client = TestClient(app)


class TestWebIndex:
    def test_index_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "OllamaMerger" in response.text
        assert "form" in response.text.lower()


class TestWebConvert:
    def test_convert_invalid_url(self):
        response = client.post("/convert", data={"url": "not-valid", "system_prompt": "", "revision": ""})
        assert response.status_code == 200
        assert "failed" in response.text.lower() or "error" in response.text.lower()

    @patch("ollama_merger.web.app.write_modelfile")
    @patch("ollama_merger.web.app.generate_modelfile")
    @patch("ollama_merger.web.app.download_model")
    def test_convert_success(self, mock_download, mock_generate, mock_write, tmp_path):
        mock_download.return_value = tmp_path / "model"
        mock_write.return_value = tmp_path / "model" / "Modelfile"
        mock_generate.return_value = "FROM .\n"

        response = client.post(
            "/convert",
            data={"url": "org/model", "system_prompt": "", "revision": ""},
        )
        assert response.status_code == 200
        assert "successfully" in response.text.lower()

    @patch("ollama_merger.web.app.download_model")
    def test_convert_download_error(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        response = client.post(
            "/convert",
            data={"url": "org/model", "system_prompt": "", "revision": ""},
        )
        assert response.status_code == 200
        assert "failed" in response.text.lower() or "error" in response.text.lower()


class TestWebCreate:
    @patch("ollama_merger.web.app.subprocess")
    def test_create_ollama_not_found(self, mock_subprocess):
        mock_subprocess.run.side_effect = FileNotFoundError()
        response = client.post(
            "/create",
            data={"model_dir": "/tmp/test", "model_name": "test-model"},
        )
        assert response.status_code == 200
        assert "not installed" in response.text.lower() or "not found" in response.text.lower()
