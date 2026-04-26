"""Tests for the web interface."""

from __future__ import annotations

from unittest.mock import patch

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
    @patch("ollama_merger.web.app.create_model")
    def test_create_ollama_not_found(self, mock_create):
        from ollama_merger.core.ollama import OllamaResult

        mock_create.return_value = OllamaResult(
            success=False, output="Ollama is not installed or not found in PATH."
        )
        response = client.post(
            "/create",
            data={"model_dir": "/tmp/test", "model_name": "test-model"},
        )
        assert response.status_code == 200
        assert "not installed" in response.text.lower() or "not found" in response.text.lower()


class TestWebPush:
    @patch("ollama_merger.web.app.push_model")
    def test_push_success(self, mock_push):
        from ollama_merger.core.ollama import OllamaResult

        mock_push.return_value = OllamaResult(success=True, output="pushing manifest")
        response = client.post("/push", data={"model_name": "user/my-model"})
        assert response.status_code == 200
        assert "successfully" in response.text.lower()

    def test_push_requires_username(self):
        response = client.post("/push", data={"model_name": "my-model"})
        assert response.status_code == 200
        assert "username" in response.text.lower()

    @patch("ollama_merger.web.app.push_model")
    def test_push_failure(self, mock_push):
        from ollama_merger.core.ollama import OllamaResult

        mock_push.return_value = OllamaResult(success=False, output="unauthorized")
        response = client.post("/push", data={"model_name": "user/my-model"})
        assert response.status_code == 200
        assert "unauthorized" in response.text
