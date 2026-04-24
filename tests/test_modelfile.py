"""Tests for Modelfile generation."""

import json
import shutil
from pathlib import Path

import pytest

from ollama_merger.core.modelfile import generate_modelfile, write_modelfile

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def model_dir(tmp_path):
    """Create a model directory with fixture metadata files."""
    for filename in ["config.json", "tokenizer_config.json", "generation_config.json"]:
        shutil.copy(FIXTURES_DIR / filename, tmp_path / filename)
    return tmp_path


class TestGenerateModelfile:
    def test_starts_with_from(self, model_dir):
        content = generate_modelfile(model_dir)
        assert content.startswith("FROM .")

    def test_includes_template(self, model_dir):
        content = generate_modelfile(model_dir)
        assert "TEMPLATE" in content
        assert "<|assistant|>" in content

    def test_includes_stop_token(self, model_dir):
        content = generate_modelfile(model_dir)
        assert 'PARAMETER stop "</s>"' in content

    def test_includes_generation_params(self, model_dir):
        content = generate_modelfile(model_dir)
        assert "PARAMETER temperature 0.7" in content
        assert "PARAMETER top_p 0.9" in content
        assert "PARAMETER top_k 50" in content
        assert "PARAMETER repeat_penalty 1.1" in content
        assert "PARAMETER num_predict 2048" in content

    def test_includes_context_length(self, model_dir):
        content = generate_modelfile(model_dir)
        assert "PARAMETER num_ctx 4096" in content

    def test_includes_system_prompt(self, model_dir):
        content = generate_modelfile(model_dir, system_prompt="You are a helpful assistant.")
        assert 'SYSTEM """You are a helpful assistant."""' in content

    def test_no_system_prompt_by_default(self, model_dir):
        content = generate_modelfile(model_dir)
        assert "SYSTEM" not in content

    def test_minimal_model_dir(self, tmp_path):
        """With no metadata files, just FROM . is generated."""
        content = generate_modelfile(tmp_path)
        assert content.strip() == "FROM ."

    def test_missing_generation_config(self, tmp_path):
        """Works with only config.json and tokenizer_config.json."""
        shutil.copy(FIXTURES_DIR / "config.json", tmp_path / "config.json")
        shutil.copy(FIXTURES_DIR / "tokenizer_config.json", tmp_path / "tokenizer_config.json")
        content = generate_modelfile(tmp_path)
        assert "TEMPLATE" in content
        assert "PARAMETER num_ctx 4096" in content
        assert "PARAMETER temperature" not in content

    def test_list_chat_template(self, tmp_path):
        """Handles chat_template as a list of templates."""
        config = {
            "eos_token": "</s>",
            "chat_template": [
                {"name": "default", "template": "{{ message }}"},
                {"name": "tool_use", "template": "{{ tool }}"},
            ],
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
        content = generate_modelfile(tmp_path)
        assert "{{ message }}" in content


class TestWriteModelfile:
    def test_writes_modelfile(self, model_dir):
        path = write_modelfile(model_dir)
        assert path == model_dir / "Modelfile"
        assert path.exists()
        content = path.read_text()
        assert content.startswith("FROM .")
