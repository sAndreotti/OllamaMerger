"""Tests for HuggingFace URL/ID parser."""

import pytest

from ollama_merger.core.parser import ParsedModel, parse_model_id


class TestParseModelId:
    def test_full_https_url(self):
        result = parse_model_id("https://huggingface.co/mistralai/Mistral-7B-v0.1")
        assert result == ParsedModel(model_id="mistralai/Mistral-7B-v0.1")

    def test_url_with_trailing_slash(self):
        result = parse_model_id("https://huggingface.co/mistralai/Mistral-7B-v0.1/")
        assert result == ParsedModel(model_id="mistralai/Mistral-7B-v0.1")

    def test_url_with_revision(self):
        result = parse_model_id(
            "https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/v2"
        )
        assert result == ParsedModel(model_id="mistralai/Mistral-7B-v0.1", revision="v2")

    def test_url_without_scheme(self):
        result = parse_model_id("huggingface.co/mistralai/Mistral-7B-v0.1")
        assert result == ParsedModel(model_id="mistralai/Mistral-7B-v0.1")

    def test_http_url(self):
        result = parse_model_id("http://huggingface.co/meta-llama/Llama-2-7b")
        assert result == ParsedModel(model_id="meta-llama/Llama-2-7b")

    def test_direct_model_id(self):
        result = parse_model_id("mistralai/Mistral-7B-v0.1")
        assert result == ParsedModel(model_id="mistralai/Mistral-7B-v0.1")

    def test_whitespace_trimming(self):
        result = parse_model_id("  mistralai/Mistral-7B-v0.1  ")
        assert result == ParsedModel(model_id="mistralai/Mistral-7B-v0.1")

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError, match="Invalid model reference"):
            parse_model_id("")

    def test_invalid_single_segment(self):
        with pytest.raises(ValueError, match="Invalid model reference"):
            parse_model_id("mistralai")

    def test_invalid_random_url(self):
        with pytest.raises(ValueError, match="Invalid model reference"):
            parse_model_id("https://example.com/some/model")
