"""Download models from HuggingFace Hub."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

ALLOW_PATTERNS = [
    "*.safetensors",
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "generation_config.json",
    "special_tokens_map.json",
]


def download_model(
    model_id: str,
    output_dir: Path,
    revision: str | None = None,
) -> Path:
    """Download a model's SafeTensor files and metadata from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g. "mistralai/Mistral-7B-v0.1").
        output_dir: Directory where the model files will be downloaded.
        revision: Optional git revision (branch, tag, or commit hash).

    Returns:
        Path to the directory containing the downloaded model files.
    """
    token = os.environ.get("HF_TOKEN")

    local_dir = output_dir / model_id.split("/")[-1]
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
        token=token,
    )
    return local_dir
