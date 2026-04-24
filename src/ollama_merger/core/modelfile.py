"""Generate Ollama Modelfiles from downloaded model metadata."""

from __future__ import annotations

import json
from pathlib import Path

# Parameters from generation_config.json that map to Ollama PARAMETER directives
GENERATION_PARAM_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "repetition_penalty": "repeat_penalty",
    "max_new_tokens": "num_predict",
}


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def _extract_template(tokenizer_config: dict) -> str | None:
    """Extract chat template from tokenizer_config.json."""
    chat_template = tokenizer_config.get("chat_template")
    if chat_template is None:
        return None

    # If it's a list of templates, use the first default one
    if isinstance(chat_template, list):
        for tpl in chat_template:
            if isinstance(tpl, dict) and tpl.get("name") == "default":
                return tpl.get("template")
        # Fall back to first template
        if chat_template and isinstance(chat_template[0], dict):
            return chat_template[0].get("template")
        return None

    return chat_template


def _extract_stop_tokens(tokenizer_config: dict) -> list[str]:
    """Extract stop tokens from tokenizer config."""
    stop_tokens = []
    eos_token = tokenizer_config.get("eos_token")
    if eos_token and isinstance(eos_token, str):
        stop_tokens.append(eos_token)
    return stop_tokens


def _extract_parameters(generation_config: dict) -> dict[str, str]:
    """Extract Ollama parameters from generation_config.json."""
    params = {}
    for gen_key, ollama_key in GENERATION_PARAM_MAP.items():
        value = generation_config.get(gen_key)
        if value is not None:
            params[ollama_key] = str(value)
    return params


def generate_modelfile(model_dir: Path, system_prompt: str | None = None) -> str:
    """Generate an Ollama Modelfile from downloaded model metadata.

    Args:
        model_dir: Path to the directory containing model files.
        system_prompt: Optional system prompt to include.

    Returns:
        The Modelfile content as a string.
    """
    lines = ['FROM .']

    # Read metadata files
    config = _read_json(model_dir / "config.json")
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    generation_config = _read_json(model_dir / "generation_config.json")

    # Extract and add template
    if tokenizer_config:
        template = _extract_template(tokenizer_config)
        if template:
            lines.append(f'TEMPLATE """{template}"""')

        # Add stop tokens
        for token in _extract_stop_tokens(tokenizer_config):
            lines.append(f'PARAMETER stop "{token}"')

    # Extract and add generation parameters
    if generation_config:
        for key, value in _extract_parameters(generation_config).items():
            lines.append(f"PARAMETER {key} {value}")

    # Add context length from config.json
    if config:
        ctx_length = config.get("max_position_embeddings")
        if ctx_length:
            lines.append(f"PARAMETER num_ctx {ctx_length}")

    # Add system prompt
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')

    content = "\n\n".join(lines) + "\n"
    return content


def write_modelfile(model_dir: Path, system_prompt: str | None = None) -> Path:
    """Generate and write the Modelfile to the model directory.

    Returns:
        Path to the written Modelfile.
    """
    content = generate_modelfile(model_dir, system_prompt)
    modelfile_path = model_dir / "Modelfile"
    modelfile_path.write_text(content)
    return modelfile_path
