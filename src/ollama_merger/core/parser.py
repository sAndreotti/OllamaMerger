"""Parse HuggingFace model URLs and IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass

HF_URL_PATTERN = re.compile(
    r"^(?:https?://)?huggingface\.co/"
    r"(?P<model_id>[^/]+/[^/]+)"
    r"(?:/tree/(?P<revision>[^/]+))?"
    r"/?$"
)

MODEL_ID_PATTERN = re.compile(r"^[^/]+/[^/]+$")


@dataclass
class ParsedModel:
    """Parsed HuggingFace model reference."""

    model_id: str
    revision: str | None = None


def parse_model_id(input_str: str) -> ParsedModel:
    """Parse a HuggingFace model URL or ID into a ParsedModel.

    Accepts:
        - Full URL: https://huggingface.co/mistralai/Mistral-7B-v0.1
        - URL with revision: https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/v2
        - Direct ID: mistralai/Mistral-7B-v0.1

    Raises:
        ValueError: If the input cannot be parsed as a valid model reference.
    """
    input_str = input_str.strip()

    # Try URL pattern first
    match = HF_URL_PATTERN.match(input_str)
    if match:
        return ParsedModel(
            model_id=match.group("model_id"),
            revision=match.group("revision"),
        )

    # Try direct model ID
    if MODEL_ID_PATTERN.match(input_str):
        return ParsedModel(model_id=input_str)

    raise ValueError(
        f"Invalid model reference: '{input_str}'. "
        "Expected a HuggingFace URL (https://huggingface.co/org/model) "
        "or model ID (org/model)."
    )
