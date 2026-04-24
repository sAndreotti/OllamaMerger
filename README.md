# OllamaMerger

[![CI](https://github.com/sAndreotti/OllamaMerger/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sAndreotti/OllamaMerger/actions/workflows/ci.yml)

Convert HuggingFace models to Ollama-ready format with a single command.

OllamaMerger automates the process of downloading SafeTensor models from HuggingFace and generating the corresponding Ollama Modelfile, so you can import any HF model into Ollama without manual configuration.

## Features

- **Download** SafeTensor models and metadata from HuggingFace
- **Generate** Ollama Modelfiles with correct parameters, templates, and system prompts
- **Interactive CLI** to browse model variants and customize the Modelfile
- **Web UI** for browser-based usage
- **Docker** support for containerized deployment

## Quick Start

### Installation

```bash
pip install ollama-merger
```

Or install from source:

```bash
git clone https://github.com/sAndreotti/OllamaMerger.git
cd OllamaMerger
pip install -e ".[cli]"
```

### Usage

```bash
# Convert a HuggingFace model
ollama-merger convert https://huggingface.co/mistralai/Mistral-7B-v0.1

# List available branches/variants
ollama-merger list https://huggingface.co/mistralai/Mistral-7B-v0.1

# Interactive mode with Modelfile customization
ollama-merger convert https://huggingface.co/mistralai/Mistral-7B-v0.1 --interactive
```

Then import into Ollama:

```bash
cd output/Mistral-7B-v0.1
ollama create my-model -f Modelfile
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src
```

## Roadmap

- [x] Project scaffolding
- [x] MVP: download + Modelfile generation
- [x] Interactive CLI
- [x] Docker + Web UI
- [x] CI/CD pipeline

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
