# OllamaMerger - Development Guide

## Build & Install

```bash
pip install -e ".[dev]"
```

## Commands

```bash
# Run tests
pytest

# Run single test
pytest tests/test_parser.py -v

# Lint
ruff check src

# Format
ruff format src
```

## Architecture

- `src/ollama_merger/core/parser.py` - HuggingFace URL/ID parsing
- `src/ollama_merger/core/downloader.py` - Model download from HF Hub
- `src/ollama_merger/core/modelfile.py` - Ollama Modelfile generation
- `src/ollama_merger/cli.py` - Typer CLI application
- `src/ollama_merger/web/app.py` - FastAPI web interface
