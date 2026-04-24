"""FastAPI web interface for OllamaMerger."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ollama_merger.core.downloader import download_model
from ollama_merger.core.modelfile import generate_modelfile, write_modelfile
from ollama_merger.core.parser import parse_model_id

app = FastAPI(title="OllamaMerger", description="Convert HuggingFace models to Ollama format")

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

OUTPUT_DIR = Path("./output")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main conversion form."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/convert")
async def convert(
    request: Request,
    url: str = Form(...),
    system_prompt: str = Form(""),
    revision: str = Form(""),
):
    """Convert a HuggingFace model to Ollama format."""
    try:
        parsed = parse_model_id(url)
    except ValueError as e:
        return templates.TemplateResponse(
            request, "result.html", {"success": False, "error": str(e)}
        )

    rev = revision if revision else parsed.revision

    try:
        model_dir = download_model(parsed.model_id, OUTPUT_DIR, revision=rev)
        modelfile_path = write_modelfile(
            model_dir, system_prompt=system_prompt if system_prompt else None
        )
        modelfile_content = generate_modelfile(
            model_dir, system_prompt=system_prompt if system_prompt else None
        )
    except Exception as e:
        return templates.TemplateResponse(
            request, "result.html", {"success": False, "error": str(e)}
        )

    model_name = parsed.model_id.split("/")[-1].lower()
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "success": True,
            "model_id": parsed.model_id,
            "model_dir": str(model_dir),
            "modelfile_path": str(modelfile_path),
            "modelfile_content": modelfile_content,
            "model_name": model_name,
        },
    )


@app.post("/create")
async def create_ollama_model(
    request: Request,
    model_dir: str = Form(...),
    model_name: str = Form(...),
):
    """Run ollama create to import the model."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["ollama", "create", model_name, "-f", "Modelfile"],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
    except FileNotFoundError:
        success = False
        output = "Ollama is not installed or not found in PATH."
    except subprocess.TimeoutExpired:
        success = False
        output = "Model creation timed out (10 min limit)."

    return templates.TemplateResponse(
        request,
        "create_result.html",
        {
            "success": success,
            "model_name": model_name,
            "output": output,
        },
    )
