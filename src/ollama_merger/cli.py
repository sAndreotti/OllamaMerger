"""CLI entry point for OllamaMerger."""

try:
    import typer
except ImportError:
    raise SystemExit(
        "Typer is required for CLI usage. Install with: pip install ollama-merger[cli]"
    )

app = typer.Typer(
    name="ollama-merger",
    help="Convert HuggingFace models to Ollama-ready format.",
)


@app.command()
def convert(url: str) -> None:
    """Convert a HuggingFace model to Ollama format."""
    typer.echo(f"Converting {url}... (not yet implemented)")


if __name__ == "__main__":
    app()
