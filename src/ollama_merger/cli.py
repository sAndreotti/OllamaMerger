"""CLI entry point for OllamaMerger."""

from __future__ import annotations

from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
except ImportError:
    raise SystemExit(
        "Typer and Rich are required for CLI usage. Install with: pip install ollama-merger[cli]"
    )

from ollama_merger.core.downloader import download_model
from ollama_merger.core.modelfile import write_modelfile
from ollama_merger.core.ollama import create_model, push_model
from ollama_merger.core.parser import parse_model_id

app = typer.Typer(
    name="ollama-merger",
    help="Convert HuggingFace models to Ollama-ready format.",
)
console = Console()


@app.command()
def convert(
    url: str = typer.Argument(help="HuggingFace model URL or ID (e.g. mistralai/Mistral-7B-v0.1)"),
    output_dir: Path = typer.Option(
        Path("./output"), "--output-dir", "-o", help="Output directory for downloaded model"
    ),
    revision: str | None = typer.Option(
        None, "--revision", "-r", help="Model revision (branch, tag, or commit)"
    ),
    system_prompt: str | None = typer.Option(
        None, "--system-prompt", "-s", help="System prompt to include in the Modelfile"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactively customize the Modelfile"
    ),
    create: bool = typer.Option(
        False, "--create", "-c", help="Run ollama create after generating the Modelfile"
    ),
    push: bool = typer.Option(
        False, "--push", "-p", help="Push the model to your Ollama account after creating it"
    ),
    model_name: str | None = typer.Option(
        None, "--name", "-n", help="Ollama model name (e.g. username/my-model). Required for --push"
    ),
) -> None:
    """Convert a HuggingFace model to Ollama format."""
    # Parse the model reference
    try:
        parsed = parse_model_id(url)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    model_id = parsed.model_id
    rev = revision or parsed.revision

    console.print(f"[bold]Model:[/bold] {model_id}")
    if rev:
        console.print(f"[bold]Revision:[/bold] {rev}")

    # Download
    console.print("\n[bold blue]Downloading model files...[/bold blue]")
    try:
        model_dir = download_model(model_id, output_dir, revision=rev)
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Downloaded to:[/green] {model_dir}")

    # Interactive customization
    if interactive:
        system_prompt = _interactive_customize(system_prompt)

    # Generate Modelfile
    console.print("\n[bold blue]Generating Modelfile...[/bold blue]")
    modelfile_path = write_modelfile(model_dir, system_prompt=system_prompt)
    console.print(f"[green]Modelfile written to:[/green] {modelfile_path}")

    # Determine model name
    name = model_name or model_id.split("/")[-1].lower()

    # Create model in Ollama
    if create or push:
        console.print(f"\n[bold blue]Creating Ollama model '{name}'...[/bold blue]")
        result = create_model(name, model_dir)
        if not result.success:
            console.print(f"[red]Create failed:[/red] {result.output}")
            raise typer.Exit(1)
        console.print(f"[green]Model '{name}' created successfully![/green]")
        if result.output.strip():
            console.print(f"[dim]{result.output.strip()}[/dim]")

    # Push model to Ollama registry
    if push:
        if "/" not in name:
            console.print(
                "[red]Error:[/red] Model name must include your Ollama username "
                "(e.g. username/my-model) to push. Use --name to set it."
            )
            raise typer.Exit(1)

        console.print(f"\n[bold blue]Pushing '{name}' to Ollama registry...[/bold blue]")
        result = push_model(name)
        if not result.success:
            console.print(f"[red]Push failed:[/red] {result.output}")
            raise typer.Exit(1)
        console.print(f"[green]Model '{name}' pushed successfully![/green]")
        if result.output.strip():
            console.print(f"[dim]{result.output.strip()}[/dim]")

    # Show next steps if not auto-created
    if not create and not push:
        console.print(
            Panel(
                f"cd {model_dir}\nollama create {name} -f Modelfile",
                title="Next steps",
                border_style="cyan",
            )
        )


@app.command()
def push_cmd(
    model_name: str = typer.Argument(help="Ollama model name (e.g. username/my-model)"),
) -> None:
    """Push an existing Ollama model to your account on the registry."""
    if "/" not in model_name:
        console.print(
            "[red]Error:[/red] Model name must include your Ollama username "
            "(e.g. username/my-model)."
        )
        raise typer.Exit(1)

    console.print(f"[bold blue]Pushing '{model_name}' to Ollama registry...[/bold blue]")
    result = push_model(model_name)
    if not result.success:
        console.print(f"[red]Push failed:[/red] {result.output}")
        raise typer.Exit(1)

    console.print(f"[green]Model '{model_name}' pushed successfully![/green]")
    if result.output.strip():
        console.print(f"[dim]{result.output.strip()}[/dim]")


@app.command()
def list(
    url: str = typer.Argument(help="HuggingFace model URL or ID"),
) -> None:
    """List model info, branches, and files from HuggingFace."""
    from huggingface_hub import HfApi

    try:
        parsed = parse_model_id(url)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    api = HfApi()

    try:
        model_info = api.model_info(parsed.model_id)
    except Exception as e:
        console.print(f"[red]Failed to fetch model info:[/red] {e}")
        raise typer.Exit(1)

    # Model info
    console.print(f"\n[bold]{model_info.modelId}[/bold]")
    if model_info.pipeline_tag:
        console.print(f"  Pipeline: {model_info.pipeline_tag}")
    if model_info.library_name:
        console.print(f"  Library: {model_info.library_name}")
    if model_info.tags:
        console.print(f"  Tags: {', '.join(model_info.tags[:10])}")

    # Branches
    try:
        refs = api.list_repo_refs(parsed.model_id)
        if refs.branches:
            console.print("\n[bold]Branches:[/bold]")
            for branch in refs.branches:
                console.print(f"  - {branch.name}")
    except Exception:
        pass

    # Files
    siblings = model_info.siblings or []
    if siblings:
        table = Table(title="Files")
        table.add_column("Filename", style="cyan")
        table.add_column("Size", justify="right")

        for sibling in siblings:
            size = _format_size(sibling.size) if sibling.size else "?"
            table.add_row(sibling.rfilename, size)

        console.print()
        console.print(table)


def _interactive_customize(current_system_prompt: str | None) -> str | None:
    """Interactively customize Modelfile options."""
    console.print("\n[bold yellow]Interactive Modelfile Customization[/bold yellow]")

    if current_system_prompt:
        console.print(f"\nCurrent system prompt: [dim]{current_system_prompt}[/dim]")

    if Confirm.ask("Set a custom system prompt?", default=bool(current_system_prompt)):
        system_prompt = Prompt.ask(
            "System prompt",
            default=current_system_prompt or "",
        )
        return system_prompt if system_prompt else None

    return current_system_prompt


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    app()
