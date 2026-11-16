"""CLI tool for document ingestion.

Usage:
    # Ingest a single file
    uv run python -m scripts.ingest ingest --tenant-id myteam data/docs/paper.pdf

    # Ingest a directory
    uv run python -m scripts.ingest ingest-dir data/docs/

    # In-memory smoke test (no API keys, no Qdrant server)
    uv run python -m scripts.ingest smoke
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.ingestion.models import DocumentStatus, IngestionResult

app = typer.Typer(help="DeepResearcher ingestion CLI", add_completion=False)
console = Console()


@app.command()
def ingest(
    source: str = typer.Argument(..., help="File path or YouTube URL"),
    tenant_id: str = typer.Option(None, "--tenant-id", "-t", help="Tenant identifier"),
    strategy: str = typer.Option(
        None, "--strategy", "-s", help="Chunking: fixed, recursive, markdown, semantic"
    ),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Chunk size in characters"),
) -> None:
    """Ingest a single document."""
    from src.core import setup_logging
    from src.ingestion.chunkers import ChunkConfig, ChunkingStrategy
    from src.ingestion.pipeline import IngestionPipeline, PipelineConfig

    setup_logging()
    config = PipelineConfig.from_settings()
    if strategy:
        config.strategy = ChunkingStrategy(strategy)
    if chunk_size:
        assert config.chunk is not None
        config.chunk = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=config.chunk.chunk_overlap,
            min_chunk_size=config.chunk.min_chunk_size,
        )

    pipeline = IngestionPipeline(config=config)
    console.print(f"\n[bold blue]Ingesting:[/bold blue] {source}")
    result = pipeline.ingest_file(source, tenant_id=tenant_id)

    _print_result(result)


@app.command()
def ingest_dir(
    directory: str = typer.Argument(..., help="Directory of documents"),
    tenant_id: str = typer.Option(None, "--tenant-id", "-t"),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories"),
) -> None:
    """Ingest all supported documents in a directory."""
    from src.core import setup_logging
    from src.ingestion.pipeline import IngestionPipeline

    setup_logging()
    path = Path(directory)
    if not path.is_dir():
        console.print(f"[red]Not a directory:[/red] {directory}")
        raise typer.Exit(1)

    pattern = "**/*" if recursive else "*"
    sources = [str(p) for p in path.glob(pattern) if p.is_file()]

    console.print(f"\n[bold blue]Ingesting {len(sources)} files from:[/bold blue] {directory}")
    pipeline = IngestionPipeline()
    results = pipeline.ingest_files(sources, tenant_id=tenant_id)

    table = Table(title="Ingestion Results")
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("Reason", style="dim")

    for r in results:
        status_str = (
            "[green]completed[/green]"
            if r.status.value == "completed"
            else f"[red]{r.status.value}[/red]"
        )
        chunks_str = str(r.chunks_created) if r.chunks_created >= 0 else "dup"
        reason = r.metadata.get("reason", "") or (r.error or "")[:40]
        table.add_row(Path(r.source).name, status_str, chunks_str, reason)

    console.print(table)

    completed = sum(1 for r in results if r.status.value == "completed")
    console.print(f"\n[green]Done:[/green] {completed}/{len(results)} documents ingested\n")


@app.command()
def smoke() -> None:
    """End-to-end smoke test with in-memory Qdrant + fake embedder.

    No API keys needed. Verifies the full pipeline: parse → chunk →
    embed → dedup → upsert.
    """
    from src.ingestion.chunkers import ChunkConfig, ChunkingStrategy
    from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
    from src.ingestion.vectorstore import VectorStore
    from tests.factories import FakeEmbedder

    console.print("\n[bold]Running ingestion smoke test[/bold]")

    store = VectorStore(url=":memory:")
    pipeline = IngestionPipeline(
        embedder=FakeEmbedder(dim=16),
        vectorstore=store,  # type: ignore[arg-type]
        config=PipelineConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk=ChunkConfig(chunk_size=250, chunk_overlap=30),
        ),
    )

    # Write sample doc to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(
            "# Smoke Test Doc\n\n"
            "## Section A\n\n"
            "RAG pipelines have four stages: load, chunk, embed, retrieve. "
            "Each stage is independently optimizable. Loaders extract text "
            "from raw formats. Chunkers split text into embeddable pieces. "
            "Embedders convert chunks to vectors. Retrieval finds matches.\n\n"
            "## Section B\n\n"
            "Production RAG adds observability and caching. Traces show "
            "latency per stage. Caching saves embedding API quota. "
            "Evaluation catches regressions automatically. "
            "Deduplication prevents re-ingesting identical content.\n"
        )
        temp_path = f.name

    r1 = pipeline.ingest_file(temp_path, tenant_id="smoke")
    console.print(
        f"  1st ingest: [{_color(r1.status)}]{r1.status}[/] — "
        f"{r1.chunks_created} chunks, {r1.duration_ms:.0f}ms"
    )

    r2 = pipeline.ingest_file(temp_path, tenant_id="smoke")
    console.print(
        f"  2nd ingest: [{_color(r2.status)}]{r2.status}[/] — "
        f"reason: {r2.metadata.get('reason', 'n/a')} (embedding API saved)"
    )

    count = store.count()
    console.print(f"  Points in store: {count}")

    checks = [
        ("First ingest succeeded", r1.status.value == "completed" and r1.chunks_created > 0),
        ("Dedup blocked re-ingest", r2.metadata.get("reason") == "duplicate"),
        ("Point count stable", count == r1.chunks_created),
    ]

    console.print()
    all_passed = True
    for name, passed in checks:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {icon}  {name}")
        all_passed = all_passed and passed

    console.print()
    if all_passed:
        console.print("[bold green]Smoke test passed![/bold green]\n")
    else:
        console.print("[bold red]Smoke test FAILED[/bold red]\n")
        raise typer.Exit(1)


def _print_result(result: IngestionResult) -> None:
    from src.ingestion.models import DocumentStatus

    color = "green" if result.status == DocumentStatus.COMPLETED else "red"
    console.print(f"  status:         [{color}]{result.status}[/]")
    console.print(f"  chunks created: {result.chunks_created}")
    console.print(f"  chunks skipped: {result.chunks_skipped}")
    console.print(f"  duration:       {result.duration_ms:.0f}ms")
    if result.error:
        console.print(f"  [red]error:[/red]     {result.error}")


def _color(status: DocumentStatus) -> str:
    return "green" if status.value == "completed" else "red"


if __name__ == "__main__":
    sys.exit(app())
