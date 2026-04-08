from __future__ import annotations

import json
from pathlib import Path

import typer

from asr_recipe.service import RecipeService
from asr_recipe.models import SamplePolicy
from asr_recipe.progress import NullProgressReporter, ProgressReporter
from asr_recipe.registry import DATASET_REGISTRY


app = typer.Typer(
    help=(
        "Inspect ASR datasets, run non-audio analyses, export a recipe manifest, materialize a filtered "
        "canonical dataset locally, and push the result to Hugging Face Hub. Audio decode stays off unless "
        "a future audio-gated path is explicitly approved."
    )
)


class CliProgressReporter:
    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet

    def emit(self, message: str) -> None:
        if not self.quiet:
            typer.echo(f"[asr-recipe] {message}", err=True)


def _sample_policy(sample_size: int, preview_rows: int) -> SamplePolicy:
    return SamplePolicy(sample_size=sample_size, preview_rows=preview_rows)


def _print_json(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _service(quiet: bool) -> RecipeService:
    progress: ProgressReporter = NullProgressReporter() if quiet else CliProgressReporter(quiet=False)
    return RecipeService(progress=progress)


@app.command("inspect")
def inspect_command(
    dataset: str = typer.Option(..., help=f"Dataset key or 'all'. Available: {', '.join(sorted(DATASET_REGISTRY))}"),
    split: str | None = typer.Option(None, help="Optional split override such as train, validation, test, or unlabeled."),
    sample_size: int = typer.Option(10_000, min=1, help="Maximum sampled rows to read without touching audio."),
    preview_rows: int = typer.Option(5, min=0, help="Preview rows to include in inspect output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output and only print JSON results."),
) -> None:
    """Preview canonical mappings and sample rows."""
    service = _service(quiet)
    payload = service.inspect(dataset_key=dataset, split=split, sample_policy=_sample_policy(sample_size, preview_rows))
    _print_json(payload)


@app.command("analyze")
def analyze_command(
    module: str = typer.Argument(..., help="Analysis module: schema, text-frequency, text-length, gender, duration, noise."),
    dataset: str = typer.Option(..., help=f"Dataset key. Available: {', '.join(sorted(DATASET_REGISTRY))}"),
    split: str | None = typer.Option(None, help="Optional split override. Defaults to the dataset's primary split."),
    sample_size: int = typer.Option(10_000, min=1, help="Maximum sampled rows to read without touching audio."),
    preview_rows: int = typer.Option(5, min=0, help="Reserved for consistent sample policy reporting."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output and only print JSON results."),
) -> None:
    """Run one non-audio analysis module in sample mode."""
    service = _service(quiet)
    try:
        payload = service.analyze(
            analysis_name=module,
            dataset_key=dataset,
            split=split,
            sample_policy=_sample_policy(sample_size, preview_rows),
        )
    except RuntimeError as exc:
        raise typer.Exit(code=_echo_error(str(exc)))
    _print_json(payload)


@app.command("export-recipe")
def export_recipe_command(
    dataset: list[str] = typer.Option(..., help="One or more dataset keys."),
    analysis: list[str] = typer.Option([], help="Analysis modules selected for the downstream notebook."),
    split_policy: str | None = typer.Option(
        None,
        help="Override split policy: preserve, train-val, or train-val-test. Defaults to preserve unless any source is train-only.",
    ),
    val_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="Validation ratio for derived splits."),
    test_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="Test ratio for train-val-test derivation."),
    sample_size: int = typer.Option(10_000, min=1, help="Sample policy stored in the recipe manifest."),
    preview_rows: int = typer.Option(5, min=0, help="Preview policy stored in the recipe manifest."),
    include_unlabeled: bool = typer.Option(False, help="Include inspectable unlabeled splits in the recipe export."),
    out: Path = typer.Option(..., help="Output path for the JSON recipe manifest."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output and only print JSON results."),
) -> None:
    """Export the JSON recipe handoff manifest without materializing a dataset."""
    allowed = {None, "preserve", "train-val", "train-val-test"}
    if split_policy not in allowed:
        raise typer.BadParameter("split-policy must be one of preserve, train-val, train-val-test")
    service = _service(quiet)
    payload = service.export_recipe(
        dataset_keys=dataset,
        analyses=analysis,
        split_policy=split_policy,
        sample_policy=_sample_policy(sample_size, preview_rows),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        out_path=str(out),
        include_unlabeled=include_unlabeled,
    )
    _print_json(payload)


@app.command("materialize-dataset")
def materialize_dataset_command(
    recipe: Path = typer.Option(..., help="Path to an exported recipe JSON manifest."),
    out_dir: Path = typer.Option(..., help="Directory to write the local materialized dataset."),
    format: str = typer.Option("parquet", help="Output format: parquet, arrow, or both."),
    batch_size: int = typer.Option(1000, min=1, help="Rows per parquet batch while materializing."),
    top_tokens_file: list[str] = typer.Option(
        [],
        help="Attach text-frequency filters as DATASET_KEY=analysis.json. Can be provided multiple times.",
    ),
    top_k_tokens: int = typer.Option(500, min=1, help="How many top tokens to keep from each analysis file."),
    min_overlap_ratio: float = typer.Option(0.5, min=0.0, max=1.0, help="Minimum token overlap ratio for text-frequency filters."),
    min_text_tokens: int = typer.Option(1, min=1, help="Minimum transcript token count to keep."),
    max_text_tokens: int | None = typer.Option(None, min=1, help="Optional maximum transcript token count to keep."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output and only print JSON results."),
) -> None:
    """Read a recipe, normalize rows, apply filters, and write local Parquet or Arrow splits."""
    allowed = {"parquet", "arrow", "both"}
    if format not in allowed:
        raise typer.BadParameter("format must be one of parquet, arrow, both")
    service = _service(quiet)
    payload = service.materialize_dataset(
        recipe_path=str(recipe),
        out_dir=str(out_dir),
        output_format=format,
        batch_size=batch_size,
        top_tokens_files=top_tokens_file,
        top_k_tokens=top_k_tokens,
        min_overlap_ratio=min_overlap_ratio,
        min_text_tokens=min_text_tokens,
        max_text_tokens=max_text_tokens,
    )
    _print_json(payload)


@app.command("push-dataset")
def push_dataset_command(
    materialized_dir: Path = typer.Option(..., help="Directory created by materialize-dataset."),
    owner: str = typer.Option(..., help="Target Hugging Face username or organization."),
    slug: str | None = typer.Option(None, help="Optional dataset slug override. Defaults to the slug derived from the recipe."),
    private: bool = typer.Option(False, help="Create or update the remote dataset as private."),
    token: str | None = typer.Option(None, help="Optional Hugging Face token. Falls back to existing login."),
    max_shard_size: str = typer.Option("5GB", help="Shard size passed to datasets.push_to_hub()."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output and only print JSON results."),
) -> None:
    """Push a locally materialized dataset directory to Hugging Face Hub."""
    service = _service(quiet)
    payload = service.push_dataset(
        materialized_dir=str(materialized_dir),
        owner=owner,
        slug=slug,
        private=private,
        token=token,
        max_shard_size=max_shard_size,
    )
    _print_json(payload)


def _echo_error(message: str) -> int:
    typer.echo(message, err=True)
    return 2


def main() -> None:
    app()


if __name__ == "__main__":
    main()
