from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from asr_recipe.models import CANONICAL_COLUMNS, CanonicalRecord, MaterializedSplitSummary
from asr_recipe.progress import NullProgressReporter, ProgressReporter
from asr_recipe.splits import deterministic_split


CANONICAL_ARROW_SCHEMA = pa.schema(
    [
        pa.field("source_dataset", pa.string()),
        pa.field("source_subset", pa.string()),
        pa.field("source_split", pa.string()),
        pa.field("dataset_key", pa.string()),
        pa.field("record_id", pa.string()),
        pa.field("speaker_id", pa.string()),
        pa.field("gender", pa.string()),
        pa.field("language", pa.string()),
        pa.field("text", pa.string()),
        pa.field("duration_seconds", pa.float64()),
    ]
)

AUDIO_AWARE_ARROW_SCHEMA = pa.schema(
    list(CANONICAL_ARROW_SCHEMA)
    + [
        pa.field(
            "audio",
            pa.struct(
                [
                    pa.field("bytes", pa.binary()),
                    pa.field("path", pa.string()),
                ]
            ),
        )
    ]
)


def load_recipe(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "asr-dataset"


def build_repo_slug(recipe: dict[str, object], recipe_path: str | None = None) -> str:
    if recipe_path:
        stem = Path(recipe_path).stem
        if stem:
            return slugify(stem)
    dataset_keys = [item.get("dataset_key", "") for item in recipe.get("selected_datasets", [])]
    if dataset_keys:
        base = "-".join(dataset_keys[:3])
        return slugify(f"{base}-dataset")
    return "asr-dataset"


def parse_key_value_pairs(items: list[str], cast=str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got '{item}'")
        key, value = item.split("=", 1)
        parsed[key] = cast(value)
    return parsed


def load_top_tokens(path: str, top_k: int) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    top_tokens = payload.get("result", {}).get("top_tokens", [])
    return [token for token, _count in top_tokens[:top_k]]


def build_text_filter_configs(
    recipe: dict[str, object],
    top_tokens_files: dict[str, str],
    top_k_tokens: int,
    min_overlap_ratio: float,
    min_text_tokens: int,
    max_text_tokens: int | None,
) -> dict[str, dict[str, object]]:
    configured: dict[str, dict[str, object]] = {}
    for dataset_key, path in top_tokens_files.items():
        configured[dataset_key] = {
            "type": "text_frequency_overlap",
            "dataset_key": dataset_key,
            "top_tokens": load_top_tokens(path, top_k=top_k_tokens),
            "min_overlap_ratio": min_overlap_ratio,
            "min_text_tokens": min_text_tokens,
            "max_text_tokens": max_text_tokens,
        }

    filters = recipe.setdefault("filters", {"status": "not_started", "pipeline": []})
    pipeline = filters.setdefault("pipeline", [])
    for config in configured.values():
        pipeline.append(config)
    if pipeline:
        filters["status"] = "configured"
    return configured


def index_filter_configs(recipe: dict[str, object]) -> dict[str, dict[str, object]]:
    filters = recipe.get("filters", {})
    pipeline = filters.get("pipeline", []) if isinstance(filters, dict) else []
    result: dict[str, dict[str, object]] = {}
    for item in pipeline:
        if item.get("type") == "text_frequency_overlap" and item.get("dataset_key"):
            result[item["dataset_key"]] = item
    return result


def record_passes_filters(record: CanonicalRecord, filter_config: dict[str, object] | None) -> bool:
    if filter_config is None:
        return True
    if not record.text:
        return False
    tokens = [token.lower() for token in record.text.split() if token.strip()]
    if len(tokens) < int(filter_config.get("min_text_tokens", 1)):
        return False
    max_text_tokens = filter_config.get("max_text_tokens")
    if max_text_tokens is not None and len(tokens) > int(max_text_tokens):
        return False
    vocabulary = set(filter_config.get("top_tokens", []))
    if not vocabulary:
        return True
    overlap = sum(1 for token in tokens if token in vocabulary)
    ratio = overlap / len(tokens) if tokens else 0.0
    return ratio >= float(filter_config.get("min_overlap_ratio", 0.0))


def assign_split(record: CanonicalRecord, split_strategy: dict[str, object]) -> str:
    policy = split_strategy.get("policy", "preserve")
    if policy == "preserve":
        return record.source_split
    if policy == "train-val":
        return deterministic_split(record.record_id, val_ratio=float(split_strategy.get("val_ratio", 0.1)))
    if policy == "train-val-test":
        return deterministic_split(
            record.record_id,
            val_ratio=float(split_strategy.get("val_ratio", 0.1)),
            test_ratio=float(split_strategy.get("test_ratio", 0.1)),
        )
    raise ValueError(f"Unsupported split policy '{policy}'")


class SplitWriter:
    def __init__(
        self,
        split: str,
        out_dir: Path,
        output_format: str,
        include_audio: bool = False,
        max_rows_per_file: int = 100_000,
    ) -> None:
        self.split = split
        self.out_dir = out_dir
        self.output_format = output_format
        self.include_audio = include_audio
        self.max_rows_per_file = max_rows_per_file
        self.parquet_path: Path | None = None
        self.arrow_path: Path | None = None
        self._parquet_writer: pq.ParquetWriter | None = None
        self._arrow_writer: ipc.RecordBatchFileWriter | None = None
        self.rows = 0
        self.duration_seconds = 0.0
        self.schema = AUDIO_AWARE_ARROW_SCHEMA if include_audio else CANONICAL_ARROW_SCHEMA
        self.parquet_files: list[str] = []
        self.arrow_files: list[str] = []
        self._rows_in_current_file = 0
        self._file_index = 0

    def write(self, rows: list[dict[str, object]]) -> None:
        start = 0
        while start < len(rows):
            if self._parquet_writer is None and self._arrow_writer is None:
                self._open_new_file()
            available = self.max_rows_per_file - self._rows_in_current_file
            chunk = rows[start : start + available]
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            if self._parquet_writer is not None:
                self._parquet_writer.write_table(table)
            if self._arrow_writer is not None:
                self._arrow_writer.write_table(table)
            self.rows += len(chunk)
            self._rows_in_current_file += len(chunk)
            self.duration_seconds += sum((row.get("duration_seconds") or 0.0) for row in chunk)
            start += len(chunk)
            if self._rows_in_current_file >= self.max_rows_per_file:
                self._close_current_file()

    def close(self) -> MaterializedSplitSummary:
        self._close_current_file()
        return MaterializedSplitSummary(
            split=self.split,
            rows=self.rows,
            duration_seconds=self.duration_seconds,
            duration_hours=round(self.duration_seconds / 3600.0, 6),
            parquet_files=self.parquet_files,
            arrow_files=self.arrow_files,
        )

    def _open_new_file(self) -> None:
        target_dir = self.out_dir / self.split
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.output_format in {"parquet", "both"}:
            self.parquet_path = target_dir / f"data-{self._file_index:05d}.parquet"
            self._parquet_writer = pq.ParquetWriter(str(self.parquet_path), self.schema)
            self.parquet_files.append(str(self.parquet_path))
        if self.output_format in {"arrow", "both"}:
            self.arrow_path = target_dir / f"data-{self._file_index:05d}.arrow"
            sink = pa.OSFile(str(self.arrow_path), "wb")
            self._arrow_writer = ipc.new_file(sink, self.schema)
            self.arrow_files.append(str(self.arrow_path))
        self._rows_in_current_file = 0
        self._file_index += 1

    def _close_current_file(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        if self._arrow_writer is not None:
            self._arrow_writer.close()
            self._arrow_writer = None
        self.parquet_path = None
        self.arrow_path = None
        self._rows_in_current_file = 0


def write_materialization_manifest(
    out_dir: str,
    recipe: dict[str, object],
    split_summaries: list[MaterializedSplitSummary],
    suggested_repo_slug: str,
    include_audio: bool,
) -> str:
    manifest_path = Path(out_dir) / "materialization_manifest.json"
    payload = {
        "manifest_type": "materialized_asr_dataset",
        "canonical_schema": list(CANONICAL_COLUMNS),
        "audio_enabled": include_audio,
        "suggested_repo_slug": suggested_repo_slug,
        "recipe": recipe,
        "total_duration_seconds": sum(summary.duration_seconds for summary in split_summaries),
        "total_duration_hours": round(sum(summary.duration_seconds for summary in split_summaries) / 3600.0, 6),
        "splits": [asdict(summary) for summary in split_summaries if summary.rows > 0],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(manifest_path)


def push_materialized_dataset(
    materialized_dir: str,
    owner: str,
    slug: str | None,
    private: bool,
    token: str | None,
    max_shard_size: str,
    progress: ProgressReporter | None = None,
) -> dict[str, object]:
    progress = progress or NullProgressReporter()
    manifest_path = Path(materialized_dir) / "materialization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_slug = slug or manifest.get("suggested_repo_slug")
    if not resolved_slug:
        raise ValueError("Unable to determine dataset slug. Provide --slug explicitly.")
    repo_id = f"{owner}/{resolved_slug}"
    split_files: dict[str, list[str] | str] = {}
    format_name = None
    for split in manifest.get("splits", []):
        if split.get("parquet_files"):
            split_files[split["split"]] = split["parquet_files"]
            format_name = "parquet"
        elif split.get("arrow_files"):
            split_files[split["split"]] = split["arrow_files"]
            format_name = "arrow"
    if not split_files or format_name is None:
        raise FileNotFoundError(f"No materialized dataset files found under {materialized_dir}")

    from datasets import load_dataset

    progress.emit(f"Loading local {format_name} dataset files for push")
    dataset_dict = load_dataset(format_name, data_files=split_files)
    if manifest.get("audio_enabled") and "audio" in dataset_dict[next(iter(dataset_dict))].column_names:
        from datasets import Audio

        progress.emit("Casting audio column to Hugging Face Audio feature")
        dataset_dict = dataset_dict.cast_column("audio", Audio(decode=False))
    progress.emit(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    dataset_dict.push_to_hub(repo_id, private=private, token=token, max_shard_size=max_shard_size)
    return {
        "repo_id": repo_id,
        "materialized_dir": materialized_dir,
        "splits": sorted(split_files),
        "private": private,
    }
