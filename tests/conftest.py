from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from asr_recipe.models import SamplePolicy, ShardRef, SourceMetadata
from asr_recipe.registry import DATASET_REGISTRY
from asr_recipe.service import RecipeService


class FakeTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return list(self._rows)


class FakeParquetReader:
    def __init__(self, rows_by_path: dict[str, list[dict[str, object]]]) -> None:
        self.rows_by_path = rows_by_path
        self.calls: list[dict[str, object]] = []

    def read_rows(self, path: str, columns: list[str], limit: int | None = None) -> FakeTable:
        self.calls.append({"path": path, "columns": list(columns), "limit": limit})
        rows = self.rows_by_path[path]
        trimmed = rows[:limit] if limit is not None else rows
        projected = [{column: row.get(column) for column in columns} for row in trimmed]
        return FakeTable(projected)

    def iter_batches(self, path: str, columns: list[str], batch_size: int = 1000):
        self.calls.append({"path": path, "columns": list(columns), "batch_size": batch_size})
        rows = self.rows_by_path[path]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            projected = [{column: row.get(column) for column in columns} for row in batch]
            yield path, projected


class FakeMetadataClient:
    def __init__(self, metadata_by_dataset: dict[tuple[str, str], SourceMetadata]) -> None:
        self.metadata_by_dataset = metadata_by_dataset

    def fetch_source_metadata(self, dataset: str, config: str, spec=None) -> SourceMetadata:
        return self.metadata_by_dataset[(dataset, config)]


@pytest.fixture
def sample_policy() -> SamplePolicy:
    return SamplePolicy(sample_size=10, preview_rows=2)


@pytest.fixture
def fake_service() -> RecipeService:
    waxal_spec = DATASET_REGISTRY["waxal_aka_asr"]
    ghana_spec = DATASET_REGISTRY["ghana_english_2700hrs"]
    twi_spec = DATASET_REGISTRY["twi_trigrams_parallel"]
    metadata = {
        (waxal_spec.dataset, waxal_spec.config): SourceMetadata(
            features={"audio": {"_type": "Audio"}, "transcription": {"dtype": "string"}},
            splits={"train": 2, "validation": 1, "test": 1, "unlabeled": 1},
            shards=(
                ShardRef(split="train", path="hf://datasets/waxal/train/0000.parquet", filename="0000.parquet", size_bytes=1),
                ShardRef(split="validation", path="hf://datasets/waxal/validation/0000.parquet", filename="0000.parquet", size_bytes=1),
                ShardRef(split="test", path="hf://datasets/waxal/test/0000.parquet", filename="0000.parquet", size_bytes=1),
                ShardRef(split="unlabeled", path="hf://datasets/waxal/unlabeled/0000.parquet", filename="0000.parquet", size_bytes=1),
            ),
        ),
        (ghana_spec.dataset, ghana_spec.config): SourceMetadata(
            features={"audio": {"_type": "Audio"}, "corrected_text": {"dtype": "string"}, "duration_ss": {"dtype": "float32"}},
            splits={"train": 2},
            shards=(ShardRef(split="train", path="hf://datasets/ghana/train/0000.parquet", filename="0000.parquet", size_bytes=1),),
        ),
        (twi_spec.dataset, twi_spec.config): SourceMetadata(
            features={"audio": {"_type": "Audio"}, "text": {"dtype": "string"}},
            splits={"train": 2},
            shards=(ShardRef(split="train", path="hf://datasets/twi/train/0000.parquet", filename="0000.parquet", size_bytes=1),),
        ),
    }
    rows_by_path = {
        "hf://datasets/waxal/train/0000.parquet": [
            {"id": "wax1", "speaker_id": "spk1", "gender": "f", "language": "aka", "transcription": "hello world", "audio": {"bytes": b"x"}},
            {"id": "wax2", "speaker_id": "spk2", "gender": "m", "language": "aka", "transcription": "hello akan", "audio": {"bytes": b"y"}},
        ],
        "hf://datasets/waxal/validation/0000.parquet": [
            {"id": "wax3", "speaker_id": "spk3", "gender": "f", "language": "aka", "transcription": "val sample", "audio": {"bytes": b"z"}},
        ],
        "hf://datasets/waxal/test/0000.parquet": [
            {"id": "wax4", "speaker_id": "spk4", "gender": "m", "language": "aka", "transcription": "test sample", "audio": {"bytes": b"w"}},
        ],
        "hf://datasets/waxal/unlabeled/0000.parquet": [
            {"id": "wax5", "speaker_id": "spk5", "gender": "f", "language": "aka", "transcription": "unlabeled sample", "audio": {"bytes": b"u"}},
        ],
        "hf://datasets/ghana/train/0000.parquet": [
            {"corrected_text": "ghana english text", "duration_ss": 5.5, "audio": {"bytes": b"a"}},
            {"corrected_text": "more speech text", "duration_ss": 2.5, "audio": {"bytes": b"b"}},
        ],
        "hf://datasets/twi/train/0000.parquet": [
            {"text": "twi sample one", "audio": {"bytes": b"c"}},
            {"text": "twi sample two", "audio": {"bytes": b"d"}},
        ],
    }
    reader = FakeParquetReader(rows_by_path=rows_by_path)
    service = RecipeService(metadata_client=FakeMetadataClient(metadata), reader=reader)
    service._fake_reader = reader
    return service


@pytest.fixture
def recipe_file(tmp_path) -> str:
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            {
                "selected_datasets": [
                    {
                        "dataset_key": "waxal_aka_asr",
                        "source_dataset": "fiifinketia/WaxalNLP",
                        "source_subset": "aka_asr",
                        "selected_splits": ["train"],
                        "canonical_column_mapping": {},
                        "upstream_split_counts": {"train": 2},
                    },
                    {
                        "dataset_key": "ghana_english_2700hrs",
                        "source_dataset": "fiifinketia/ghana-english-asr-2700hrs",
                        "source_subset": "default",
                        "selected_splits": ["train"],
                        "canonical_column_mapping": {},
                        "upstream_split_counts": {"train": 2},
                    },
                ],
                "selected_analyses": ["text-frequency"],
                "compatibility_notes": [],
                "sample_policy": {"mode": "sample", "sample_size": 1000, "preview_rows": 5},
                "split_strategy": {"policy": "train-val", "val_ratio": 0.1, "test_ratio": 0.0, "deterministic_key": "record_id"},
                "audio_policy": {"decode_enabled": False, "download_enabled": False},
                "filters": {"status": "not_started", "pipeline": []},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def text_frequency_file(tmp_path) -> str:
    path = tmp_path / "waxal_text_frequency.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "top_tokens": [
                        ["hello", 10],
                        ["world", 9],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)
