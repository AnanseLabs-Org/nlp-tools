from __future__ import annotations

from dataclasses import dataclass, field


CANONICAL_COLUMNS = (
    "source_dataset",
    "source_subset",
    "source_split",
    "dataset_key",
    "record_id",
    "speaker_id",
    "gender",
    "language",
    "text",
    "duration_seconds",
)


@dataclass(frozen=True)
class CanonicalRecord:
    source_dataset: str
    source_subset: str
    source_split: str
    dataset_key: str
    record_id: str
    speaker_id: str | None
    gender: str | None
    language: str | None
    text: str | None
    duration_seconds: float | None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset: str
    config: str
    text_column: str
    duration_column: str | None
    id_column: str | None
    speaker_id_column: str | None
    gender_column: str | None
    language_column: str | None
    default_language: str | None
    default_splits: tuple[str, ...]
    export_excluded_splits: tuple[str, ...] = ()
    train_only: bool = False
    repo_parquet_pattern: str | None = None


@dataclass(frozen=True)
class ShardRef:
    split: str
    path: str
    filename: str
    size_bytes: int


@dataclass(frozen=True)
class SourceMetadata:
    features: dict[str, object]
    splits: dict[str, int]
    shards: tuple[ShardRef, ...]


@dataclass(frozen=True)
class SamplePolicy:
    mode: str = "sample"
    sample_size: int = 10_000
    preview_rows: int = 5


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    record_id: str


@dataclass(frozen=True)
class MaterializedSplitSummary:
    split: str
    rows: int
    duration_seconds: float
    parquet_path: str | None = None
    arrow_path: str | None = None


@dataclass
class RecipeManifest:
    selected_datasets: list[dict[str, object]]
    selected_analyses: list[str]
    compatibility_notes: list[str]
    sample_policy: dict[str, object]
    split_strategy: dict[str, object]
    audio_policy: dict[str, object]
    filters: dict[str, object]
    manifest_version: str = "1.0"
    manifest_type: str = "asr_recipe"
    notebook_contract: str = "Akan_Finetuning_Standard_specAug.ipynb"
    canonical_schema: list[str] = field(default_factory=lambda: list(CANONICAL_COLUMNS))
