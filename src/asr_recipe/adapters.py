from __future__ import annotations

import hashlib
from dataclasses import dataclass

from asr_recipe.models import CanonicalRecord, DatasetSpec, SamplePolicy, ShardRef, SourceMetadata
from asr_recipe.parquet_reader import ParquetReader


NON_AUDIO_REQUIRED_COLUMNS = {
    "id",
    "speaker_id",
    "gender",
    "language",
    "transcription",
    "corrected_text",
    "duration_ss",
    "text",
}


@dataclass
class DatasetAdapter:
    spec: DatasetSpec
    metadata: SourceMetadata
    reader: ParquetReader

    def available_splits(self) -> tuple[str, ...]:
        return tuple(self.metadata.splits)

    def default_export_splits(self, include_unlabeled: bool = False) -> tuple[str, ...]:
        allowed = []
        for split in self.available_splits():
            if split in self.spec.export_excluded_splits and not include_unlabeled:
                continue
            allowed.append(split)
        return tuple(allowed)

    def iter_canonical_records(self, split: str, sample_policy: SamplePolicy) -> list[CanonicalRecord]:
        remaining = sample_policy.sample_size
        records: list[CanonicalRecord] = []
        shards = [shard for shard in self.metadata.shards if shard.split == split]
        for shard in shards:
            if remaining <= 0:
                break
            rows = self.reader.read_rows(
                shard.path,
                columns=self.required_source_columns(),
                limit=remaining,
            ).to_pylist()
            for row_index, row in enumerate(rows):
                records.append(self._to_canonical_record(split=split, shard=shard, row=row, row_index=row_index))
                remaining -= 1
                if remaining <= 0:
                    break
        return records

    def required_source_columns(self) -> list[str]:
        requested = {
            self.spec.text_column,
            self.spec.duration_column,
            self.spec.id_column,
            self.spec.speaker_id_column,
            self.spec.gender_column,
            self.spec.language_column,
        }
        return sorted(column for column in requested if column and column in NON_AUDIO_REQUIRED_COLUMNS)

    def canonical_mapping(self) -> dict[str, str | None]:
        return {
            "source_dataset": None,
            "source_subset": None,
            "source_split": None,
            "dataset_key": None,
            "record_id": self.spec.id_column or "<derived>",
            "speaker_id": self.spec.speaker_id_column,
            "gender": self.spec.gender_column,
            "language": self.spec.language_column or f"<default:{self.spec.default_language}>",
            "text": self.spec.text_column,
            "duration_seconds": self.spec.duration_column,
        }

    def compatibility_notes(self) -> list[str]:
        notes: list[str] = []
        if self.spec.train_only:
            notes.append("Only a train split exists upstream; export defaults derive validation deterministically.")
        if "audio" in self.metadata.features:
            notes.append("Audio is present upstream but v1 keeps decode disabled and does not request audio columns.")
        if self.spec.key == "waxal_aka_asr" and "unlabeled" in self.metadata.splits:
            notes.append("Unlabeled examples remain inspectable but are excluded from recipe export by default.")
        return notes

    def _to_canonical_record(
        self,
        split: str,
        shard: ShardRef,
        row: dict[str, object],
        row_index: int,
    ) -> CanonicalRecord:
        return CanonicalRecord(
            source_dataset=self.spec.dataset,
            source_subset=self.spec.config,
            source_split=split,
            dataset_key=self.spec.key,
            record_id=self._resolve_record_id(row=row, shard=shard, row_index=row_index),
            speaker_id=self._optional_str(row.get(self.spec.speaker_id_column)) if self.spec.speaker_id_column else None,
            gender=self._optional_str(row.get(self.spec.gender_column)) if self.spec.gender_column else None,
            language=self._optional_str(row.get(self.spec.language_column))
            if self.spec.language_column
            else self.spec.default_language,
            text=self._optional_str(row.get(self.spec.text_column)),
            duration_seconds=self._optional_float(row.get(self.spec.duration_column)) if self.spec.duration_column else None,
        )

    def _resolve_record_id(self, row: dict[str, object], shard: ShardRef, row_index: int) -> str:
        if self.spec.id_column:
            explicit_id = row.get(self.spec.id_column)
            if explicit_id is not None and str(explicit_id).strip():
                return str(explicit_id)
        seed = f"{self.spec.key}|{shard.path}|{row_index}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"derived:{digest}"

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        return float(value)
