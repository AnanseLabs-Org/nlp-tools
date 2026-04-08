from __future__ import annotations

from asr_recipe.adapters import DatasetAdapter
from asr_recipe.registry import DATASET_REGISTRY


def test_waxal_mapping(fake_service, sample_policy) -> None:
    adapter = fake_service._adapter_for("waxal_aka_asr")
    records = adapter.iter_canonical_records("train", sample_policy)

    assert adapter.canonical_mapping()["text"] == "transcription"
    assert adapter.metadata.shards[0].path == "hf://datasets/waxal/train/0000.parquet"
    assert records[0].record_id == "wax1"
    assert records[0].speaker_id == "spk1"
    assert records[0].language == "aka"


def test_ghana_mapping_derives_ids(fake_service, sample_policy) -> None:
    adapter = fake_service._adapter_for("ghana_english_2700hrs")
    records = adapter.iter_canonical_records("train", sample_policy)

    assert adapter.canonical_mapping()["duration_seconds"] == "duration_ss"
    assert records[0].language == "en-GH"
    assert records[0].record_id.startswith("derived:")
    assert records[0].duration_seconds == 5.5


def test_twi_mapping_derives_ids(fake_service, sample_policy) -> None:
    adapter = fake_service._adapter_for("twi_trigrams_parallel")
    records = adapter.iter_canonical_records("train", sample_policy)

    assert adapter.canonical_mapping()["text"] == "text"
    assert records[0].language == "twi"
    assert records[0].record_id.startswith("derived:")
