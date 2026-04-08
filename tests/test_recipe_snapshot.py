from __future__ import annotations

import json


def test_recipe_manifest_snapshot(fake_service, tmp_path) -> None:
    out = tmp_path / "recipe.json"

    fake_service.export_recipe(
        dataset_keys=["waxal_aka_asr", "ghana_english_2700hrs", "twi_trigrams_parallel"],
        analyses=["schema", "text-length"],
        split_policy=None,
        sample_policy=__import__("asr_recipe.models", fromlist=["SamplePolicy"]).SamplePolicy(),
        val_ratio=0.1,
        test_ratio=0.1,
        out_path=str(out),
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {
        "audio_policy": {
            "analysis_gate": "explicit_approval_required",
            "decode_enabled": False,
            "download_enabled": False,
        },
        "canonical_schema": [
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
        ],
        "compatibility_notes": [
            "Audio is present upstream but v1 keeps decode disabled and does not request audio columns.",
            "Only a train split exists upstream; export defaults derive validation deterministically.",
            "Unlabeled examples remain inspectable but are excluded from recipe export by default.",
        ],
        "filters": {"pipeline": [], "status": "not_started"},
        "manifest_type": "asr_recipe",
        "manifest_version": "1.0",
        "notebook_contract": "Akan_Finetuning_Standard_specAug.ipynb",
        "sample_policy": {"mode": "sample", "preview_rows": 5, "sample_size": 10000},
        "selected_analyses": ["schema", "text-length"],
        "selected_datasets": [
            {
                "canonical_column_mapping": {
                    "dataset_key": None,
                    "duration_seconds": None,
                    "gender": "gender",
                    "language": "language",
                    "record_id": "id",
                    "source_dataset": None,
                    "source_split": None,
                    "source_subset": None,
                    "speaker_id": "speaker_id",
                    "text": "transcription",
                },
                "dataset_key": "waxal_aka_asr",
                "selected_splits": ["train", "validation", "test"],
                "source_dataset": "fiifinketia/WaxalNLP",
                "source_subset": "aka_asr",
                "upstream_split_counts": {"test": 1, "train": 2, "unlabeled": 1, "validation": 1},
            },
            {
                "canonical_column_mapping": {
                    "dataset_key": None,
                    "duration_seconds": "duration_ss",
                    "gender": None,
                    "language": "<default:en-GH>",
                    "record_id": "<derived>",
                    "source_dataset": None,
                    "source_split": None,
                    "source_subset": None,
                    "speaker_id": None,
                    "text": "corrected_text",
                },
                "dataset_key": "ghana_english_2700hrs",
                "selected_splits": ["train"],
                "source_dataset": "fiifinketia/ghana-english-asr-2700hrs",
                "source_subset": "default",
                "upstream_split_counts": {"train": 2},
            },
            {
                "canonical_column_mapping": {
                    "dataset_key": None,
                    "duration_seconds": None,
                    "gender": None,
                    "language": "<default:twi>",
                    "record_id": "<derived>",
                    "source_dataset": None,
                    "source_split": None,
                    "source_subset": None,
                    "speaker_id": None,
                    "text": "text",
                },
                "dataset_key": "twi_trigrams_parallel",
                "selected_splits": ["train"],
                "source_dataset": "fiifinketia/twi-trigrams-speech-text-parallel",
                "source_subset": "default",
                "upstream_split_counts": {"train": 2},
            },
        ],
        "split_strategy": {
            "deterministic_key": "record_id",
            "policy": "train-val",
            "test_ratio": 0.1,
            "val_ratio": 0.1,
        },
    }
