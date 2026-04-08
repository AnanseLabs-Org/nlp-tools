from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from asr_recipe.adapters import DatasetAdapter
from asr_recipe.models import RecipeManifest, SamplePolicy


def build_recipe_manifest(
    adapters: list[DatasetAdapter],
    analyses: list[str],
    split_policy: str,
    sample_policy: SamplePolicy,
    val_ratio: float,
    test_ratio: float,
    include_unlabeled: bool,
) -> RecipeManifest:
    dataset_entries: list[dict[str, object]] = []
    compatibility_notes: list[str] = []
    for adapter in adapters:
        selected_splits = list(adapter.default_export_splits(include_unlabeled=include_unlabeled))
        dataset_entries.append(
            {
                "dataset_key": adapter.spec.key,
                "source_dataset": adapter.spec.dataset,
                "source_subset": adapter.spec.config,
                "selected_splits": selected_splits,
                "canonical_column_mapping": adapter.canonical_mapping(),
                "upstream_split_counts": adapter.metadata.splits,
            }
        )
        compatibility_notes.extend(adapter.compatibility_notes())

    return RecipeManifest(
        selected_datasets=dataset_entries,
        selected_analyses=analyses,
        compatibility_notes=sorted(set(compatibility_notes)),
        sample_policy=asdict(sample_policy),
        split_strategy={
            "policy": split_policy,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "deterministic_key": "record_id",
        },
        audio_policy={
            "decode_enabled": False,
            "download_enabled": False,
            "analysis_gate": "explicit_approval_required",
        },
        filters={
            "status": "not_started",
            "pipeline": [],
        },
    )


def write_recipe_manifest(path: str, manifest: RecipeManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
