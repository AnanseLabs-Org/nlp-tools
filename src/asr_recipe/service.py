from __future__ import annotations

from dataclasses import asdict

from asr_recipe.adapters import DatasetAdapter
from asr_recipe.analyses import ANALYSIS_REGISTRY
from asr_recipe.hf import HfMetadataClient
from asr_recipe.models import SamplePolicy
from asr_recipe.parquet_reader import ParquetReader, PyArrowParquetReader
from asr_recipe.progress import NullProgressReporter, ProgressReporter
from asr_recipe.recipe import build_recipe_manifest, write_recipe_manifest
from asr_recipe.registry import DATASET_REGISTRY, get_dataset_spec


class RecipeService:
    def __init__(
        self,
        metadata_client: HfMetadataClient | None = None,
        reader: ParquetReader | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.progress = progress or NullProgressReporter()
        self.metadata_client = metadata_client or HfMetadataClient(progress=self.progress)
        self.reader = reader or PyArrowParquetReader(progress=self.progress)

    def inspect(self, dataset_key: str, split: str | None, sample_policy: SamplePolicy) -> dict[str, object]:
        dataset_keys = sorted(DATASET_REGISTRY) if dataset_key == "all" else [dataset_key]
        datasets = []
        for key in dataset_keys:
            self.progress.emit(f"Inspecting dataset: {key}")
            adapter = self._adapter_for(key)
            splits = [split] if split else list(adapter.available_splits())
            split_summaries = []
            for split_name in splits:
                self.progress.emit(f"Sampling split: {key}/{split_name}")
                records = adapter.iter_canonical_records(split_name, sample_policy)
                split_summaries.append(
                    {
                        "split": split_name,
                        "rows_sampled": len(records),
                        "preview": [asdict(record) for record in records[: sample_policy.preview_rows]],
                    }
                )
            datasets.append(
                {
                    "dataset_key": key,
                    "source_dataset": adapter.spec.dataset,
                    "source_subset": adapter.spec.config,
                    "available_splits": list(adapter.available_splits()),
                    "canonical_column_mapping": adapter.canonical_mapping(),
                    "split_summaries": split_summaries,
                }
            )
        return {"sample_policy": asdict(sample_policy), "datasets": datasets}

    def analyze(self, analysis_name: str, dataset_key: str, split: str | None, sample_policy: SamplePolicy) -> dict[str, object]:
        if analysis_name not in ANALYSIS_REGISTRY:
            supported = ", ".join(sorted(ANALYSIS_REGISTRY))
            raise KeyError(f"Unsupported analysis '{analysis_name}'. Expected one of: {supported}")
        self.progress.emit(f"Running analysis: {analysis_name} on {dataset_key}")
        adapter = self._adapter_for(dataset_key)
        split_name = split or adapter.spec.default_splits[0]
        self.progress.emit(f"Sampling split: {dataset_key}/{split_name}")
        records = adapter.iter_canonical_records(split_name, sample_policy)
        result = ANALYSIS_REGISTRY[analysis_name](records)
        return {
            "dataset_key": dataset_key,
            "split": split_name,
            "analysis": analysis_name,
            "sample_policy": asdict(sample_policy),
            "result": result,
        }

    def export_recipe(
        self,
        dataset_keys: list[str],
        analyses: list[str],
        split_policy: str | None,
        sample_policy: SamplePolicy,
        val_ratio: float,
        test_ratio: float,
        out_path: str,
        include_unlabeled: bool = False,
    ) -> dict[str, object]:
        self.progress.emit(f"Preparing recipe export for {len(dataset_keys)} dataset(s)")
        adapters = [self._adapter_for(key) for key in dataset_keys]
        resolved_policy = split_policy or self._default_split_policy(adapters)
        self.progress.emit(f"Using split policy: {resolved_policy}")
        manifest = build_recipe_manifest(
            adapters=adapters,
            analyses=analyses,
            split_policy=resolved_policy,
            sample_policy=sample_policy,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            include_unlabeled=include_unlabeled,
        )
        write_recipe_manifest(out_path, manifest)
        self.progress.emit(f"Wrote recipe manifest: {out_path}")
        return {"out": out_path, "split_policy": resolved_policy, "dataset_keys": dataset_keys}

    def _adapter_for(self, dataset_key: str) -> DatasetAdapter:
        spec = get_dataset_spec(dataset_key)
        metadata = self.metadata_client.fetch_source_metadata(spec.dataset, spec.config, spec)
        return DatasetAdapter(spec=spec, metadata=metadata, reader=self.reader)

    @staticmethod
    def _default_split_policy(adapters: list[DatasetAdapter]) -> str:
        if any(adapter.spec.train_only for adapter in adapters):
            return "train-val"
        return "preserve"
