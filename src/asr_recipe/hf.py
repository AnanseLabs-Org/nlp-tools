from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, urlparse

import requests

from asr_recipe.models import DatasetSpec, ShardRef, SourceMetadata
from asr_recipe.progress import NullProgressReporter, ProgressReporter


class HfMetadataError(RuntimeError):
    """Raised when Hugging Face metadata cannot be resolved."""


@dataclass
class HfMetadataClient:
    base_url: str = "https://datasets-server.huggingface.co"
    timeout_seconds: float = 30.0
    session: requests.Session | None = None
    progress: ProgressReporter = NullProgressReporter()

    def fetch_source_metadata(self, dataset: str, config: str, spec: DatasetSpec | None = None) -> SourceMetadata:
        self.progress.emit(f"Fetching dataset metadata: {dataset} [{config}]")
        info_payload = self._get_json(
            f"{self.base_url}/info?dataset={quote(dataset, safe='')}&config={quote(config, safe='')}"
        )
        parquet_payload = self._get_json(
            f"{self.base_url}/parquet?dataset={quote(dataset, safe='')}&config={quote(config, safe='')}"
        )
        dataset_info = self._extract_dataset_info(info_payload, config)
        features = dataset_info.get("features", {})
        split_info = dataset_info.get("splits", {})
        splits = {name: int(details["num_examples"]) for name, details in split_info.items()}
        shards = tuple(self._build_shard_refs(parquet_payload=parquet_payload, dataset=dataset, spec=spec))
        return SourceMetadata(features=features, splits=splits, shards=shards)

    def _build_shard_refs(
        self,
        parquet_payload: dict[str, object],
        dataset: str,
        spec: DatasetSpec | None,
    ) -> list[ShardRef]:
        refs: list[ShardRef] = []
        seen_counts: dict[tuple[str, str], int] = {}
        for item in parquet_payload.get("parquet_files", []):
            split = item["split"]
            path = self._to_dataset_path(url=item["url"], dataset=dataset, split=split, spec=spec)
            seen_counts[(split, path)] = seen_counts.get((split, path), 0) + 1
            refs.append(
                ShardRef(
                    split=split,
                    path=path,
                    filename=item["filename"],
                    size_bytes=int(item["size"]),
                )
            )
        for (split, path), count in sorted(seen_counts.items()):
            noun = "shard" if count == 1 else "shards"
            self.progress.emit(f"Resolved {count} {noun} for split '{split}': {path}")
        return refs

    def _get_json(self, url: str) -> dict[str, object]:
        session = self.session or requests.Session()
        try:
            response = session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HfMetadataError(f"Failed to fetch Hugging Face metadata from {url}: {exc}") from exc
        return response.json()

    @staticmethod
    def _extract_dataset_info(payload: dict[str, object], config: str) -> dict[str, object]:
        dataset_info = payload.get("dataset_info")
        if isinstance(dataset_info, dict) and "features" in dataset_info:
            return dataset_info
        if isinstance(dataset_info, dict) and config in dataset_info:
            config_info = dataset_info.get(config)
            if isinstance(config_info, dict):
                return config_info
        raise HfMetadataError(f"Unexpected dataset_info payload shape for config '{config}'")

    @staticmethod
    def _to_dataset_path(url: str, dataset: str, split: str, spec: DatasetSpec | None) -> str:
        if spec and spec.repo_parquet_pattern:
            return spec.repo_parquet_pattern.format(dataset=dataset, split=split)
        parsed = urlparse(url)
        if parsed.netloc != "huggingface.co":
            raise HfMetadataError(f"Unexpected parquet host in shard url: {url}")
        path = parsed.path.lstrip("/")
        if "/resolve/refs%2Fconvert%2Fparquet/" in path:
            path = path.replace("/resolve/refs%2Fconvert%2Fparquet/", "@refs/convert/parquet/")
        if path.startswith("datasets/"):
            return f"hf://{path}"
        return f"hf://datasets/{path}"
