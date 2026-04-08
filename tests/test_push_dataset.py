from __future__ import annotations

import json
import sys
import types

from asr_recipe.materialize import push_materialized_dataset


def test_push_materialized_dataset_uses_datasets_library(tmp_path, monkeypatch) -> None:
    materialized_dir = tmp_path / "materialized"
    materialized_dir.mkdir()
    parquet_dir = materialized_dir / "train"
    parquet_dir.mkdir()
    parquet_file = parquet_dir / "data.parquet"
    parquet_file.write_bytes(b"PAR1")
    (materialized_dir / "materialization_manifest.json").write_text(
        json.dumps(
            {
                "splits": [
                    {
                        "split": "train",
                        "rows": 1,
                        "duration_seconds": 0.0,
                        "parquet_path": str(parquet_file),
                        "arrow_path": None,
                    }
                ],
                "suggested_repo_slug": "derived-slug",
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    class FakeDatasetDict:
        def push_to_hub(self, repo_id, private, token, max_shard_size):
            calls["push"] = {
                "repo_id": repo_id,
                "private": private,
                "token": token,
                "max_shard_size": max_shard_size,
            }

    fake_module = types.SimpleNamespace(load_dataset=lambda format_name, data_files: calls.setdefault("load", {"format": format_name, "data_files": data_files}) or FakeDatasetDict())
    fake_module.load_dataset = lambda format_name, data_files: (
        calls.setdefault("load", {"format": format_name, "data_files": data_files}),
        FakeDatasetDict(),
    )[1]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    payload = push_materialized_dataset(
        materialized_dir=str(materialized_dir),
        owner="AnanseLabs-Org",
        slug=None,
        private=True,
        token="secret",
        max_shard_size="1GB",
    )

    assert calls["load"]["format"] == "parquet"
    assert calls["push"]["repo_id"] == "AnanseLabs-Org/derived-slug"
    assert payload["splits"] == ["train"]


def test_push_materialized_dataset_accepts_slug_override(tmp_path, monkeypatch) -> None:
    materialized_dir = tmp_path / "materialized"
    materialized_dir.mkdir()
    parquet_dir = materialized_dir / "train"
    parquet_dir.mkdir()
    parquet_file = parquet_dir / "data.parquet"
    parquet_file.write_bytes(b"PAR1")
    (materialized_dir / "materialization_manifest.json").write_text(
        json.dumps(
            {
                "splits": [
                    {
                        "split": "train",
                        "rows": 1,
                        "duration_seconds": 0.0,
                        "parquet_path": str(parquet_file),
                        "arrow_path": None,
                    }
                ],
                "suggested_repo_slug": "derived-slug",
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    class FakeDatasetDict:
        def push_to_hub(self, repo_id, private, token, max_shard_size):
            calls["repo_id"] = repo_id

    fake_module = types.SimpleNamespace()
    fake_module.load_dataset = lambda format_name, data_files: FakeDatasetDict()
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    push_materialized_dataset(
        materialized_dir=str(materialized_dir),
        owner="AnanseLabs-Org",
        slug="custom-slug",
        private=False,
        token=None,
        max_shard_size="1GB",
    )

    assert calls["repo_id"] == "AnanseLabs-Org/custom-slug"
