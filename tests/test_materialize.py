from __future__ import annotations

import json


def test_materialize_dataset_writes_filtered_parquet(fake_service, recipe_file, text_frequency_file, tmp_path) -> None:
    out_dir = tmp_path / "materialized"

    payload = fake_service.materialize_dataset(
        recipe_path=recipe_file,
        out_dir=str(out_dir),
        output_format="parquet",
        include_audio=False,
        batch_size=1,
        max_rows_per_file=1,
        top_tokens_files=[f"waxal_aka_asr={text_frequency_file}"],
        top_k_tokens=10,
        min_overlap_ratio=0.5,
        min_text_tokens=1,
        max_text_tokens=None,
    )

    manifest = json.loads((out_dir / "materialization_manifest.json").read_text(encoding="utf-8"))
    assert payload["out_dir"] == str(out_dir)
    assert payload["suggested_repo_slug"] == "recipe"
    assert "total_duration_hours" in payload
    assert manifest["suggested_repo_slug"] == "recipe"
    assert "total_duration_hours" in manifest
    assert manifest["recipe"]["filters"]["status"] == "configured"
    assert any(split["split"] in {"train", "validation"} for split in manifest["splits"])
    assert (out_dir / "train" / "data-00000.parquet").exists() or (out_dir / "validation" / "data-00000.parquet").exists()
