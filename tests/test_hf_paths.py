from __future__ import annotations

from asr_recipe.hf import HfMetadataClient
from asr_recipe.registry import DATASET_REGISTRY


def test_waxal_prefers_repo_parquet_pattern() -> None:
    spec = DATASET_REGISTRY["waxal_aka_asr"]
    path = HfMetadataClient._to_dataset_path(
        url="https://huggingface.co/datasets/fiifinketia/WaxalNLP/resolve/refs%2Fconvert%2Fparquet/aka_asr/train/0000.parquet",
        dataset=spec.dataset,
        split="train",
        spec=spec,
    )

    assert path == "hf://datasets/fiifinketia/WaxalNLP@main/data/ASR/aka/aka-train-*.parquet"


def test_default_datasets_map_convert_ref_to_hf_revision_path() -> None:
    path = HfMetadataClient._to_dataset_path(
        url="https://huggingface.co/datasets/fiifinketia/twi-trigrams-speech-text-parallel/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        dataset="fiifinketia/twi-trigrams-speech-text-parallel",
        split="train",
        spec=None,
    )

    assert path == "hf://datasets/fiifinketia/twi-trigrams-speech-text-parallel@refs/convert/parquet/default/train/0000.parquet"
