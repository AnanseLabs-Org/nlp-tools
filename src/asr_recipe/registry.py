from __future__ import annotations

from asr_recipe.models import DatasetSpec


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "waxal_aka_asr": DatasetSpec(
        key="waxal_aka_asr",
        dataset="fiifinketia/WaxalNLP",
        config="aka_asr",
        text_column="transcription",
        duration_column=None,
        id_column="id",
        speaker_id_column="speaker_id",
        gender_column="gender",
        language_column="language",
        default_language=None,
        default_splits=("train", "validation", "test"),
        export_excluded_splits=("unlabeled",),
        repo_parquet_pattern="hf://datasets/{dataset}@main/data/ASR/aka/aka-{split}-*.parquet",
    ),
    "ghana_english_2700hrs": DatasetSpec(
        key="ghana_english_2700hrs",
        dataset="fiifinketia/ghana-english-asr-2700hrs",
        config="default",
        text_column="corrected_text",
        duration_column="duration_ss",
        id_column=None,
        speaker_id_column=None,
        gender_column=None,
        language_column=None,
        default_language="en-GH",
        default_splits=("train",),
        train_only=True,
    ),
    "twi_trigrams_parallel": DatasetSpec(
        key="twi_trigrams_parallel",
        dataset="fiifinketia/twi-trigrams-speech-text-parallel",
        config="default",
        text_column="text",
        duration_column=None,
        id_column=None,
        speaker_id_column=None,
        gender_column=None,
        language_column=None,
        default_language="twi",
        default_splits=("train",),
        train_only=True,
    ),
}


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unsupported dataset '{key}'. Expected one of: {supported}") from exc
