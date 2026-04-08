from __future__ import annotations

from asr_recipe.splits import deterministic_split


def test_deterministic_split_is_stable() -> None:
    first = [deterministic_split("record-1", val_ratio=0.1) for _ in range(5)]
    second = [deterministic_split("record-1", val_ratio=0.1) for _ in range(5)]

    assert first == second


def test_deterministic_split_supports_test_ratio() -> None:
    split = deterministic_split("record-2", val_ratio=0.1, test_ratio=0.1)

    assert split in {"train", "validation", "test"}
