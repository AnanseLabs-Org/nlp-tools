from __future__ import annotations

import hashlib


def deterministic_split(record_id: str, val_ratio: float, test_ratio: float = 0.0) -> str:
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if test_ratio > 0 and bucket < test_ratio:
        return "test"
    shifted = bucket - test_ratio
    remaining = 1.0 - test_ratio
    if remaining <= 0:
        return "test"
    if shifted / remaining < val_ratio:
        return "validation"
    return "train"
