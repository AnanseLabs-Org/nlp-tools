from __future__ import annotations

from collections import Counter
from statistics import mean

from asr_recipe.models import CanonicalRecord


def schema(records: list[CanonicalRecord]) -> dict[str, object]:
    observed = {
        field: sorted({type(getattr(record, field)).__name__ for record in records if getattr(record, field) is not None})
        for field in CanonicalRecord.__dataclass_fields__
    }
    return {"rows": len(records), "observed_types": observed}


def text_frequency(records: list[CanonicalRecord]) -> dict[str, object]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.text:
            counter.update(token.lower() for token in record.text.split())
    return {"rows": len(records), "top_tokens": counter.most_common(20)}


def text_length(records: list[CanonicalRecord]) -> dict[str, object]:
    lengths = [len(record.text.split()) for record in records if record.text]
    if not lengths:
        return {"rows": len(records), "count_with_text": 0}
    return {
        "rows": len(records),
        "count_with_text": len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": round(mean(lengths), 4),
    }


def gender(records: list[CanonicalRecord]) -> dict[str, object]:
    values = Counter(record.gender or "<missing>" for record in records)
    return {"rows": len(records), "distribution": dict(values)}


def duration(records: list[CanonicalRecord]) -> dict[str, object]:
    durations = [record.duration_seconds for record in records if record.duration_seconds is not None]
    if not durations:
        return {"rows": len(records), "count_with_duration": 0}
    return {
        "rows": len(records),
        "count_with_duration": len(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_seconds": round(mean(durations), 4),
    }


def noise(_: list[CanonicalRecord]) -> dict[str, object]:
    raise RuntimeError("The 'noise' analysis requires explicit audio approval and is not implemented yet.")


ANALYSIS_REGISTRY = {
    "schema": schema,
    "text-frequency": text_frequency,
    "text-length": text_length,
    "gender": gender,
    "duration": duration,
    "noise": noise,
}
