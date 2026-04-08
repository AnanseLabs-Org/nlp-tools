from __future__ import annotations


def test_non_audio_inspect_never_requests_audio_column(fake_service, sample_policy) -> None:
    fake_service.inspect("waxal_aka_asr", "train", sample_policy)

    assert fake_service._fake_reader.calls
    assert all("audio" not in call["columns"] for call in fake_service._fake_reader.calls)


def test_duration_analysis_never_requests_audio_column(fake_service, sample_policy) -> None:
    fake_service.analyze("duration", "ghana_english_2700hrs", "train", sample_policy)

    assert fake_service._fake_reader.calls
    assert all("audio" not in call["columns"] for call in fake_service._fake_reader.calls)
