from __future__ import annotations

import json

from typer.testing import CliRunner

import asr_recipe.cli as cli


runner = CliRunner()


def _patch_service(monkeypatch, fake_service) -> None:
    monkeypatch.setattr(cli, "RecipeService", lambda *args, **kwargs: fake_service)


def test_inspect_cli(monkeypatch, fake_service, sample_policy) -> None:
    _patch_service(monkeypatch, fake_service)

    result = runner.invoke(cli.app, ["inspect", "--dataset", "waxal_aka_asr", "--split", "train", "--sample-size", "2"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["datasets"][0]["dataset_key"] == "waxal_aka_asr"
    assert payload["datasets"][0]["split_summaries"][0]["rows_sampled"] == 2


def test_inspect_cli_quiet_has_no_stderr(monkeypatch, fake_service) -> None:
    _patch_service(monkeypatch, fake_service)

    result = runner.invoke(cli.app, ["inspect", "--dataset", "waxal_aka_asr", "-q"])

    assert result.exit_code == 0
    assert result.stderr == ""


def test_analyze_noise_cli_fails(monkeypatch, fake_service) -> None:
    _patch_service(monkeypatch, fake_service)

    result = runner.invoke(cli.app, ["analyze", "noise", "--dataset", "waxal_aka_asr"])

    assert result.exit_code == 2
    assert "requires explicit audio approval" in result.stderr


def test_export_recipe_cli(monkeypatch, fake_service, tmp_path) -> None:
    _patch_service(monkeypatch, fake_service)
    out = tmp_path / "recipe.json"

    result = runner.invoke(
        cli.app,
        [
            "export-recipe",
            "--dataset",
            "waxal_aka_asr",
            "--dataset",
            "ghana_english_2700hrs",
            "--analysis",
            "schema",
            "--analysis",
            "text-length",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["selected_analyses"] == ["schema", "text-length"]
    assert payload["audio_policy"]["decode_enabled"] is False
