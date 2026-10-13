"""The saved-score script and notebook must agree with the corrected paper."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from longitudinal_asr.analysis import analyze, bootstrap_md, mean_degradation, read_scores

ROOT = Path(__file__).parents[1]


def test_one_sided_bootstrap_can_pass_when_two_sided_interval_includes_zero():
    frame = pd.DataFrame(
        [
            {"native_language": accent, "model": model, "wer": value}
            for accent, model, values in [
                ("standard", "m1", [0]),
                ("standard", "m2", [0]),
                ("minority", "m1", [0]),
                ("minority", "m2", [0, 1, 1]),
            ]
            for value in values
        ]
    )
    result = bootstrap_md(frame, ["m1", "m2"], "standard", "minority", n=1000, seed=0)
    assert 0.025 <= result["bootstrap_fraction_nonpositive"] < 0.05
    assert result["md_difference_ci_low"] == 0
    assert result["significant"]


def test_spontaneous_bootstrap_matches_paper():
    frame = pd.read_csv(ROOT / "data/scores_asr.csv")
    frame = frame[frame.dataset == "L2-Arctic Spontaneous"]
    models = frame[["model", "release_date"]].drop_duplicates().sort_values("release_date").model
    result = bootstrap_md(frame, models, "USA", "Korean", n=1000, seed=0)
    assert result["bootstrap_fraction_nonpositive"] == pytest.approx(0.041)
    assert result["md_difference_ci_low"] < 0 < result["md_difference_ci_high"]
    assert result["significant"]


def test_stored_allstar_tasks_and_accents_are_loaded_without_model_relabeling(tmp_path):
    rows = [
        dict(
            dataset="ALLSTAR",
            task=task,
            model="Qwen3 Audio 14B",
            release_date="2025-06",
            sample_id="1",
            native_language=accent,
            groundtruth="hello",
            prediction="hello",
            wer=0,
            cer=0,
        )
        for task, accent in [("LPP", "ENG"), ("DHR", "CCT"), ("HT1", "RUN"), ("HT2", "VIE")]
    ]
    path = tmp_path / "allstar.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = read_scores([path])
    assert frame.dataset.tolist() == ["ALLSTAR-LPP", "ALLSTAR-DHR", "ALLSTAR-HT1", "ALLSTAR-HT2"]
    assert frame.native_language.tolist() == ["English", "Cantonese", "Nkore", "Vietnamese"]
    assert frame.model.unique().tolist() == ["Qwen3 Audio 14B"]
    pd.DataFrame(rows).drop(columns="task").to_csv(path, index=False)
    with pytest.raises(ValueError, match="valid task"):
        read_scores([path])


def test_notebook_and_script_rank_by_mean_with_minimum_tie_rank(tmp_path, monkeypatch):
    # Minority has a smaller total increase (0.19 vs 0.28), but a larger MD
    # (0.095 vs 0.0933). Ranking by total would reverse the paper's ordering.
    profiles = {"Minority": [0, 0.1, 0.09], "Standard": [0, 0.07, 0.14], "Tied": [0, 0.07, 0.14]}
    models = [dict(id=f"m{i}", name=f"m{i}", release_date=f"2021-0{i + 1}") for i in range(3)]
    frame = pd.DataFrame(
        [
            dict(
                dataset="Example",
                model=m["name"],
                release_date=m["release_date"],
                sample_id=accent,
                native_language=accent,
                groundtruth="a",
                prediction="a",
                wer=values[i],
                cer=0,
            )
            for accent, values in profiles.items()
            for i, m in enumerate(models)
        ]
    )
    config = dict(
        models=models,
        analysis=dict(datasets={"Example": dict(standard="Standard", highlight="Minority")}),
    )
    monkeypatch.setattr("longitudinal_asr.analysis.plot_trends", lambda *args, **kwargs: None)
    script = analyze(frame, config, tmp_path / "script", bootstrap=20).iloc[0]
    assert script.minority_rank == 3
    assert script.standard_rank == 1
    assert script.accents == 3
    assert "9.5*" in (tmp_path / "script/degradations.tex").read_text()
    notebook = json.loads((ROOT / "reproduce_paper.ipynb").read_text())
    namespace = dict(
        scores=frame,
        DATASETS={"Example": dict(title="Example", minority="Minority", standard="Standard")},
        pd=pd,
        mean_degradation=mean_degradation,
        bootstrap_md=bootstrap_md,
        N_BOOTSTRAP=20,
        SEED=0,
        OUTPUT_DIR=str(tmp_path),
        os=__import__("os"),
        accent_name=lambda accent: accent,
    )
    exec("".join(notebook["cells"][7]["source"]), namespace)
    exec("".join(notebook["cells"][11]["source"]), namespace)
    table = namespace["table2"].iloc[0]
    assert table["Minority rank"] == "3/3"
    assert table["Standard rank"] == "1/3"
    assert table["Minority MD (%)"] == round(100 * script.minority_md, 1)
    assert table["Significant (p < 0.05)"] == ("*" if script.significant else "")


def test_allstar_analysis_preserves_exported_model_and_caps_only_wer(tmp_path, monkeypatch):
    models = [dict(id=f"m{i}", name=f"model{i}", release_date=f"2021-0{i + 1}") for i in range(3)]
    rows = []
    for accent in ["English", "Vietnamese"]:
        for i, m in enumerate(models):
            for sample, wer, cer in [("kept", 0.1 * (i + 1), 2), ("all-fail", 2, 0)]:
                rows.append(
                    dict(
                        dataset="ALLSTAR-LPP",
                        model="exported model" if i == 1 else m["name"],
                        release_date=m["release_date"],
                        sample_id=accent + sample,
                        native_language=accent,
                        groundtruth="a",
                        prediction="a",
                        wer=wer + (0.03 * i if accent == "Vietnamese" and sample == "kept" else 0),
                        cer=cer,
                    )
                )
    config = dict(
        models=models,
        analysis=dict(
            datasets={
                "ALLSTAR-LPP": dict(
                    standard="English", highlight="Vietnamese", model_names={"m1": "exported model"}
                )
            }
        ),
    )
    captured = []
    monkeypatch.setattr(
        "longitudinal_asr.analysis.plot_trends",
        lambda *args, **kwargs: captured.append(kwargs["providers"]),
    )
    analyze(pd.DataFrame(rows), config, tmp_path, bootstrap=20)
    means = pd.read_csv(tmp_path / "accent_scores.csv")
    assert means[means.stage == "trend"]["count"].eq(1).all()
    assert means[means.stage == "degradation"]["count"].eq(1).all()
    assert "exported model" in set(means.model)
    assert "model1" not in set(means.model)
    assert "exported model" in captured[0]
    np.testing.assert_allclose(
        means[means.stage == "trend"]["mean"].sort_values(), [0.1, 0.1, 0.2, 0.23, 0.3, 0.36]
    )
