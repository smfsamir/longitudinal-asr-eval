from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from longitudinal_asr.analysis import bootstrap_md, filter_allstar, mean_degradation, read_scores


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.1, 0.3, 0.2], 0.15),
        ([0.3, 0.2, 0.1], 0),
        ([0.1, 0.1, 0.1], 0),
        ([0.1, 0.2, 0.3], 2 / 15),
    ],
)
def test_positive_pairs_only(values, expected):
    assert mean_degradation(values) == pytest.approx(expected)


def test_invalid_degradation():
    with pytest.raises(ValueError):
        mean_degradation([0.1, np.nan])


def test_allstar_filter_requires_all_models():
    df = pd.DataFrame(
        {"sample_id": ["a", "a", "b", "b"], "model": ["m1", "m2"] * 2, "wer": [1.1, 2, 1.0, 2]}
    )
    assert filter_allstar(df, ["m1", "m2"]).sample_id.tolist() == ["b", "b"]
    with pytest.raises(ValueError, match="every selected model"):
        filter_allstar(df.iloc[:3], ["m1", "m2"])


def test_bootstrap_is_reproducible_and_positive():
    df = pd.DataFrame(
        [
            {"native_language": a, "model": m, "wer": v}
            for a, values in [("standard", [0.1, 0.1, 0.1]), ("minority", [0.1, 0.5, 0.8])]
            for m, v in zip(["m1", "m2", "m3"], values)
            for _ in range(4)
        ]
    )
    first = bootstrap_md(df, ["m1", "m2", "m3"], "standard", "minority", n=40)
    assert first == bootstrap_md(df, ["m1", "m2", "m3"], "standard", "minority", n=40)
    assert first["significant"]
    assert first["md_difference_ci_low"] > 0


def test_paper_statistics():
    root = Path(__file__).parents[1]
    means = pd.read_csv(root / "data/reference/accent_scores.csv")
    published = pd.read_csv(root / "data/paper_values.csv")
    accents = {
        "L2-Arctic": ("Vietnamese", "USA"),
        "L2-Arctic Spontaneous": ("Korean", "USA"),
        "SAA": ("Korean", "English"),
        "Openslr83": ("Irish English", "Southern English"),
    }
    for dataset, (minority, standard) in accents.items():
        raw = means[(means.dataset == dataset) & (means.stage == "degradation")]
        for role, accent in [("minority", minority), ("standard", standard)]:
            values = raw[raw.native_language == accent].sort_values("release_date")["mean"]
            expected = published[
                (published.dataset == dataset) & (published.metric == role + "_md")
            ].iloc[0]
            assert abs(mean_degradation(values) - expected.value) <= expected.tolerance
        if dataset in ["L2-Arctic", "Openslr83"]:
            trend = means[(means.dataset == dataset) & (means.stage == "trend")]
            matrix = trend.pivot(
                index="release_date", columns="native_language", values="mean"
            ).sort_index()
            rho = spearmanr(range(len(matrix)), matrix[minority] - matrix[standard]).statistic
            expected = published[
                (published.dataset == dataset) & (published.metric == "spearman_rho")
            ].iloc[0]
            assert rho == pytest.approx(expected.value)


def test_duplicate_scores_rejected(tmp_path):
    row = dict(
        dataset="d",
        model="m",
        sample_id="1",
        release_date="2021-01",
        native_language="a",
        groundtruth="hello",
        prediction="hello",
        wer=0,
        cer=0,
    )
    path = tmp_path / "scores.csv"
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_scores([path])
