"""Visual regression against the paper and checks for configurable model labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from longitudinal_asr.analysis import plot_trends

ROOT = Path(__file__).parents[1]


def test_spontaneous_plot_matches_paper(tmp_path):
    config = json.loads((ROOT / "configs/paper.json").read_text())
    models = sorted(config["models"], key=lambda m: (m["release_date"], m["id"]))
    dates = {m["name"]: m["release_date"] for m in models}
    data = pd.read_csv(ROOT / "data/reference/accent_scores.csv")
    data = data[(data.dataset == "L2-Arctic Spontaneous") & (data.stage == "trend")]
    means = data.pivot(index="model", columns="native_language", values="mean").reindex(dates)
    sem = data.pivot(index="model", columns="native_language", values="sem").reindex(dates)
    gap = means.Korean - means.USA
    rho, p = spearmanr(range(len(gap)), gap)
    output = tmp_path / "spontaneous.png"
    plot_trends(
        means,
        dates,
        "USA",
        "Korean",
        {
            "dataset": "L2-Arctic Spontaneous",
            "mean_gap": gap.mean(),
            "spearman_rho": rho,
            "spearman_p": p,
        },
        output,
        sem,
        {m["name"]: m["provider"] for m in models},
    )
    with (
        Image.open(output) as actual,
        Image.open(ROOT / "docs/figures/l2_arctic_spontaneous.png") as paper,
    ):
        assert actual.size == paper.size == (2400, 1500)
        difference = np.abs(
            np.asarray(actual.convert("RGB"), dtype=float)
            - np.asarray(paper.convert("RGB"), dtype=float)
        )
    # Allow antialiasing/font rasterization differences, but catch layout, palette,
    # date spacing, missing logos, or uncertainty bands replacing gap shading.
    assert difference.mean() < 3
    assert np.mean(difference.max(axis=2) < 64) > 0.97


def test_allstar_labels_preserve_each_model_and_computed_statistic(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.offsetbox import AnnotationBbox

    means = pd.DataFrame(
        {"English": [0.1, 0.2, 0.1], "Nkore": [0.3, 0.5, 0.2]},
        index=["custom-a", "custom-b", "custom-c"],
    )
    dates = dict(zip(means.index, ["2021-01", "2021-01", "2025-11"]))
    saved = []
    original = Figure.savefig

    def inspect(fig, *args, **kwargs):
        top, bottom = fig.axes
        assert top.get_ylabel() == "Mean WER (%)"
        assert [t.get_text() for t in bottom.get_xticklabels()] == ["21-01", "21-01", "25-11"]
        np.testing.assert_array_equal(bottom.get_xticks(), [0, 1, 2])
        np.testing.assert_allclose(bottom.lines[0].get_ydata(), [20, 30, 10])
        assert len([a for a in bottom.artists if isinstance(a, AnnotationBbox)]) == 2
        saved.append(True)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    plot_trends(
        means,
        dates,
        "English",
        "Nkore",
        {
            "dataset": "ALLSTAR-HT1",
            "mean_gap": 0.2,
            "spearman_rho": -0.5,
            "spearman_p": 0.5,
        },
        tmp_path / "allstar.png",
        providers={"custom-a": "openai", "custom-b": "meta"},
    )
    assert saved
