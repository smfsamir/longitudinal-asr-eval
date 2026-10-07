"""Paper analyses: trends, positive-pair degradations, and bootstrap intervals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .manifest import digest, sha256_file

KEY = ["dataset", "model", "sample_id"]


def read_scores(paths):
    frames = [pd.read_csv(p, keep_default_na=False, dtype={"sample_id": str}) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    required = {*KEY, "release_date", "native_language", "groundtruth", "prediction", "wer", "cer"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if df.duplicated(KEY).any():
        raise ValueError("Duplicate dataset/model/sample_id results")
    for column in ["wer", "cer"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all() or (df[column] < 0).any():
            raise ValueError(f"Invalid {column}")
    return df


def mean_degradation(values):
    """Mean WER increase over all chronologically ordered pairs with an increase."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Mean degradation needs at least two finite, ordered model means")
    i, j = np.triu_indices(len(values), 1)
    delta = values[j] - values[i]
    positive = delta[delta > 0]
    return float(positive.mean()) if len(positive) else 0.0


def pairwise(means):
    rows = []
    for accent in means.columns:
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                old, new = means.index[i], means.index[j]
                delta = means.iloc[j][accent] - means.iloc[i][accent]
                rows.append(
                    {
                        "native_language": accent,
                        "older_model": old,
                        "newer_model": new,
                        "delta_wer": delta,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_md(df, order, standard, minority, n=1000, seed=0):
    """Resample utterance WERs independently within each accent/model cell.

    Return a percentile interval for minority-minus-standard mean degradation.
    """
    if n < 1:
        raise ValueError("Bootstrap count must be positive")
    rng = np.random.default_rng(seed)
    groups = {
        (accent, model): group.wer.to_numpy()
        for (accent, model), group in df.groupby(["native_language", "model"])
    }
    differences = np.empty(n)
    for b in range(n):
        md = {}
        for accent in [standard, minority]:
            values = []
            for model in order:
                scores = groups[accent, model]
                values.append(rng.choice(scores, len(scores), replace=True).mean())
            md[accent] = mean_degradation(values)
        differences[b] = md[minority] - md[standard]
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return {
        "md_difference_ci_low": float(lo),
        "md_difference_ci_high": float(hi),
        "bootstrap_fraction_nonpositive": float(np.mean(differences <= 0)),
        "significant": bool(lo > 0),
        "bootstrap_samples": n,
    }


def filter_allstar(df, models):
    """Drop an utterance only when every evaluated model has WER > 1."""
    groups = df.groupby("sample_id")
    if not groups.model.nunique().eq(len(models)).all():
        raise ValueError("ALLSTAR filtering requires every selected model for every utterance")
    bad = groups.wer.min()
    return df[~df.sample_id.isin(bad[bad > 1].index)].copy()


def analyze(df, config, output, datasets=None, allow_partial=False, bootstrap=None, seed=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = config["analysis"]
    expected = [
        m["name"] for m in sorted(config["models"], key=lambda x: (x["release_date"], x["id"]))
    ]
    dates = {m["name"]: m["release_date"] for m in config["models"]}
    requested = list(datasets or settings["datasets"])
    unknown = set(requested) - set(settings["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    n = bootstrap if bootstrap is not None else settings.get("bootstrap", 1000)
    seed = config.get("seed", 0) if seed is None else seed
    all_summary, all_means, all_md, all_pairs, all_examples, all_gaps, diagnostics = (
        [],
        [],
        [],
        [],
        [],
        [],
        {},
    )
    for dataset in requested:
        cfg = settings["datasets"][dataset]
        raw = df[df.dataset == dataset].copy()
        if raw.empty:
            if not allow_partial:
                raise ValueError(
                    f"No results for {dataset}; select --datasets or use --allow-partial"
                )
            diagnostics[dataset] = {"missing": True}
            continue
        standard = cfg["standard"]
        if "control_dataset" in cfg and standard not in set(raw.native_language):
            control = df[
                (df.dataset == cfg["control_dataset"]) & (df.native_language == standard)
            ].copy()
            control["sample_id"] = cfg["control_dataset"] + "/" + control.sample_id
            raw = pd.concat([raw, control], ignore_index=True)
        raw = raw[raw.model.isin(expected)]
        if "accents" in cfg:
            raw = raw[raw.native_language.isin(cfg["accents"])]
        if not raw.release_date.eq(raw.model.map(dates)).all():
            raise ValueError(f"Model dates differ from the config: {dataset}")
        order = [m for m in expected if m in set(raw.model)]
        if len(order) < 2 or (not allow_partial and order != expected):
            raise ValueError(
                f"Missing model results for {dataset}: {sorted(set(expected) - set(order))}"
            )
        if dataset.startswith("ALLSTAR-"):
            raw = filter_allstar(raw, order)
        means = raw.pivot_table(index="model", columns="native_language", values="wer").reindex(
            order
        )
        if means.empty or means.isna().any().any():
            raise ValueError(
                f"Every accent needs observations from every selected model: {dataset}"
            )
        if standard not in means:
            raise ValueError(f"Missing standard accent {standard!r} for {dataset}")
        trend = raw.copy()
        if settings.get("trend_filter", "wer_cer_le_1") == "wer_cer_le_1":
            # The L1-Arctic fallback is appended after the notebook's filter.
            keep = (trend.wer <= 1) & (trend.cer <= 1)
            if "control_dataset" in cfg:
                keep |= trend.dataset.eq(cfg["control_dataset"])
            trend = trend[keep]
        elif settings["trend_filter"] != "none":
            raise ValueError("Unknown trend_filter")
        trend_means = trend.pivot_table(
            index="model", columns="native_language", values="wer"
        ).reindex(order)
        highest = trend_means.drop(columns=standard).mean().idxmax()
        minority = cfg.get("highlight", highest)
        if minority not in means or minority not in trend_means:
            raise ValueError(f"Missing highlighted accent {minority!r} for {dataset}")
        md = pd.Series({accent: mean_degradation(means[accent]) for accent in means})
        ranks = md.rank(method="min", ascending=True)
        pairs = pairwise(means)
        pairs.insert(0, "dataset", dataset)
        all_pairs.append(pairs)
        for accent in means:
            all_md.append(
                {
                    "dataset": dataset,
                    "native_language": accent,
                    "md": md[accent],
                    "rank": int(ranks[accent]),
                    "accents": len(means.columns),
                }
            )
        gap = (trend_means[minority] - trend_means[standard]).dropna()
        if len(gap) < 3:
            raise ValueError(f"Not enough paired model means for a trend: {dataset}")
        time = pd.to_datetime([dates[m] for m in gap.index]).map(pd.Timestamp.toordinal)
        rho, p = spearmanr(time, gap)
        for model, value in gap.items():
            all_gaps.append(
                {"dataset": dataset, "model": model, "release_date": dates[model], "gap": value}
            )
        summary = {
            "dataset": dataset,
            "standard": standard,
            "minority": minority,
            "highest_mean_wer_accent": highest,
            "models": len(order),
            "trend_models": len(gap),
            "mean_gap": gap.mean(),
            "spearman_rho": rho,
            "spearman_p": p,
            "standard_md": md[standard],
            "minority_md": md[minority],
            "standard_rank": int(ranks[standard]),
            "minority_rank": int(ranks[minority]),
            **bootstrap_md(raw, order, standard, minority, n, seed),
        }
        all_summary.append(summary)
        for stage, data in [("degradation", raw), ("trend", trend)]:
            agg = (
                data.groupby(["model", "native_language"])
                .wer.agg(["mean", "sem", "count"])
                .reset_index()
            )
            agg["dataset"], agg["stage"] = dataset, stage
            agg["release_date"] = agg.model.map(dates)
            all_means.append(agg)
        # Report reference-ID drift instead of treating old integer IDs as paired observations.
        ref_conflicts = raw.groupby("sample_id").groundtruth.nunique().gt(1)
        diagnostics[dataset] = {
            "rows": len(raw),
            "trend_rows": len(trend),
            "reference_id_conflicts": int(ref_conflicts.sum()),
            "highlight_matches_highest": minority == highest,
            "missing_models": sorted(set(expected) - set(order)),
            "samples_per_model": raw.groupby("model").size().to_dict(),
        }
        example = cfg.get("example")
        if example:
            row = degradation_example(trend, **example, seed=seed)
            if row is not None:
                all_examples.append({"dataset": dataset, **row})
        trend_sem = trend.groupby(["model", "native_language"]).wer.sem().unstack().reindex(order)
        plot_trends(
            trend_means,
            dates,
            standard,
            minority,
            summary,
            output / f"{dataset}.png",
            trend_sem,
            providers={m["name"]: m.get("provider") for m in config["models"]},
        )
    if not all_summary:
        raise ValueError("No datasets analyzed")
    summary = pd.DataFrame(all_summary)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(all_means).to_csv(output / "accent_scores.csv", index=False)
    pd.DataFrame(all_md).to_csv(output / "degradations.csv", index=False)
    pd.concat(all_pairs).to_csv(output / "model_pairs.csv", index=False)
    pd.DataFrame(all_gaps).to_csv(output / "gaps.csv", index=False)
    pd.DataFrame(all_examples).to_csv(output / "examples.csv", index=False)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "config": config,
                "seed": seed,
                "bootstrap": n,
                "datasets": requested,
                "scores_hash": digest(df.to_csv(index=False)),
                "implementation_sha256": sha256_file(__file__),
                "allow_partial": allow_partial,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [r"\begin{tabular}{llrr}", r"Dataset & Accent & MD (\%) & Rank \\", r"\hline"]
    for row in all_summary:
        count = next(r["accents"] for r in all_md if r["dataset"] == row["dataset"])
        for role in ["minority", "standard"]:
            lines.append(
                f"{row['dataset']} & {row[role]} & {100 * row[role + '_md']:.1f} & {row[role + '_rank']}/{count}"
                + r" \\"
            )
    lines.append(r"\end{tabular}")
    (output / "degradations.tex").write_text("\n".join(lines) + "\n")
    return summary


def degradation_example(df, accent, older_model, newer_model, seed=0):
    subset = df[df.native_language == accent]
    old = subset[subset.model == older_model]
    new = subset[subset.model == newer_model]
    if old.empty or new.empty:
        return None
    matched = old.merge(new, on=["sample_id", "groundtruth"], suffixes=("_old", "_new"))
    matched["delta"] = matched.wer_new - matched.wer_old
    top = matched[(matched.delta >= matched.delta.quantile(0.75)) & (matched.delta > 0)]
    if top.empty:
        return None
    row = top.sample(n=1, random_state=seed).iloc[0]
    return {
        "native_language": accent,
        "older_model": older_model,
        "newer_model": newer_model,
        "sample_id": row.sample_id,
        "groundtruth": row.groundtruth,
        "older_prediction": row.prediction_old,
        "newer_prediction": row.prediction_new,
        "delta_wer": row.delta,
        "group_delta_wer": new.wer.mean() - old.wer.mean(),
    }


def plot_trends(means, dates, standard, minority, summary, path, sem=None, providers=None):
    """Render the paper's two-panel plots using the ML notebook's visual conventions.

    Pink shading denotes the accent gap. The faint standard-accent SEM ribbon
    follows the notebook; minority SEM ribbons are absent in the paper figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    providers = providers or {}
    allstar = summary["dataset"].startswith("ALLSTAR-")
    x = np.arange(len(means))
    minority_wer = means[minority].to_numpy() * 100
    standard_wer = means[standard].to_numpy() * 100
    gap = minority_wer - standard_wer
    blue, green = ("steelblue", "seagreen") if allstar else ("#0072B2", "#009E73")
    pink, other_color = "crimson", "#6C7394"

    def maximum_increase(values):
        # The notebook keeps the first pair in case of a tie.
        best = (0.0, None, None)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[j] - values[i] > best[0]:
                    best = (values[j] - values[i], i, j)
        return best

    with plt.style.context(["default", "seaborn-v0_8-colorblind"]):
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(8, 5), dpi=150 if allstar else 300, sharex=True
        )
        try:
            # The ALLSTAR panels use corpus language codes in their labels.
            from .datasets import ALLSTAR_ACCENTS

            codes = {name: code for code, name in ALLSTAR_ACCENTS.items()}
            label = codes.get(minority, minority) if allstar else minority
            control = codes.get(standard, standard) if allstar else "standard"
            top.plot(x, minority_wer, "-o", color=blue, label=f"WER ({label})")
            top.plot(x, standard_wer, "--o", color=green, label=f"WER ({control})")
            if not allstar and sem is not None:
                error = sem[standard].reindex(means.index).fillna(0).to_numpy() * 100
                top.fill_between(
                    x, standard_wer - error, standard_wer + error, color=green, alpha=0.1
                )
            others = [a for a in means if a not in {standard, minority}]
            others.sort(key=lambda a: -maximum_increase(means[a].to_numpy())[0])
            for i, accent in enumerate(others[:10]):
                top.plot(
                    x,
                    means[accent] * 100,
                    ":^",
                    color=other_color,
                    alpha=0.5,
                    label=("WER (other accents)" if allstar else "WER (other dialects)")
                    if i == 0
                    else None,
                    zorder=-1,
                )
            top.fill_between(x, standard_wer, minority_wer, color=pink, alpha=0.15, label="Gap")
            dataset = summary["dataset"]
            title = (
                f"{dataset.replace('-', '_')} | {label} vs {control}"
                if allstar
                else f"{dataset} - {minority} (highest WER) vs {standard} (standard): WER over time"
            )
            top.set(title=title, ylabel="Mean WER (%)" if allstar else "Error Rate (%)")
            top.legend(loc="upper right")
            bottom.plot(
                x,
                gap,
                "--s" if allstar else "-s",
                color=pink,
                alpha=0.6,
                label="Gap size" if allstar else "Gap (minority − standard)",
            )
            bottom.fill_between(x, 0, gap, color=pink, alpha=0.15)
            average = (
                f"Avg gap: {summary['mean_gap'] * 100:.1f}%  |  "
                f"ρ = {summary['spearman_rho']:.2f} (p = {summary['spearman_p']:.3f})"
            )
            bottom.axhline(
                summary["mean_gap"] * 100,
                color=pink,
                linestyle="--",
                alpha=0.6,
                linewidth=1,
                label=average,
            )
            bottom.axhline(0, color="gray", linestyle=":", linewidth=0.5)
            bottom.set_ylabel("Gap (WER difference)" if allstar else "Gap in WER (%)")
            bottom.set_xlabel("Release Date", labelpad=20)
            bottom.set_xticks(x, [pd.Timestamp(dates[m]).strftime("%y-%m") for m in means.index])
            handles, labels = bottom.get_legend_handles_labels()
            if not allstar:
                handles[0] = Line2D([], [], color=pink, marker="s", linestyle="none", alpha=0.6)
            bottom.legend(handles, labels, loc="upper right")
            for ax in (top, bottom):
                ax.grid(True, linestyle="--", alpha=0.6 if not allstar else 0.4)
            for position, model in enumerate(means.index):
                provider = providers.get(model)
                if not provider:
                    continue
                logo = Path(__file__).parent / "logos" / f"{provider}.png"
                if not logo.is_file():
                    raise ValueError(f"Unknown logo provider {provider!r} for {model}")
                pixels = plt.imread(logo)
                bottom.add_artist(
                    AnnotationBbox(
                        OffsetImage(pixels, zoom=10 / pixels.shape[0]),
                        (position, 0),
                        xycoords=("data", "axes fraction"),
                        xybox=(0, -25),
                        boxcoords="offset points",
                        frameon=False,
                        box_alignment=(0.5, 0.5),
                        annotation_clip=False,
                    )
                )
            fig.tight_layout()
            # Match the notebook arrow after layout so offsets stay in display pixels.
            delta, i, j = maximum_increase(minority_wer)
            if delta > 0 and not allstar:
                fig.canvas.draw()
                start = np.array([x[i], minority_wer[i]])
                end = np.array([x[j], minority_wer[j]])
                p0, p1 = top.transData.transform([start, end])
                direction = (p1 - p0) / np.linalg.norm(p1 - p0)
                offset = min(15, np.linalg.norm(p1 - p0) / 4)
                a, b = top.transData.inverted().transform(
                    [p0 + offset * direction, p1 - offset * direction]
                )
                top.annotate(
                    "",
                    xy=b,
                    xytext=a,
                    arrowprops={
                        "headlength": 10,
                        "headwidth": 10,
                        "width": 2,
                        "color": "red",
                    },
                )
                perpendicular = np.array([-(end[1] - start[1]), end[0] - start[0]])
                perpendicular /= np.linalg.norm(perpendicular)
                text_position = top.transData.inverted().transform(
                    top.transData.transform((start + end) / 2) + 8 * perpendicular
                )
                top.text(*text_position, f"{delta:.1f}%", color="red", va="bottom", ha="right")
            fig.savefig(path, bbox_inches="tight" if allstar else None)
        finally:
            plt.close(fig)
