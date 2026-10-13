# Analysis

```sh
# Reproduce from the checked-in transcripts/scores; no inference is needed.
asr-eval analyze \
  --scores data/scores_asr.csv data/allstar_results_for_plotting_0314.csv \
  --config configs/paper.json --output outputs/analysis

# Or analyze newly generated score files.
asr-eval analyze --scores outputs/*/scores.csv --output outputs/new-analysis
```

The configuration specifies model release dates, standard accents, highlighted minority accents, and example model pairs. All configured models are required; use `--datasets` to select partitions or `--allow-partial` for a subset of models.

WER and CER use normalized English text. Empty references are skipped; empty predictions count as deletions. CER includes spaces. Scores are averaged across utterances within each model/accent.

- **Trends:** retain WER ≤ 1 and CER ≤ 1 for non-ALLSTAR datasets, then append the scripted L1 control. ALLSTAR plots cap WER only; its exported CER column is not comparable to the other score exports. Compute Spearman correlation between release dates and minority-minus-standard mean WER.
- **Mean degradation:** use unbounded WERs and average positive later-minus-earlier differences across all model pairs. Rank accents by ascending MD, not by the sum of positive increases. Ties receive the minimum rank; the denominator is the number of accents remaining after dataset selection and filtering.
- **Bootstrap:** independently resample utterance WERs within each model/accent for 1,000 replicates, with seed 0 restarted per dataset. `bootstrap_fraction_nonpositive` is the fraction of minority-minus-standard MD differences ≤ 0; the paper's one-sided rule marks `significant` when that fraction is strictly < 0.05. Also report the 2.5th and 97.5th percentiles as a descriptive two-sided 95% interval. Its lower bound does not determine significance: L2-Arctic Spontaneous has a nonpositive fraction of 0.041 and is significant even though its interval includes zero.
- **Examples:** select a paired utterance from the upper quartile of WER increases for each configured model pair. IDs and reference text must match.

ALLSTAR utterances are excluded from MD only when every model's WER exceeds 1. SAA uses the twelve accent groups specified in `configs/paper.json`. The notebook uses the same selections, filtering, MD and bootstrap functions. It provides plots and Table II from saved scores, while the script interface additionally supports corpus preparation and model inference.

The shared CSV loader splits the stored `ALLSTAR` rows by `task` into `ALLSTAR-LPP`, `ALLSTAR-DHR`, `ALLSTAR-HT1` and `ALLSTAR-HT2`, and translates the corpus's accent codes using the dataset loader's mapping. Per-dataset `model_names` in `configs/paper.json` retain the stored ALLSTAR label `Qwen3 Audio 14B` at the June 2025 release slot. This is not an alias for the `Qwen3 ASR 1.7B` inference checkpoint: the existing ALLSTAR results are the paper's adopted dataset, and no predictions are renamed to imply the checkpoints are interchangeable. Newly generated scores must match the dataset's configured model names; use a separate configuration when analyzing different checkpoints.

Plots use the paper’s colors, markers, gap shading, legends, and provider logos. Model releases are equally spaced and labeled `YY-MM`; correlations still use actual release dates. Pink shading shows the minority–standard WER gap; the faint standard-accent ribbon shows one standard error. ALLSTAR axes label the computed mean WER.

Outputs include `summary.csv`, `accent_scores.csv`, `gaps.csv`, `degradations.csv`, `model_pairs.csv`, `examples.csv`, figures, and a LaTeX table. `analysis.json` records the configuration and seed; `diagnostics.json` records coverage and reference-ID conflicts.

Aggregate regression fixtures and source hashes are in `data/reference/`. To compare computed statistics with the manuscript table:

```sh
asr-eval audit outputs/analysis/summary.csv --output outputs/analysis/check.csv
```
