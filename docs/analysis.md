# Analysis

```sh
asr-eval analyze --scores outputs/*/scores.csv --output outputs/analysis
```

The configuration specifies model release dates, standard accents, highlighted minority accents, and example model pairs. All configured models are required; use `--datasets` to select partitions or `--allow-partial` for a subset of models.

WER and CER use normalized English text. Empty references are skipped; empty predictions count as deletions. CER includes spaces. Scores are averaged across utterances within each model/accent.

- **Trends:** retain WER ≤ 1 and CER ≤ 1, then append the scripted L1 control. Compute Spearman correlation between release dates and minority-minus-standard mean WER.
- **Mean degradation:** use unbounded WERs and average positive later-minus-earlier differences across all model pairs. Rank accents by ascending MD.
- **Bootstrap:** independently resample utterance WERs within each model/accent for 1,000 replicates. Report the 2.5th and 97.5th percentiles of the minority-minus-standard MD difference. `bootstrap_fraction_nonpositive` is the fraction of replicates with a difference ≤ 0.
- **Examples:** select a paired utterance from the upper quartile of WER increases for each configured model pair. IDs and reference text must match.

ALLSTAR utterances are excluded only when every model's WER exceeds 1. SAA uses the twelve accent groups specified in `configs/paper.json`.

Plots use the paper’s colors, markers, gap shading, legends, and provider logos. Model releases are equally spaced and labeled `YY-MM`; correlations still use actual release dates. Pink shading shows the minority–standard WER gap; the faint standard-accent ribbon shows one standard error. ALLSTAR axes label the computed mean WER.

Outputs include `summary.csv`, `accent_scores.csv`, `gaps.csv`, `degradations.csv`, `model_pairs.csv`, `examples.csv`, figures, and a LaTeX table. `analysis.json` records the configuration and seed; `diagnostics.json` records coverage and reference-ID conflicts.

Aggregate regression fixtures and source hashes are in `data/reference/`. To compare computed statistics with the manuscript table:

```sh
asr-eval audit outputs/analysis/summary.csv --output outputs/analysis/check.csv
```
