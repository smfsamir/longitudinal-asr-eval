"""A deterministic integration example; these are not paper results."""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .analysis import analyze, read_scores
from .manifest import Sample, write_manifest
from .runner import run


def demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio = output / "tone.wav"
    sf.write(audio, (0.1 * np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000)), 16000)
    samples = [
        Sample("Example", f"{a}-{i}", str(audio), "please call stella today", a, f"{a}-{i}")
        for a in ["Standard", "Minority"]
        for i in range(4)
    ]
    manifest = output / "samples.jsonl"
    if not manifest.exists():
        write_manifest(samples, manifest)
    configs, paths = [], []
    hypotheses = {
        "Standard": ["please call stella today", "please call stella", "please call stella today"],
        "Minority": ["please call stella", "please", "please call"],
    }
    for i in range(3):
        predictions = output / f"predictions-{i}.csv"
        with predictions.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "sample_id", "prediction"])
            writer.writeheader()
            writer.writerows(
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "prediction": hypotheses[s.accent][i],
                }
                for s in samples
            )
        spec = {
            "id": f"example-{i}",
            "name": f"Example {i}",
            "release_date": f"202{i + 1}-01",
            "backend": "replay",
            "checkpoint": str(predictions),
        }
        run(manifest, spec, output / spec["id"])
        configs.append(spec)
        paths.append(output / spec["id"] / "scores.csv")
    config = {
        "models": configs,
        "seed": 0,
        "analysis": {
            "bootstrap": 100,
            "trend_filter": "wer_cer_le_1",
            "datasets": {"Example": {"standard": "Standard"}},
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    analyze(read_scores(paths), config, output / "analysis")
    print(f"Synthetic example written to {output}")
