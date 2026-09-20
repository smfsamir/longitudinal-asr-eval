"""One model per process, append-only results, and checked resumption."""

import csv
import importlib.metadata
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import load_audio
from .manifest import digest, read_manifest, sha256_file
from .metrics import score
from .models import load_model

COLUMNS = [
    "dataset",
    "model",
    "release_date",
    "sample_id",
    "native_language",
    "speaker_id",
    "groundtruth",
    "prediction",
    "wer",
    "cer",
    "sample_hash",
    "model_hash",
]


def run(manifest, spec, output, device="auto", seed=0, limit=None, keep_going=False):
    samples = read_manifest(manifest)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        samples = samples[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    # Torch remains optional for data preparation, scoring, and replay.
    if spec["backend"] not in {"replay", "google", "stt"}:
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            if spec["backend"] != "whisper":
                raise
    audio_hashes, hashes = {}, {}
    for sample in samples:
        if sample.audio not in audio_hashes:
            audio_hashes[sample.audio] = sha256_file(sample.audio)
        row = asdict(sample)
        row.pop("audio")  # Moving a dataset should not invalidate identical content.
        hashes[sample.dataset, sample.sample_id] = digest(
            {**row, "audio_sha256": audio_hashes[sample.audio]}
        )
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    local_assets = {}
    for path in [spec["checkpoint"], spec.get("options", {}).get("scorer", "")]:
        if Path(path).is_file():
            local_assets[path] = sha256_file(path)
    # Hash implementation as well as inputs so an edited decoder cannot silently resume.
    implementation = {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    signature = {
        "model": spec,
        "device": device,
        "seed": seed,
        "samples": sorted(hashes.values()),
        "packages": packages,
        "implementation": implementation,
        "local_assets": local_assets,
    }
    run_hash = digest(signature)
    metadata_path, csv_path = output / "run.json", output / "scores.csv"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["run_hash"] != run_hash:
            raise ValueError("Run inputs, environment, or code changed; use a new output directory")
    else:
        if csv_path.exists():
            raise ValueError("scores.csv exists without run.json; use a new output directory")
        metadata_path.write_text(
            json.dumps(
                {
                    **signature,
                    "run_hash": run_hash,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "python": sys.version,
                },
                indent=2,
            )
            + "\n"
        )
    done = set()
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = row["dataset"], row["sample_id"]
                if (
                    key in done
                    or row["sample_hash"] != hashes.get(key)
                    or row["model_hash"] != run_hash
                ):
                    raise ValueError(f"Conflicting or damaged result row: {key}")
                if row["prediction"] is None or row["cer"] is None:
                    raise ValueError(f"Incomplete result row: {key}")
                float(row["wer"]), float(row["cer"])
                done.add(key)
    todo = [s for s in samples if (s.dataset, s.sample_id) not in done]
    if not todo:
        return {"completed": len(done), "new": 0, "skipped": 0}
    if spec["backend"] == "replay":
        with open(spec["checkpoint"]) as f:
            predictions = {
                (r["dataset"], str(r["sample_id"])): r["prediction"] for r in csv.DictReader(f)
            }
        transcribe = None
    else:
        transcribe = load_model(spec, device)
    failures, added, skipped = 0, 0, 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
            f.flush()
        for sample in todo:
            key = sample.dataset, sample.sample_id
            if score(sample.text, "") is None:
                skipped += 1
                continue
            try:
                pcm = load_audio(sample)
                prediction = predictions[key] if transcribe is None else transcribe(pcm)
                if not isinstance(prediction, str):
                    raise TypeError(f"Decoder returned {type(prediction).__name__}, expected text")
                metrics = score(sample.text, prediction)
                writer.writerow(
                    dict(
                        dataset=sample.dataset,
                        model=spec["name"],
                        release_date=spec["release_date"],
                        sample_id=sample.sample_id,
                        native_language=sample.accent,
                        speaker_id=sample.speaker_id,
                        groundtruth=sample.text,
                        prediction=prediction,
                        **metrics,
                        sample_hash=hashes[key],
                        model_hash=run_hash,
                    )
                )
                f.flush()
                added += 1
            except Exception as e:
                with (output / "errors.jsonl").open("a") as errors:
                    errors.write(
                        json.dumps(
                            {
                                "dataset": key[0],
                                "sample_id": key[1],
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        + "\n"
                    )
                if not keep_going:
                    raise
                failures += 1
    if failures:
        raise RuntimeError(
            f"{failures} samples failed; see {output / 'errors.jsonl'}. Rerun to retry."
        )
    return {"completed": len(done) + added, "new": added, "skipped": skipped}
