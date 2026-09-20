"""Portable, explicit sample selections."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    audio: str
    text: str
    accent: str
    speaker_id: str
    gender: str = ""
    start: float = 0.0
    end: float | None = None
    exclude: list[list[float]] = field(default_factory=list)
    legacy_id: str = ""

    def __post_init__(self):
        for key in ("dataset", "sample_id", "audio", "accent", "speaker_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"Missing {key}: {self.sample_id}")
        import math

        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError(f"Invalid start: {self.sample_id}")
        if self.end is not None and (not math.isfinite(self.end) or self.end <= self.start):
            raise ValueError(f"Invalid end: {self.sample_id}")
        for a, b in self.exclude:
            if not math.isfinite(a + b) or a < 0 or b <= a:
                raise ValueError(f"Invalid excluded interval: {self.sample_id}")


def read_manifest(path):
    path = Path(path)
    samples, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["sample_id"] = str(row["sample_id"])
        row["speaker_id"] = str(row["speaker_id"])
        audio = Path(row["audio"]).expanduser()
        row["audio"] = str((path.parent / audio).resolve() if not audio.is_absolute() else audio)
        sample = Sample(**row)
        key = sample.dataset, sample.sample_id
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        samples.append(sample)
    if not samples:
        raise ValueError(f"Empty manifest: {path}")
    return samples


def write_manifest(samples, path):
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = list(samples)
    if not samples:
        raise ValueError("No samples found")
    seen = set()
    for sample in samples:
        key = sample.dataset, sample.sample_id
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
    with path.open("x") as f:
        for sample in samples:
            row = asdict(sample)
            row["audio"] = os.path.relpath(Path(sample.audio).resolve(), path.parent.resolve())
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
