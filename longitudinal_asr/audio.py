"""16 kHz mono PCM, retaining the ML repository's linear resampler."""

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load_audio(sample):
    data, rate = sf.read(sample.audio, dtype="int16", always_2d=True)
    pcm = data.mean(axis=1).astype(np.int16)
    if rate != SAMPLE_RATE:
        pcm = np.interp(
            np.linspace(0, len(pcm), int(len(pcm) * SAMPLE_RATE / rate)),
            np.arange(len(pcm)),
            pcm,
        ).astype(np.int16)
    start = int(sample.start * SAMPLE_RATE)
    end = len(pcm) if sample.end is None else int(sample.end * SAMPLE_RATE)
    if start >= len(pcm) or end > len(pcm) + 1:
        raise ValueError(f"Annotation exceeds recording: {sample.sample_id}")
    mask = np.ones(len(pcm), dtype=bool)
    mask[:start] = False
    mask[end:] = False
    for a, b in sample.exclude:
        mask[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] = False
    pcm = pcm[mask]
    if not len(pcm):
        raise ValueError(f"Empty audio: {sample.sample_id}")
    return pcm


def as_float(pcm):
    return pcm.astype(np.float32) / 32768
