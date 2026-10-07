from jiwer import wer
from rapidfuzz.distance.Levenshtein import distance

from .normalization import normalize_english


def score(reference, prediction):
    ref = normalize_english(reference)
    hyp = normalize_english(prediction)
    if not ref.strip():
        return None
    # CER includes whitespace, as in scripts/eval/metrics.py.
    return {"wer": wer(ref, hyp), "cer": distance(ref, hyp) / len(ref)}
