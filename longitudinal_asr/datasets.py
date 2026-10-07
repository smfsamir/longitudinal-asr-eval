"""Dataset preparation, adapted from KoelLabs/ML's ASR loaders."""

import csv
import io
import json
import math
import random
import re
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd
import soundfile as sf
import textgrids

from .manifest import Sample
from .saa_text import clean_text

SPEAKERS = json.loads((Path(__file__).parent / "speakers.json").read_text())
DIALECTS = dict(
    ir="Irish English",
    mi="Midlands English",
    no="Northern English",
    sc="Scottish English",
    so="Southern English",
    we="Welsh English",
)


def read_grid(source):
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    grid = textgrids.TextGrid()
    if data.startswith(b"ooBinaryFile"):
        grid.parse(data)
        return grid
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    # Repair the multiline labels handled by the original L2-Arctic loader.
    patched = []
    for line in data.decode(encoding).split("\n"):
        if patched and (
            line in ['oʊ, ə, " ', '" ', 's" '] or (line.count('"') == 1 and line.endswith('" '))
        ):
            patched[-1] = patched[-1].strip() + line
        else:
            patched.append(line)
    grid.parse("\n".join(patched).encode("utf-8"))
    return grid


def tier(grid, name):
    if name not in grid:
        raise ValueError(f"Missing TextGrid tier {name!r}; available: {list(grid)}")
    return [(float(x.xmin), float(x.xmax), str(x.text)) for x in grid[name]]


def l2_arctic(root, cache, spontaneous=False, max_phones=50):
    """Read the nested v5 archive or its extracted speaker directories."""
    root, cache = Path(root), Path(cache)
    speakers = ["suitcase_corpus"] if spontaneous else list(SPEAKERS["L2ARCTIC"])
    outer = zipfile.ZipFile(root) if root.is_file() else None
    legacy_id = 0
    try:
        for split in speakers:
            archive = None
            if outer:
                archive = zipfile.ZipFile(io.BytesIO(outer.read(f"{split}.zip")))
                files = [
                    n
                    for n in archive.namelist()
                    if n.startswith(f"{split}/annotation/") and n.endswith(".TextGrid")
                ]
            else:
                files = sorted(
                    str(p.relative_to(root))
                    for p in (root / split / "annotation").glob("*.TextGrid")
                )
            if not files:
                raise ValueError(f"No expert annotations found for {split}")
            try:
                for name in files:
                    stem = Path(name).stem
                    grid = read_grid(archive.read(name) if archive else root / name)
                    words = tier(grid, "words")
                    end = None
                    if spontaneous:
                        # Includes nonempty silence labels in the count, as upstream does.
                        phones = [p for p in tier(grid, "phones") if p[2]]
                        if not phones:
                            raise ValueError(f"No phone annotations: {name}")
                        end = int(phones[:max_phones][-1][1] * 16000) / 16000
                        words = [w for w in words if w[1] <= end]
                    text = " ".join(w[2] for w in words if w[2])
                    audio_name = f"{split}/wav/{stem}.wav"
                    if archive:
                        audio = cache / audio_name
                        audio.parent.mkdir(parents=True, exist_ok=True)
                        content = archive.read(audio_name)
                        if not audio.exists() or audio.read_bytes() != content:
                            audio.write_bytes(content)
                    else:
                        audio = root / audio_name
                    speaker = stem.upper() if spontaneous else split
                    meta = SPEAKERS["L2ARCTIC"][speaker]
                    yield Sample(
                        "L2-Arctic Spontaneous" if spontaneous else "L2-Arctic",
                        f"{split}/{stem}",
                        str(audio),
                        text,
                        meta["native-language"],
                        speaker,
                        meta["gender"],
                        end=end,
                        legacy_id=str(legacy_id),
                    )
                    legacy_id += 1
            finally:
                if archive:
                    archive.close()
    finally:
        if outer:
            outer.close()


def suitcase_control(audio, transcript):
    return Sample(
        "L2-Arctic Spontaneous",
        "L1Suitcase",
        str(audio),
        Path(transcript).read_text().strip(),
        "USA",
        "L1Suitcase",
        legacy_id="22",
    )


def l1_arctic(root):
    root = Path(root)
    for speaker, meta in SPEAKERS["L1ARCTIC"].items():
        folder = root / f"cmu_us_{speaker}_arctic"
        if not folder.exists():
            continue
        textfile = folder / "etc/txt.done.data"
        transcripts = dict(re.findall(r'\(\s*(\S+)\s+"(.+)"\s*\)', textfile.read_text()))
        for audio in sorted((folder / "wav").glob("*.wav")):
            if audio.stem in transcripts:
                yield Sample(
                    "L1-Arctic",
                    f"{speaker}/{audio.stem}",
                    str(audio),
                    transcripts[audio.stem],
                    meta["accent"],
                    speaker,
                    meta["sex"],
                )


def openslr83(root):
    root = Path(root)
    with (root / "line_index_all.csv").open() as f:
        for i, line in enumerate(f):
            # The reference can itself contain commas.
            speaker, filename, text = line.rstrip().split(", ", 2)
            code = filename.split("_")[0]
            yield Sample(
                "Openslr83",
                filename,
                str(root / "audios" / f"{filename}.wav"),
                text,
                DIALECTS[code[:2]],
                speaker,
                "male" if code.endswith("m") else "female",
                legacy_id=str(i),
            )


def balanced_openslr(samples, seed=0, per_gender=50):
    groups = defaultdict(list)
    for sample in samples:
        if len(sample.text.split()) >= 4:
            groups[sample.accent, sample.gender].append(sample)
    rng = random.Random(seed)
    result = []
    for accent in DIALECTS.values():
        for gender in ["male"] if accent == "Irish English" else ["female", "male"]:
            pool = sorted(groups[accent, gender], key=lambda s: s.sample_id)
            if len(pool) < per_gender:
                raise ValueError(
                    f"Need {per_gender} samples for {accent}/{gender}; got {len(pool)}"
                )
            result.extend(rng.sample(pool, per_gender))
    return sorted(result, key=lambda s: s.sample_id)


def select_samples(samples, selection):
    with open(selection) as f:
        rows = list(csv.DictReader(f))
    available = {(s.dataset, s.sample_id): s for s in samples}
    selected = []
    for row in rows:
        key = row["dataset"], row["sample_id"]
        if key not in available:
            raise ValueError(f"Selected sample is missing: {key}")
        selected.append(replace(available[key], legacy_id=row.get("legacy_id", "")))
    return selected


SAA_REMAP = {
    "poonchi1": "pahari1",
    "japanese1a": "japanese1",
    "kirgiz1": "kyrgyz2",
    **{f"sinhalese{i}": f"sinhala{i}" for i in range(1, 5)},
    "sa_a1": "sa'a1",
}
SAA_REMOVE = {
    "chatter",
    "chatter2",
    "selfintroduction",
    "self-talk",
    "experimenter_comments_taketwo",
}


def speech_accent(root, metadata=None, min_speakers=20, max_phones=50):
    root = Path(root)
    metadata = (
        Path(metadata) if metadata else root / "excel_spreadsheets/speakers-1 november 2023.xlsx"
    )
    df = pd.read_excel(metadata, dtype={"notes": str}).fillna("")
    if df.speech_sample.duplicated().any():
        raise ValueError("Duplicate SAA speech_sample metadata")
    info = df.set_index("speech_sample").to_dict("index")
    samples = []
    legacy_id = 0
    for path in sorted((root / "textgrids").glob("*/*.TextGrid")):
        grid = read_grid(path)
        if "MAU" in grid:
            phone_tier, word_tier = "MAU", "ORT"
        elif "phones" in grid:
            phone_tier, word_tier = "phones", "words"
        else:
            continue
        phones = [
            p for p in tier(grid, phone_tier) if p[2] and p[2] not in {"sil", "<p:>", "(...)"}
        ]
        if not phones:
            raise ValueError(f"No phone annotations: {path}")
        end = int(phones[:max_phones][-1][1] * 16000) / 16000
        # Upstream includes the word intersecting the crop boundary.
        words = [w for w in tier(grid, word_tier) if int(w[0] * 16000) <= int(end * 16000)]
        exclude = [[a, b] for a, b, w in words if w.strip() in SAA_REMOVE]
        text = clean_text(
            " ".join(w.strip() for _, _, w in words if w and w not in SAA_REMOVE | {"sil", "sp"})
        )
        sample_id = SAA_REMAP.get(path.stem, path.stem)
        meta = info[f"{sample_id}.mp3"]
        samples.append(
            Sample(
                "SAA",
                sample_id,
                str(path.with_suffix(".wav")),
                text,
                str(meta["native_language"]).capitalize(),
                str(meta["speakerid"]),
                str(meta["gender"]),
                end=end,
                exclude=exclude,
                legacy_id=str(legacy_id),
            )
        )
        legacy_id += 1
    counts = Counter((s.accent, s.speaker_id) for s in samples)
    accent_counts = Counter(accent for accent, _ in counts)
    return [s for s in samples if accent_counts[s.accent] >= min_speakers]


ALLSTAR_ACCENTS = {
    "ENG": "English",
    "CCT": "Cantonese",
    "CMN": "Mandarin",
    "CSP": "Chinese Singaporean",
    "CTW": "Chinese Taiwanese",
    "FAR": "Farsi",
    "FRA": "French",
    "GER": "German",
    "GIS": "Gishu",
    "GRE": "Greek",
    "GUJ": "Gujarati",
    "HEB": "Hebrew",
    "HIN": "Hindi",
    "IND": "Indonesian",
    "JPN": "Japanese",
    "KOR": "Korean",
    "PBR": "Brazilian Portuguese",
    "RUN": "Nkore",
    "RUS": "Russian",
    "SHS": "Spanish Heritage Speaker",
    "SPA": "Spanish",
    "TUR": "Turkish",
    "VIE": "Vietnamese",
}


def allstar_recordings(root):
    """Index ALL_###_G_LLL_TTT_CCC files from an extracted ALLSSTAR download."""
    files = defaultdict(dict)
    pattern = re.compile(r"ALL_(\d+)_([MF])_([A-Z]{3})_([A-Z]{3})_([A-Z0-9]+)")
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in {".wav", ".textgrid"}:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        match = pattern.fullmatch(path.stem.upper())
        if not match:
            raise ValueError(f"Unrecognized ALLSTAR filename: {path.name}; use an index CSV")
        speaker, gender, native, language, task = match.groups()
        if language != "ENG" or task == "NWS":
            continue
        key = path.stem.upper()
        if suffix in files[key]:
            raise ValueError(f"Duplicate ALLSTAR recording file: {path.name}")
        files[key][suffix] = path
    for key, pair in sorted(files.items()):
        if set(pair) != {".wav", ".textgrid"}:
            raise ValueError(f"Missing ALLSTAR WAV/TextGrid partner: {key}")
        speaker, gender, native, language, task = pattern.fullmatch(key).groups()
        if native not in ALLSTAR_ACCENTS:
            raise ValueError(f"Unknown ALLSTAR native-language code: {native}")
        yield dict(
            audio=str(pair[".wav"].relative_to(root)),
            textgrid=str(pair[".textgrid"].relative_to(root)),
            speaker_id=speaker,
            gender=gender,
            accent=ALLSTAR_ACCENTS[native],
            task=task,
            tier="utt",
            recording_id=key,
        )


def allstar(index, skip_invalid_intervals=False):
    """Read the extracted corpus or an explicit recording index.

    Required CSV columns: audio,textgrid,speaker_id,accent,task,tier.
    Paths are relative to the index. task is LPP, DHR, HT1, HT2, or NWS.
    """
    index = Path(index)
    if index.is_dir():
        base, rows = index, allstar_recordings(index)
    else:
        base = index.parent
        with index.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    seen = set()
    speakers = {}
    for row in rows:
        if row.get("task", "").upper() == "NWS":
            continue
        for field in ["audio", "textgrid", "speaker_id", "accent", "task", "tier"]:
            if not row.get(field, "").strip():
                raise ValueError(f"Missing ALLSTAR {field}")
        if re.search(r"\b(words?|phones?|phonemes?|mau|ort)\b", row["tier"].lower()):
            raise ValueError("ALLSTAR requires sentence-level annotations")
        speaker = row["speaker_id"]
        if speaker in speakers and speakers[speaker] != row["accent"]:
            raise ValueError(f"Inconsistent accent for ALLSTAR speaker {speaker}")
        speakers[speaker] = row["accent"]
        task = row["task"].upper()
        if task not in {"LPP", "DHR", "HT1", "HT2"}:
            raise ValueError(f"Unknown ALLSTAR task: {task}")
        audio, gridpath = base / row["audio"], base / row["textgrid"]
        duration = sf.info(audio).duration
        intervals = tier(read_grid(gridpath), row["tier"])
        previous_end = 0.0
        count = 0
        for i, (a, b, text) in enumerate(intervals):
            if not math.isfinite(a + b) or a < previous_end - 1e-6 or a < 0 or b <= a:
                raise ValueError(f"Invalid/overlapping ALLSTAR interval {i}: {gridpath}")
            previous_end = b
            if not text.strip() or text.strip().lower() in {"sil", "sp", "<sil>"}:
                continue
            if a >= duration or b > duration + 0.02:
                message = (
                    f"Invalid ALLSTAR interval {i}: {gridpath.name}: "
                    f"{a:.3f}–{b:.3f}s exceeds {duration:.3f}s audio"
                )
                if not skip_invalid_intervals:
                    raise ValueError(message + "; use --skip-invalid-intervals to exclude it")
                warnings.warn(message + "; skipped", stacklevel=2)
                continue
            recording_id = row.get("recording_id", Path(row["audio"]).as_posix())
            sample_id = f"{recording_id}:{i}"
            if sample_id in seen:
                raise ValueError(f"Duplicate ALLSTAR interval: {sample_id}")
            seen.add(sample_id)
            count += 1
            yield Sample(
                f"ALLSTAR-{task}",
                sample_id,
                str(audio),
                text.strip(),
                row["accent"],
                row["speaker_id"],
                row.get("gender", ""),
                start=a,
                end=min(b, duration),
            )
        if not count:
            raise ValueError(f"No sentence annotations: {gridpath}")


def validate_samples(samples):
    """Validate references and timestamps before downloading any models."""
    from .normalization import normalize_english

    seen, audio_info, counts = set(), {}, Counter()
    for s in samples:
        key = s.dataset, s.sample_id
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        if s.audio not in audio_info:
            audio_info[s.audio] = sf.info(s.audio)
        duration = audio_info[s.audio].duration
        if s.start >= duration or (s.end is not None and s.end > duration + 1 / 16000):
            raise ValueError(f"Annotation exceeds audio duration: {key}")
        if not normalize_english(s.text).strip():
            raise ValueError(f"Empty normalized reference: {key}")
        counts[s.dataset, s.accent] += 1
    if not seen:
        raise ValueError("No samples")
    return [{"dataset": d, "accent": a, "samples": n} for (d, a), n in sorted(counts.items())]
