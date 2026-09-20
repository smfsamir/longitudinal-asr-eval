import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from longitudinal_asr.audio import load_audio
from longitudinal_asr.datasets import (
    DIALECTS,
    allstar,
    balanced_openslr,
    l2_arctic,
    read_grid,
)
from longitudinal_asr.manifest import Sample, read_manifest, write_manifest
from longitudinal_asr.metrics import score
from longitudinal_asr.normalization import normalize_english
from longitudinal_asr.saa_text import clean_text


def grid_file(path, tiers, end=2.0):
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {end}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for i, (name, intervals) in enumerate(tiers.items(), 1):
        lines += [
            f"    item [{i}]:",
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            "        xmin = 0",
            f"        xmax = {end}",
            f"        intervals: size = {len(intervals)}",
        ]
        for j, (a, b, text) in enumerate(intervals, 1):
            lines += [
                f"        intervals [{j}]:",
                f"            xmin = {a}",
                f"            xmax = {b}",
                f'            text = "{text}"',
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def sample(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.arange(32000, dtype=np.int16), 16000)
    return Sample("d", "s", str(path), "please call stella", "English", "speaker")


def test_audio_cropping_and_overlapping_exclusions(tmp_path):
    s = sample(tmp_path)
    clipped = replace(s, start=0.25, end=1.75, exclude=[[0.5, 1], [0.75, 1.25]])
    pcm = load_audio(clipped)
    assert len(pcm) == 12000
    np.testing.assert_array_equal(pcm[:4000], np.arange(4000, 8000, dtype=np.int16))
    with pytest.raises(ValueError, match="exceeds"):
        load_audio(replace(s, end=3))


def test_mono_and_original_resampler(tmp_path):
    audio = np.column_stack([np.arange(100, dtype=np.int16), np.arange(100, dtype=np.int16) * 2])
    p = tmp_path / "stereo.wav"
    sf.write(p, audio, 8000)
    s = Sample("d", "s", str(p), "hello", "a", "s")
    mono = audio.mean(axis=1).astype(np.int16)
    expected = np.interp(np.linspace(0, 100, 200), np.arange(100), mono).astype(np.int16)
    np.testing.assert_array_equal(load_audio(s), expected)


def test_manifest_roundtrip_and_duplicates(tmp_path):
    s = sample(tmp_path)
    path = tmp_path / "manifest.jsonl"
    write_manifest([s], path)
    assert read_manifest(path) == [s]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)


def test_allstar_sentences_and_nws_exclusion(tmp_path):
    sample(tmp_path)
    tg = tmp_path / "a.TextGrid"
    grid_file(tg, {"sentences": [(0, 1, "Please call Stella."), (1, 2, "Bring the bags.")]})
    index = tmp_path / "index.csv"
    with index.open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio", "textgrid", "speaker_id", "accent", "task", "tier"]
        )
        writer.writeheader()
        for task in ["LPP", "NWS"]:
            writer.writerow(
                dict(
                    audio="audio.wav",
                    textgrid="a.TextGrid",
                    speaker_id="s",
                    accent="English",
                    task=task,
                    tier="sentences",
                )
            )
    samples = list(allstar(index))
    assert len(samples) == 2
    assert samples[1].start == 1
    assert samples[0].dataset == "ALLSTAR-LPP"
    assert samples[1].text == "Bring the bags."
    index.write_text(index.read_text().replace("sentences", "words"))
    with pytest.raises(ValueError, match="sentence-level"):
        list(allstar(index))


def test_allstar_out_of_bounds(tmp_path):
    sample(tmp_path)
    grid_file(tmp_path / "a.TextGrid", {"sentences": [(0, 3, "hello")]}, end=3)
    index = tmp_path / "index.csv"
    index.write_text(
        "audio,textgrid,speaker_id,accent,task,tier\naudio.wav,a.TextGrid,s,English,HT1,sentences\n"
    )
    with pytest.raises(ValueError, match="Invalid"):
        list(allstar(index))


def native_allstar(tmp_path, stem="ALL_073_M_CCT_ENG_DHR", intervals=None, rate=44100):
    folder = tmp_path / "2936" / "recordings"
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / f"{stem}.wav", np.zeros(rate * 2, dtype=np.int16), rate)
    grid_file(
        folder / f"{stem}.TextGrid",
        {
            "utt": intervals or [(0, 0.2, ""), (0.2, 1.8, "Please call Stella."), (1.8, 2, "")],
            "Speaker - word": [(0, 2, "Please")],
            "Speaker - phone": [(0, 2, "P")],
        },
    )
    return folder


def test_allstar_native_download_and_stable_ids(tmp_path):
    folder = native_allstar(tmp_path)
    native_allstar(tmp_path, "ALL_049_F_ENG_ENG_HT1", rate=22050)
    native_allstar(tmp_path, "ALL_003_M_RUN_ENG_LPP")
    # Non-English speech and NWS are excluded even without a paired TextGrid.
    (folder / "ALL_073_M_CCT_CCT_DHR.wav").touch()
    (folder / "ALL_073_M_CCT_ENG_NWS.wav").touch()
    samples = list(allstar(tmp_path))
    assert [(s.accent, s.gender) for s in samples] == [
        ("Nkore", "M"),
        ("English", "F"),
        ("Cantonese", "M"),
    ]
    assert samples[-1].speaker_id == "073"
    assert samples[-1].sample_id == "ALL_073_M_CCT_ENG_DHR:1"
    assert samples[-1].text == "Please call Stella."
    assert [s.sample_id for s in allstar(folder)] == [s.sample_id for s in samples]
    assert len(load_audio(samples[-1])) == 25600


def test_allstar_missing_and_duplicate_partners(tmp_path):
    folder = native_allstar(tmp_path)
    annotation = next(folder.glob("*.TextGrid"))
    duplicate = tmp_path / annotation.name
    duplicate.write_bytes(annotation.read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        list(allstar(tmp_path))
    annotation.unlink()
    duplicate.unlink()
    with pytest.raises(ValueError, match="partner"):
        list(allstar(tmp_path))


def test_allstar_truncated_audio_requires_explicit_exclusion(tmp_path):
    native_allstar(
        tmp_path,
        intervals=[
            (0, 1, "Complete sentence."),
            (1, 2.5, "Truncated sentence."),
            (2.5, 3, ""),
            (3, 4, "Missing sentence."),
        ],
    )
    with pytest.raises(ValueError, match="exceeds"):
        list(allstar(tmp_path))
    with pytest.warns(UserWarning, match="skipped") as warnings:
        samples = list(allstar(tmp_path, skip_invalid_intervals=True))
    assert len(warnings) == 2
    assert len(samples) == 1
    assert samples[0].text == "Complete sentence."


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_textgrid_encodings(tmp_path, encoding):
    path = tmp_path / "a.TextGrid"
    grid_file(path, {"utt": [(0, 2, "café")]})
    path.write_bytes(path.read_text().encode(encoding))
    assert read_grid(path)["utt"][0].text == "café"


def test_suitcase_first_50_not_repeated_chunks(tmp_path):
    phones = [(i / 50, (i + 1) / 50, "sil" if i == 0 else "AH") for i in range(60)]
    tg = tmp_path / "suitcase_corpus/annotation/ABA.TextGrid"
    grid_file(
        tg, {"phones": phones, "words": [(0, 0.5, "hello"), (0.5, 1, "there"), (1, 1.2, "later")]}
    )
    samples = list(l2_arctic(tmp_path, tmp_path / "cache", spontaneous=True))
    assert len(samples) == 1
    assert samples[0].end == 1
    assert samples[0].text == "hello there"
    assert samples[0].accent == "Arabic"


def test_openslr_balancing_is_seeded():
    samples = [
        Sample("Openslr83", f"{a}/{g}/{i}", "unused.wav", "one two three four", a, str(i), g)
        for a in DIALECTS.values()
        for g in (["male"] if a == "Irish English" else ["male", "female"])
        for i in range(60)
    ]
    first = balanced_openslr(samples)
    assert len(first) == 550
    assert first == balanced_openslr(samples[::-1])
    assert first != balanced_openslr(samples, seed=2)
    with pytest.raises(ValueError):
        balanced_openslr(samples[:3])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Dr. Smith can't bring twenty-one bags.", "doctor smith can not bring 21 bags"),
        ("[noise] um colour", "color"),
        ("<laugh>", ""),
        ("Please call Stella.", "please call stella"),
    ],
)
def test_normalizer(text, expected):
    assert normalize_english(text).strip() == expected


def test_metrics_keep_insertions_and_empty_predictions():
    assert score("a b", "a b c d e")["wer"] == 1.5
    assert score("hello", "")["wer"] == 1
    assert score("<noise>", "anything") is None
    assert score("a b", "ab")["cer"] == pytest.approx(1 / 3)


def test_saa_lexical_repairs():
    assert (
        clean_text("pleasecall stellaw ask-rep her 2 bring")
        == "please call stella ask ask her to bring"
    )
    assert clean_text("blue cheese-del and") == "blue and"


def test_packaged_control_is_only_reference():
    text = (Path(__file__).parents[1] / "data/suitcase/L1Suitcase.txt").read_text()
    assert text.startswith("There is a man and a woman")
    assert "whisper" not in text
    assert len(text.split()) == 85
