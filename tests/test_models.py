"""Adapter contracts without downloading checkpoints."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from longitudinal_asr.models import load_model


def torch_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))


@pytest.mark.parametrize(
    "checkpoint,language", [("espnet/owsm_v1", "<en>"), ("espnet/owsm_v4_medium_1B", "<eng>")]
)
def test_owsm_checkpoint_language_and_amplitude(monkeypatch, checkpoint, language):
    torch_stub(monkeypatch)
    calls, sizes = [], []

    class Speech:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append(kwargs)
            return cls()

        def __call__(self, audio, **kwargs):
            sizes.append(len(audio))
            assert audio.dtype == np.float64
            assert audio[0] == pytest.approx(32767 / 32768)
            assert kwargs["lang_sym"] == language
            return [(None, "recognized", None)]

    monkeypatch.setitem(
        sys.modules, "espnet2.bin.s2t_inference", SimpleNamespace(Speech2Text=Speech)
    )
    decode = load_model({"backend": "owsm", "checkpoint": checkpoint}, device="cpu")
    assert decode(np.full(31 * 16000, 32767, dtype=np.int16)) == "recognized recognized"
    assert sizes == [30 * 16000, 16000]
    assert calls[0]["model_tag"] == checkpoint
    assert calls[0]["lang_sym"] == language


def test_whisper_checkpoint_and_pcm(monkeypatch):
    calls = []

    class Whisper:
        def __init__(self, checkpoint, **kwargs):
            calls.append((checkpoint, kwargs))

        def transcribe(self, audio, **kwargs):
            assert audio.dtype == np.float32
            assert audio[0] == -0.5
            assert kwargs["language"] == "en"
            return iter([SimpleNamespace(text="hello"), SimpleNamespace(text="world")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Whisper))
    decoder = load_model(
        {"backend": "whisper", "checkpoint": "large-v2", "revision": "pinned"}, "cpu"
    )
    assert decoder(np.array([-16384], dtype=np.int16)) == "hello world"
    assert calls == [("large-v2", {"device": "cpu", "compute_type": "int8", "revision": "pinned"})]


def test_canary_returns_text_and_preserves_pcm(monkeypatch):
    torch_stub(monkeypatch)

    class SALM:
        audio_locator_tag = "<audio>"
        tokenizer = SimpleNamespace(ids_to_text=lambda ids: "recognized" if ids == [1] else None)

        @classmethod
        def from_pretrained(cls, checkpoint):
            assert checkpoint == "nvidia/canary-qwen-2.5b"
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, prompts, max_new_tokens):
            pcm, rate = sf.read(prompts[0][0]["audio"][0], dtype="int16")
            assert rate == 16000
            assert pcm[0] == 16384
            assert max_new_tokens == 128
            return [SimpleNamespace(cpu=lambda: [1])]

    monkeypatch.setitem(
        sys.modules, "nemo.collections.speechlm2.models", SimpleNamespace(SALM=SALM)
    )
    decode = load_model({"backend": "canary", "checkpoint": "nvidia/canary-qwen-2.5b"}, "cpu")
    assert decode(np.full(16000, 16384, dtype=np.int16)) == "recognized"


def test_omni_preserves_long_recording(monkeypatch):
    lengths = []

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["model_card"] == "omniASR_LLM_7B"

        def transcribe(self, paths, **kwargs):
            lengths.append(sf.info(paths[0]).frames)
            assert kwargs["lang"] == ["eng_Latn"]
            return ["text"]

    monkeypatch.setitem(
        sys.modules,
        "omnilingual_asr.models.inference.pipeline",
        SimpleNamespace(ASRInferencePipeline=Pipeline),
    )
    decoder = load_model({"backend": "omni", "checkpoint": "omniASR_LLM_7B"}, "cpu")
    assert decoder(np.zeros(57 * 16000, dtype=np.int16)) == "text text"
    assert lengths == [40 * 16000, 17 * 16000]
