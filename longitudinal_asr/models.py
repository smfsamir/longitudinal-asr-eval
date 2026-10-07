"""Lazy model adapters. Checkpoints and decoding options live in the run config."""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE, as_float


@contextmanager
def wav_file(pcm):
    with NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, pcm, SAMPLE_RATE, subtype="PCM_16")
        yield f.name


def device_name(requested):
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(spec, device="auto"):
    backend, checkpoint = spec["backend"], spec["checkpoint"]
    options = spec.get("options", {})
    revision = {"revision": spec["revision"]} if spec.get("revision") else {}
    if backend == "replay":
        raise ValueError("Replay is handled by the runner, not an ASR model")
    if backend == "google":
        return Google(checkpoint, options)
    if backend == "stt":
        return Coqui(checkpoint, options)
    if backend == "omni":
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        pipeline = ASRInferencePipeline(model_card=checkpoint, device=device_name(device))

        def transcribe(pcm):
            size = int(options.get("chunk_seconds", 40) * SAMPLE_RATE)
            texts = []
            for start in range(0, len(pcm), size):
                with wav_file(pcm[start : start + size]) as path:
                    texts.append(
                        pipeline.transcribe(
                            [path], lang=[options.get("language", "eng_Latn")], batch_size=1
                        )[0]
                    )
            return " ".join(texts)

        return transcribe
    if backend == "whisper":
        from faster_whisper import WhisperModel

        # CTranslate2 has no MPS backend. "auto" mirrors upstream CUDA/CPU choice.
        if device == "auto":
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        model = WhisperModel(
            checkpoint,
            device=device,
            compute_type=options.get("compute_type", "float16" if device == "cuda" else "int8"),
            **revision,
        )

        def transcribe(pcm):
            segments, _ = model.transcribe(
                as_float(pcm),
                language=options.get("language", "en"),
                beam_size=options.get("beam_size", 5),
            )
            return " ".join(s.text for s in segments)

        return transcribe
    device = device_name(device)
    import torch

    if backend == "ctc":
        from transformers import AutoModelForCTC, AutoProcessor

        processor = AutoProcessor.from_pretrained(checkpoint, **revision)
        model = AutoModelForCTC.from_pretrained(checkpoint, **revision).to(device).eval()
        if "language" in options:
            processor.tokenizer.set_target_lang(options["language"])
            model.load_adapter(options["language"], **revision)

        def transcribe(pcm):
            inputs = processor(as_float(pcm), sampling_rate=SAMPLE_RATE, return_tensors="pt").to(
                device
            )
            with torch.inference_mode():
                ids = model(**inputs).logits.argmax(dim=-1)
            return processor.batch_decode(ids)[0]

        return transcribe
    if backend == "owsm":
        from espnet2.bin.s2t_inference import Speech2Text

        legacy = checkpoint in {"espnet/owsm_v1", "espnet/owsm_v2", "espnet/owsm_v2_ebranchformer"}
        language = "<en>" if legacy else "<eng>"
        model = Speech2Text.from_pretrained(
            model_tag=checkpoint,
            device=device,
            beam_size=options.get("beam_size", 5),
            ctc_weight=0.0,
            maxlenratio=0.0,
            lang_sym=language,
            task_sym="<asr>",
            predict_time=False,
        )

        def transcribe(pcm):
            audio = pcm.astype(np.float64) / 32768
            size = int(options.get("chunk_seconds", 30) * SAMPLE_RATE)
            with torch.inference_mode():
                return " ".join(
                    model(
                        audio[i : i + size], lang_sym=language, task_sym="<asr>", predict_time=False
                    )[0][-2]
                    for i in range(0, len(audio), size)
                )

        return transcribe
    if backend == "seamless":
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        processor = AutoProcessor.from_pretrained(checkpoint, **revision)
        model = SeamlessM4Tv2Model.from_pretrained(checkpoint, **revision).to(device).eval()

        def transcribe(pcm):
            audio, texts = as_float(pcm), []
            size = int(options.get("chunk_seconds", 10) * SAMPLE_RATE)
            with torch.inference_mode():
                for i in range(0, len(audio), size):
                    inputs = processor(
                        audio=audio[i : i + size],
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                        src_lang="eng",
                    ).to(device)
                    tokens = model.generate(
                        **inputs,
                        tgt_lang="eng",
                        generate_speech=False,
                        max_new_tokens=options.get("max_new_tokens", 512),
                    )
                    texts.append(processor.batch_decode(tokens[0], skip_special_tokens=True)[0])
            return " ".join(texts)

        return transcribe
    if backend == "qwen2":
        from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

        processor = Qwen2AudioProcessor.from_pretrained(checkpoint, **revision)
        model = (
            Qwen2AudioForConditionalGeneration.from_pretrained(checkpoint, **revision)
            .to(device)
            .eval()
        )
        prompt = (
            "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the exact English speech in this audio:"
        )

        def transcribe(pcm):
            inputs = processor(
                text=prompt, audio=as_float(pcm), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).to(device)
            generation = (
                {"max_new_tokens": options["max_new_tokens"]}
                if "max_new_tokens" in options
                else {"max_length": np.inf}
            )
            with torch.inference_mode():
                ids = model.generate(**inputs, **generation)
            return processor.batch_decode(
                ids[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return transcribe
    if backend == "qwen3":
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )
        from qwen_asr.inference.utils import parse_asr_output

        dtype = getattr(torch, options.get("dtype", "bfloat16"))
        processor = Qwen3ASRProcessor.from_pretrained(checkpoint, **revision)
        model = (
            Qwen3ASRForConditionalGeneration.from_pretrained(checkpoint, dtype=dtype, **revision)
            .to(device)
            .eval()
        )

        def transcribe(pcm):
            audio = as_float(pcm)
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": audio}]},
            ]
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=prompt,
                audio=audio,
                return_tensors="pt",
                padding=True,
                sampling_rate=SAMPLE_RATE,
            ).to(device, dtype=dtype)
            with torch.inference_mode():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=options.get("max_new_tokens", 2048),
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            if hasattr(ids, "sequences"):
                ids = ids.sequences
            decoded = processor.batch_decode(
                ids[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return parse_asr_output(decoded)[1]

        return transcribe
    if backend == "canary":
        from nemo.collections.speechlm2.models import SALM

        model = SALM.from_pretrained(checkpoint).to(device).eval()

        def transcribe(pcm):
            with wav_file(pcm) as path, torch.inference_mode():
                ids = model.generate(
                    prompts=[
                        [
                            {
                                "role": "user",
                                "content": f"Transcribe the speech in English: {model.audio_locator_tag}",
                                "audio": [path],
                            }
                        ]
                    ],
                    max_new_tokens=options.get("max_new_tokens", 128),
                )
            return model.tokenizer.ids_to_text(ids[0].cpu())

        return transcribe
    raise ValueError(f"Unknown backend: {backend}")


class Coqui:
    def __init__(self, checkpoint, options):
        self.checkpoint, self.scorer = checkpoint, options["scorer"]
        for path in [checkpoint, self.scorer]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing Coqui model asset: {path}; see docs/models.md")
        self.binary = options.get("binary", os.environ.get("STT_BINARY"))
        if not self.binary:
            from stt import Model

            self.model = Model(checkpoint)
            self.model.enableExternalScorer(self.scorer)

    def __call__(self, pcm):
        if not self.binary:
            return self.model.stt(pcm)
        with wav_file(pcm) as path:
            result = subprocess.run(
                [self.binary, "--model", self.checkpoint, "--scorer", self.scorer, "--audio", path],
                capture_output=True,
                text=True,
                check=True,
            )
        lines = result.stdout.strip().splitlines()
        return lines[-1] if lines else ""


class Google:
    def __init__(self, checkpoint, options):
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechClient

        self.options, self.checkpoint = options, checkpoint
        region = options.get("region", "us")
        project = options.get("project") or os.environ.get("GOOGLE_PROJECT_ID")
        if not project:
            raise ValueError("Set GOOGLE_PROJECT_ID for Chirp")
        self.recognizer = f"projects/{project}/locations/{region}/recognizers/_"
        self.client = SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{region}-speech.googleapis.com")
        )

    def __call__(self, pcm):
        import uuid

        from google.cloud.speech_v2.types import cloud_speech as cs

        config = cs.RecognitionConfig(
            auto_decoding_config=cs.AutoDetectDecodingConfig(),
            language_codes=[self.options.get("language", "en-US")],
            model=self.checkpoint,
        )
        with wav_file(pcm) as path:
            content = Path(path).read_bytes()
        duration = len(pcm) / SAMPLE_RATE
        if duration <= 60:
            response = self.client.recognize(
                request=cs.RecognizeRequest(
                    recognizer=self.recognizer, config=config, content=content
                )
            )
        else:
            from google.cloud import storage

            bucket_name = self.options.get("bucket") or os.environ.get("GOOGLE_BUCKET")
            if not bucket_name:
                raise ValueError("Set GOOGLE_BUCKET for recordings longer than 60 seconds")
            blob = storage.Client().bucket(bucket_name).blob(f"asr-eval/{uuid.uuid4()}.wav")
            blob.upload_from_string(content, content_type="audio/wav")
            uri = f"gs://{bucket_name}/{blob.name}"
            try:
                operation = self.client.batch_recognize(
                    request=cs.BatchRecognizeRequest(
                        recognizer=self.recognizer,
                        config=config,
                        files=[cs.BatchRecognizeFileMetadata(uri=uri)],
                        recognition_output_config=cs.RecognitionOutputConfig(
                            inline_response_config=cs.InlineOutputConfig()
                        ),
                    )
                )
                result = operation.result(timeout=max(120, int(duration * 2))).results[uri]
                if result.error.code:
                    raise RuntimeError(result.error.message)
                response = result.transcript
            finally:
                blob.delete()
        results = list(response.results)
        if self.options.get("first_result_only", True):
            results = results[:1]  # The original notebook uses only results[0].
        return " ".join(r.alternatives[0].transcript for r in results if r.alternatives)
