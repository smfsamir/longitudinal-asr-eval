# Models

Run one model per process. Install only its backend, preferably in a separate environment for NeMo, ESPnet, Qwen3, or Omnilingual.

| IDs in `configs/paper.json` | Installation |
| --- | --- |
| `whisper-v1`, `whisper-v2`, `whisper-v3` | `pip install -e '.[whisper]'` |
| `wavlm`, `mms`, `seamless`, `qwen2` | `pip install -e '.[transformers]'` |
| `owsm-v1`, `owsm-v2`, `owsm-v3.1`, `owsm-v4` | `pip install -e '.[owsm]'` |
| `qwen3` | `pip install -e '.[qwen3]'` |
| `omni` | `pip install -e '.[omnilingual]'` |
| `chirp3` | `pip install -e '.[google]'` |
| `canary` | NeMo installation below |
| `deepspeech` | Coqui installation below |

For a checkpoint change, copy `configs/paper.json`, edit the model's `checkpoint` (and `id`, `name`, `release_date` for a new comparison), then pass `--config`. Hugging Face backends accept an optional `revision` commit. The optional `provider` field selects a bundled logo for analysis plots (`mozilla`, `microsoft`, `openai`, `meta`, `cmu`, `qwen`, `nvidia`, or `google`); omit it for other providers. Use a new output directory when changing checkpoints, options, manifests, code, or installed packages; resumption checks these inputs.

`--device auto` selects CUDA when available, otherwise CPU. Use `--device mps` only with a backend that supports it. Large models generally need a GPU. Model libraries and weights are loaded only when their model is selected.

## Canary

Install NeMo:

```sh
pip install 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@6294bc7708ce67522b92f5e9b6917ea0b2e23429'
pip install -e .
```

See the [Canary model card](https://huggingface.co/nvidia/canary-qwen-2.5b) for platform requirements. The adapter uses an English transcription prompt and a 128-token limit.

## Coqui

The `deepspeech` configuration uses the [Coqui English v1.0.0 huge-vocabulary release](https://github.com/coqui-ai/STT-models/releases/tag/english%2Fcoqui%2Fv1.0.0-huge-vocab). Download `model.tflite` and `huge-vocabulary.scorer` into `models/deepspeech/`. Install a compatible Coqui `stt` runtime, or set `STT_BINARY` to a working `stt` executable in another environment. No model assets are downloaded at import time.

## Chirp

Use Google Application Default Credentials and set `GOOGLE_PROJECT_ID`. Set `GOOGLE_BUCKET` for recordings longer than 60 seconds. These requests use the paid Speech-to-Text API; long recordings are temporarily uploaded to that bucket and deleted afterward. The `first_result_only` option selects the first recognition result. Set it to `false` in a separate configuration to concatenate all results.

## Decoding

OWSM uses beam size 5, legacy `<en>` tags for v1/v2, and nonoverlapping 30-second chunks. MMS selects the English adapter. SeamlessM4T uses 10-second chunks and 512 output tokens. Qwen2 uses an English transcription prompt and uncapped maximum length; set `max_new_tokens` to impose a limit. Qwen3 keeps language detection and the 2,048-token limit.

Omnilingual uses `omniASR_LLM_7B`, without context examples. Its [pipeline](https://github.com/facebookresearch/omnilingual-asr) limits this model to 40 seconds. The adapter splits longer recordings into 40-second chunks, including the 57.216-second L1 suitcase control.

`run.json` records the installed versions, configuration, input hashes, and code hashes for each new run. Pin revisions or use local checkpoints when freezing a new experiment. Remote Chirp outputs may change over time.
