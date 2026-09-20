# Data

Obtain the corpora from their providers. Keep downloads under `data/corpora/` or pass another path with `--root`.

| Corpus | Input |
| --- | --- |
| [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) | `l2arctic_release_v5.0.zip`, containing speaker ZIPs and `suitcase_corpus.zip` |
| [CMU ARCTIC](http://www.festvox.org/cmu_arctic/) | `CMU_ARCTIC/cmu_us_<speaker>_arctic/{wav,etc}` |
| [Speech Accent Archive](https://accent.gmu.edu/) | Expert TextGrids with adjacent WAVs under `textgrids/*/`; speaker spreadsheet under `excel_spreadsheets/` |
| [OpenSLR 83](https://www.openslr.org/83/) | `line_index_all.csv` and `audios/*.wav` |
| [ALLSSTAR](https://speechbox.linguistics.northwestern.edu/ALLSSTARcentral/#!/recordings) | Extracted WAV and TextGrid download folders |

```sh
asr-eval prepare l2-arctic --root data/corpora/l2arctic_release_v5.0.zip \
  --output data/manifests/l2-arctic.jsonl
asr-eval prepare suitcase --root data/corpora/l2arctic_release_v5.0.zip \
  --control-audio data/suitcase/L1Suitcase.wav --control-text data/suitcase/L1Suitcase.txt \
  --output data/manifests/suitcase.jsonl
asr-eval prepare l1-arctic --root data/corpora/CMU_ARCTIC \
  --selection data/selections/l1_arctic.csv --accents USA --output data/manifests/l1-arctic.jsonl
asr-eval prepare saa --root data/corpora/SpeechAccentArchive \
  --output data/manifests/saa.jsonl
asr-eval prepare openslr83 --root data/corpora/OpenSLR83 \
  --selection data/selections/openslr83.csv --output data/manifests/openslr83.jsonl
asr-eval prepare allstar --root data/corpora/ALLSTAR --skip-invalid-intervals \
  --output data/manifests/allstar.jsonl
asr-eval merge data/manifests/l2-arctic.jsonl data/manifests/suitcase.jsonl \
  data/manifests/l1-arctic.jsonl data/manifests/saa.jsonl \
  data/manifests/openslr83.jsonl data/manifests/allstar.jsonl \
  --output data/manifests/all.jsonl
asr-eval validate data/manifests/all.jsonl
```

L2 preparation extracts WAVs to `data/cache/l2-arctic/`; extracted speaker directories are also accepted. SAA accepts `--metadata` for a different spreadsheet location. Only annotated recordings are included. SAA and suitcase use the first 50 annotated phones. Scripted L2 uses full expert-annotated utterances. The added L1 suitcase recording is included with its reference transcript.

The OpenSLR selection has 306 samples (51 per dialect). To draw a 550-sample balanced subset, replace `--selection …` with `--balanced --seed 0`. Save the resulting manifest and reuse it for every model. The supplied CMU selection has 918 samples; the analysis uses its 408 `USA` samples as the scripted L2 control.

## ALLSTAR

Request the English task recordings and annotations from ALLSSTAR Central, then extract the downloaded ZIP under `data/corpora/ALLSTAR/`. Nested download folders are accepted.

The loader reads the `utt` tier and derives speaker, gender, native language, and task from the [documented filenames](https://speechbox.linguistics.northwestern.edu/assets/allsstar/ALLSSTAR-Quick-Overview.pdf), e.g. `ALL_073_M_CCT_ENG_DHR.wav`. Only English `LPP`, `DHR`, `HT1`, and `HT2` recordings are included; NWS and non-English tasks are excluded. `ENG` maps to `English` and `RUN` to `Nkore`.

Preparation checks WAV/TextGrid pairs, duplicate recordings, speaker metadata, sentence tiers, interval order, and audio bounds. Some annotations extend beyond the recording. `--skip-invalid-intervals` reports and excludes those sentences; without it, preparation stops. Blank intervals are omitted.

For renamed files or different tier names, pass a CSV as `--root` instead:

```csv
audio,textgrid,speaker_id,accent,task,tier,gender
wav/s01.wav,annotations/s01.TextGrid,s01,English,HT1,utt,F
```

Paths are relative to the CSV. Word and phone tiers are rejected. After inference, analysis excludes an utterance only if every model's WER exceeds 1; incomplete model coverage fails this check.

## Manifest

Each JSONL row contains `dataset`, `sample_id`, `audio`, `text`, `accent`, and `speaker_id`. Optional fields are `gender`, `start`, `end`, `exclude`, and `legacy_id`. Times are seconds; `exclude` lists intervals removed from the audio. Relative audio paths resolve from the manifest directory. Sample IDs must be unique within each dataset.
