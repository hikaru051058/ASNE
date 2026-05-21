# ASNE TTS Backends

ASNE can route `.txt` stimuli through different text-to-speech backends before TRIBE v2 audio/event extraction. This is intended for controlled comparison of preprocessing choices, not for emotion detection, diagnosis, or measurement of an individual person's mental state.

## Backends

| backend | behavior | network | notes |
| --- | --- | --- | --- |
| `tribev2_gtts` | Uses upstream TRIBE v2 `TextToEvents`, which calls `gTTS` internally. | May require access to Google Translate TTS endpoints. | This preserves the original TRIBE demo text path. |
| `macos_say` | Uses local macOS `say`, converts AIFF to WAV with `afconvert`, then calls TRIBE `audio_path`. | No network. | Useful first local baseline on Apple Silicon. |
| `higgs_audio` | Uses Boson AI Higgs Audio v2 locally, then calls TRIBE `audio_path`. | No API calls after model download. | Experimental local neural TTS backend; large model and separate license. |
| `openai` | Uses OpenAI TTS to create cached audio, then calls TRIBE `audio_path`. | Requires API access. | Useful for a more natural generated voice baseline. |

Generated audio is cached under `./cache/asne_tts/<backend>/` by default. The cache key includes text and TTS settings, so changing voice/model/rate creates a separate audio file.

## Smoke Examples

Original upstream TRIBE text path:

```bash
python -m asne.cli run-tribev2-smoke \
  --stimulus-path data/stimuli/text/dark_alley_neutral.txt \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --tts-backend tribev2_gtts
```

Local macOS TTS path:

```bash
python -m asne.cli run-tribev2-smoke \
  --stimulus-path data/stimuli/text/dark_alley_neutral.txt \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --tts-backend macos_say \
  --tts-voice Samantha \
  --tts-rate 180
```

OpenAI TTS path:

```bash
OPENAI_API_KEY=... python -m asne.cli run-tribev2-smoke \
  --stimulus-path data/stimuli/text/dark_alley_neutral.txt \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --tts-backend openai \
  --openai-tts-model gpt-4o-mini-tts \
  --openai-tts-voice alloy \
  --openai-tts-format mp3
```

Higgs Audio local neural TTS path:

```bash
python -m asne.cli run-tribev2-smoke \
  --stimulus-path data/stimuli/text/dark_alley_neutral.txt \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --tts-backend higgs_audio \
  --higgs-device mps
```

If MPS fails in the upstream Higgs Audio stack, retry with:

```bash
--higgs-device cpu
```

Higgs Audio requires the upstream Boson AI package and model artifacts. ASNE does not redistribute those files. Check the Higgs Audio repository and model license before using it in published work:

- https://github.com/boson-ai/higgs-audio
- https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base
- https://huggingface.co/bosonai/higgs-audio-v2-tokenizer

Install the optional OpenAI Python package if needed:

```bash
python -m pip install -e '.[openai]'
```

## Dictionary Comparison

For controlled comparisons, build separate dictionary outputs per TTS backend. ASNE automatically appends the TTS backend/settings to non-default dictionary output names, so the generated signatures do not overwrite each other.

Example:

```bash
python scripts/run_asne_dictionary.py \
  --dictionary data/stimuli/dictionaries/contrasts/contradiction_vs_consistency.json \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --tts-backend macos_say
```

Then evaluate against the matching dictionary index:

```bash
python scripts/evaluate_asne_dictionary.py \
  --dictionary outputs/asne_dictionaries/contradiction_vs_consistency_tts_macos_say_samantha_180/dictionary_index.json \
  --eval data/stimuli/evals/contrasts/contradiction_vs_consistency_eval.json \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --signature delta_from_neutral \
  --aggregation centroid \
  --scoring full \
  --tts-backend macos_say
```

Use the same `--tts-backend` and voice/model settings for dictionary construction and evaluation when comparing a backend end to end.

For Higgs Audio dictionary runs, use a separate backend-specific dictionary output:

```bash
python scripts/run_asne_dictionary.py \
  --dictionary data/stimuli/dictionaries/contrasts/contradiction_vs_consistency.json \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --tts-backend higgs_audio \
  --higgs-device mps
```

## Interpretation

If different TTS backends produce different rankings, treat that as a preprocessing sensitivity finding. It does not mean one backend reveals a real mental state. It means the predicted cortical response signature depends on how text was rendered into audio/events before TRIBE v2 inference.
