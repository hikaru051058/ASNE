# TRIBE v2 Integration Plan

ASNE currently runs in mock mode by default. The `TribeV2Adapter` is a placeholder intended to define the boundary for future frozen TRIBE v2 inference without forcing installation, model downloads, or redistribution of upstream artifacts.

## Intended Adapter Boundary

The future adapter should map ASNE inputs into the TRIBE v2 inference workflow:

```text
stimulus/video/text input
  -> TRIBE v2 events dataframe
  -> frozen TRIBE v2 predicted cortical response
  -> ASNE response dict
  -> response delta and mock/real ROI analysis layers
```

For text or video experiments, ASNE should keep steering-condition metadata separate from the upstream model inputs unless a verified steering mechanism is implemented.

## Known Current Issue

Depending on the local checkout and install state, this may fail:

```python
from tribev2 import TribeModel
```

In this repository, TRIBE v2 may exist as a local folder without being installed as an importable package with all runtime dependencies. ASNE therefore imports TRIBE v2 lazily inside `TribeV2Adapter` and raises a clear `AdapterUnavailableError` when the package or `TribeModel` entry point is unavailable.

When working from the ASNE repository root, the outer `./tribev2` checkout can shadow the inner importable package at `./tribev2/tribev2`. Use ASNE's local checkout option to test imports without manually editing `PYTHONPATH`:

```bash
python -m asne.cli check-adapter --adapter tribev2 --tribev2-package-path ./tribev2
```

This command only imports `TribeModel`. It does not call `from_pretrained`, download weights, or run inference.

To check whether model construction can start, explicitly opt in:

```bash
python -m asne.cli check-adapter \
  --adapter tribev2 \
  --tribev2-package-path ./tribev2 \
  --load-model \
  --cache-folder ./cache
```

`--load-model` calls:

```python
TribeModel.from_pretrained(model_name, cache_folder=cache_folder)
```

This may download model artifacts, initialize PyTorch and feature-extraction dependencies, and require enough disk, memory, and network access. It still does not run stimulus prediction and does not add ASNE steering into TRIBE v2.

Manual alternatives:

```bash
PYTHONPATH=./tribev2 python -c "from tribev2 import TribeModel; print(TribeModel)"
python -m pip install -e ./tribev2 --no-deps
```

Use `--no-deps` only when the required TRIBE v2 dependencies are already installed. A plain editable install may install or modify heavy dependencies such as PyTorch.

## Expected Future Mapping

The local TRIBE v2 checkout exposes the inference wrapper in `tribev2/tribev2/demo_utils.py` and exports it from `tribev2/tribev2/__init__.py`:

```python
from tribev2 import TribeModel
```

The documented smoke API is:

```python
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
```

For text files, the local API also exposes:

```python
df = model.get_events_dataframe(text_path="path/to/stimulus.txt")
preds, segments = model.predict(events=df)
```

Findings from the local API:

- `get_events_dataframe` accepts exactly one of `text_path`, `audio_path`, or `video_path`.
- Supported video suffixes include `.mp4`, `.avi`, `.mkv`, `.mov`, and `.webm`.
- Text input is supported through `text_path`, but it may trigger upstream text-to-speech/audio-event preprocessing.
- `predict(events, verbose=True)` returns `preds, segments`.
- The docstring describes `preds` as shape `(n_kept_segments, n_vertices)`.
- ASNE's first smoke path reduces 2D predictions to a 1D response vector using `mean_over_time_axis_0` and records the original raw shape.

Current high-level mapping:

- Load a frozen upstream model with `TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")`.
- Convert one local video or `.txt` text file into a TRIBE events dataframe with `model.get_events_dataframe(video_path=stimulus_path)` or `model.get_events_dataframe(text_path=stimulus_path)`.
- Run TRIBE prediction with `model.predict(events=df, verbose=False)`.
- Convert the output into an ASNE response dictionary with:
  - `model`
  - `stimulus_path`
  - `condition_id`
  - `response`
  - raw output shape, segment count, aggregation method, and warning metadata

Run the first one-video smoke path:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m asne.cli run-tribev2-smoke \
  --stimulus-path path/to/short.mp4 \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache
```

This command does not run ASNE steering conditions and does not run multiple conditions. It writes one JSON file under `outputs/tribev2_smoke/`.

Run the first text smoke path:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m asne.cli run-tribev2-smoke \
  --stimulus-path path/to/stimulus.txt \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache
```

Text smoke runs still use TRIBE's upstream preprocessing, so they may initialize audio/text dependencies even though ASNE is only passing a `.txt` file.

## TTS Backend Note

TRIBE v2 supports `text_path`, `audio_path`, and `video_path` inputs. Its upstream `text_path` helper uses `gTTS` to synthesize audio and then runs audio/text event extraction. ASNE exposes this as `--tts-backend tribev2_gtts` and also supports pre-rendering `.txt` stimuli to cached audio with `--tts-backend macos_say`, `--tts-backend higgs_audio`, or `--tts-backend openai` before calling TRIBE `audio_path`.

This lets ASNE compare whether predicted response similarity rankings are sensitive to the text-to-audio backend. See [TTS_BACKENDS.md](TTS_BACKENDS.md).

## Licensing Warning

TRIBE v2 model code, checkpoints, model weights, datasets, and demo assets may use separate upstream licenses. These artifacts should not be redistributed inside ASNE unless their license explicitly permits it. ASNE should document the upstream source, version, and license for every real TRIBE v2 run.

## Troubleshooting

- If `from tribev2 import TribeModel` fails from the ASNE root but succeeds with `PYTHONPATH=./tribev2`, pass `--tribev2-package-path ./tribev2`.
- If import emits `neuralset` warnings, note them but distinguish import-time warnings from failed model loading.
- If `--load-model` fails with Hugging Face, checkpoint, network, or cache errors, retry only after confirming upstream model access and license terms.
- If dependencies are already installed but the local checkout is not importable, a local editable install may help:

```bash
python -m pip install -e ./tribev2 --no-deps
```

Avoid plain `pip install -e ./tribev2` unless you intend to let pip install or change heavy dependencies such as PyTorch.

## TODO Checklist

- Confirm supported Python version and dependency set for the selected TRIBE v2 release.
- Confirm whether `from tribev2 import TribeModel` is the stable import path.
- Keep `--tribev2-package-path ./tribev2` as the local checkout path for diagnostics and future adapter work.
- Confirm model loading call and whether it downloads weights.
- Confirm accepted stimulus inputs: video, audio, text, or precomputed events dataframe.
- Define how ASNE steering conditions are represented without overclaiming psychological meaning.
- Confirm output tensor shape and units.
- Add real cortical parcellation or ROI metadata only when supplied by a verified upstream source.
- Update manifest fields to capture TRIBE v2 model version, checkpoint source, cache path, and license.
- Add integration tests that use a tiny fixture or mocked TRIBE object without downloading model weights.
