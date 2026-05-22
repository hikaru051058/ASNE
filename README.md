# ASNE - Activation-Steered Neural Encoding

ASNE is an experimental NeuroAI research framework for studying whether controllable steering conditions can systematically alter predicted cortical response patterns in frozen neural encoding models such as TRIBE v2.

The project treats upstream neural-response foundation models as frozen components. ASNE adds experiment definitions, steering-condition orchestration, baseline-versus-steered comparisons, response delta analysis, and reporting tools around those models.

## Research Motivation

Traditional neural encoding models often estimate a response as:

```text
response = f(stimulus)
```

ASNE investigates a state-conditioned formulation:

```text
response = f(stimulus, steering_state)
```

The central question is whether controllable latent steering conditions can produce reproducible, distinguishable, and interpretable changes in predicted cortical-response patterns for the same stimulus.

## What ASNE Does

ASNE is intended to support:

- Steering condition definitions, such as prompt prefixes, soft prompts, activation vectors, representation steering layers, or future LoRA/adapter steering.
- Experiment orchestration over shared stimuli and multiple steering conditions.
- Baseline and steered predicted cortical response comparison.
- Response delta computation between neutral and steered runs.
- ROI- or region-level analysis of predicted response shifts.
- Visualization and report generation for reproducible research review.

## What ASNE Does Not Do

ASNE does not perform mind reading, emotion detection, consciousness simulation, psychological diagnosis, or direct measurement of human thoughts. Its outputs should be interpreted as predicted cortical responses from computational models, not as verified internal states of a person.

Steering conditions are experimental controls. They should not be described as real emotions, intentions, or psychological states.

## High-Level Pipeline

```text
Stimulus
  |
  v
Frozen upstream neural encoding model
  |
  +--> Baseline predicted cortical response
  |
  +--> Steering condition definition
          |
          v
      Steered predicted cortical response
          |
          v
      Response delta computation
          |
          v
      ROI analysis, visualization, and report
```

## Expected MVP

The initial ASNE MVP should include:

1. A frozen TRIBE v2 or similar neural-response model runner.
2. A neutral baseline inference path.
3. A configurable set of steering conditions.
4. A steered inference path.
5. Response delta computation between baseline and steered outputs.
6. ROI-level ranking or summary statistics.
7. Static visualizations and a paper-style experiment report.

TODO: Confirm the exact TRIBE v2 loading, input, output, and ROI metadata APIs before implementing runners or analysis code.

## Mock Experiment Quickstart

Install the ASNE prototype package in editable mode:

```bash
pip install -e .
```

Run a deterministic mock experiment:

```bash
python -m asne.cli run-mock --stimulus "A person walks into a dark alley."
```

Run the same mock experiment with simple matplotlib charts embedded in the report:

```bash
python -m asne.cli run-mock --stimulus "A person walks into a dark alley." --with-plots
```

Run with placeholder mock ROI labels and ROI-level plots:

```bash
python -m asne.cli run-mock --stimulus "A person walks into a dark alley." --with-plots --roi-map configs/mock_rois.example.yaml
```

This command loads the example steering condition config, generates mock predicted cortical responses, computes response deltas for each non-neutral steering condition, and writes JSON outputs plus a markdown report under `outputs/`.

The mock ROI map groups synthetic response indices into placeholder regions for workflow development only. It is not a real cortical parcellation and should not be interpreted as anatomical or psychological evidence.

The mock adapter is only a prototype scaffold. It does not use real TRIBE v2 inference. TODO: replace the mock adapter with a real frozen-model adapter after confirming the TRIBE v2 API, input preprocessing, output format, and ROI metadata.

## Adapter Modes

ASNE currently supports two adapter modes:

- `mock`: default deterministic mock predicted cortical response adapter for local prototype runs.
- `tribev2`: placeholder adapter for future TRIBE v2 integration.

Use mock mode for the current scaffold:

```bash
python -m asne.cli run-mock --adapter mock --stimulus "A person walks into a dark alley."
```

The TRIBE v2 adapter performs a lazy availability check and fails with an actionable message if TRIBE v2 is not installed or `TribeModel` is not importable. It does not download model weights or run real TRIBE v2 inference yet. See [docs/TRIBEV2_INTEGRATION.md](docs/TRIBEV2_INTEGRATION.md).

## Experiment Manifests

Each mock run writes a manifest JSON file under `outputs/manifests/`. The manifest records the experiment ID, UTC timestamp, stimulus text, adapter type, steering config path, optional mock ROI map path, steering condition IDs, and generated output paths.

Manifests are intended to make mock experiments and future frozen TRIBE v2 runs reproducible and auditable. They do not make mock predicted cortical responses scientifically valid; they only document how a run was produced.

## Dictionary Experiments

ASNE dictionary experiments compare text inputs against controlled stimulus-response dictionaries using predicted response similarity. The current default is full neutral-subtracted centroid scoring:

```bash
--signature delta_from_neutral --aggregation centroid --scoring full
```

Top-k and weighted category-sensitive dimension scoring remain available as experimental modes. On `emotion_context_v0`, they did not improve over full scoring; see [docs/DICTIONARY_EXPERIMENTS.md](docs/DICTIONARY_EXPERIMENTS.md).

## Contrast Experiments

ASNE also includes controlled binary contrast dictionaries for testing concrete stimulus axes such as matching versus conflicting information, static scenes versus an entering agent/object, item found versus not found, and unresolved versus handled situations. These are response-similarity experiments, not emotion detection or measurement of an individual person's mental state.

See [docs/CONTRAST_EXPERIMENTS.md](docs/CONTRAST_EXPERIMENTS.md) for the contrast files, baseline-category handling, and run commands.

## ASNE Current Result Snapshot

ASNE is currently most useful as a controlled semantic contrast testing workflow. It compares TRIBE v2 predicted cortical response signatures for paired text/TTS stimuli and asks whether the predicted response signatures are separable.

Current strongest finding: semantic and logical violation contrasts separate better than narrated motion-style contrasts. Temporal segment analysis can expose late predicted response shifts that whole-stimulus mean response may hide, especially for `expected_vs_unexpected_paired`.

| contrast | static top1 | top2 | temporal late | temporal final | majority | recovered |
|---|---:|---:|---:|---:|---:|---:|
| `contradiction_vs_consistency_paired` | 0.83 | 1.00 | 0.83 | 0.83 | 0.83 | 1 |
| `expected_vs_unexpected_paired` | 0.83 | 1.00 | 0.83 | 1.00 | 0.83 | 1 |
| `approach_vs_static_paired` | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 0 |
| `cause_effect_valid_vs_invalid_paired` | 0.83 | 1.00 | 0.83 | 0.67 | 0.67 | 1 |

Safety framing: ASNE compares predicted TRIBE cortical response signatures. It is not measured brain activity, emotion detection, mental-state measurement, diagnosis, or a stable clinical classifier.

Reproduce or verify the current report without rerunning existing TRIBE outputs:

```bash
python scripts/run_asne_semantic_contrast_suite.py --skip-build
```

The current report is written to:

```text
outputs/asne_reports/semantic_contrast_report_v0.md
```

Demo report index:

```text
outputs/asne_reports/index.html
```

The demo index now presents v0.3-lite as the current headline benchmark and keeps v0.2 parcel scoring as a historical prototype milestone.

## GitHub Pages Export

ASNE can export the current static report set into `docs/` for GitHub Pages. This keeps generated report copies separate from private/local outputs while preserving the handwritten documentation already in `docs/`.

Generate or refresh reports, then export:

```bash
python scripts/run_asne_semantic_contrast_suite.py --skip-build
python scripts/generate_asne_demo_index.py
python scripts/export_asne_docs_site.py
```

For the current v0.3 benchmark report, refresh with:

```bash
python scripts/run_asne_v03_benchmark.py --skip-build
python scripts/generate_asne_demo_index.py
python scripts/export_asne_docs_site.py
```

Configure GitHub Pages to serve from:

```text
main /docs
```

## ASNE v0.1 Direction: ROI / Parcel Signatures

ASNE v0.1 adds ROI/parcel-level aggregation as an interpretability layer over predicted cortical response vectors. The raw `20,484`-dimension scoring pipeline remains the benchmark layer for now; parcel aggregation is for analysis and reporting first.

ROI analysis is intended to answer which parcels shift most for a contrast and which parcels show the strongest temporal movement. It requires a parcellation file mapping each cortical response dimension to a parcel label. Tests currently use a small mock parcellation; a real TRIBE/fsaverage5-compatible parcellation should be added only after verifying output-space compatibility and atlas licensing.

## ASNE v0.2 Direction: Parcel-Level Scoring

ASNE v0.2 adds an experimental HCP-MMP parcel feature space for dictionary evaluation. On the frozen semantic contrast suite, parcel-level centroid scoring preserved the current vertex-level benchmark accuracy:

```text
contradiction_vs_consistency_paired: vertex 0.83, parcel 0.83
expected_vs_unexpected_paired:      vertex 0.83, parcel 0.83
approach_vs_static_paired:          vertex 0.50, parcel 0.50
cause_effect_valid_vs_invalid:      vertex 0.83, parcel 0.83
```

This makes parcel-level scoring a viable interpretable experiment path, while the vertex-level benchmark remains the primary reference until larger eval sets confirm stability.

## TTS Backends

For `.txt` stimuli, ASNE can preserve the upstream TRIBE v2 `gTTS` text path or pre-render text to cached audio with local macOS `say`, local Higgs Audio, or OpenAI TTS before calling TRIBE v2 audio mode. This makes it possible to compare preprocessing sensitivity across text-to-audio systems.

See [docs/TTS_BACKENDS.md](docs/TTS_BACKENDS.md) for backend options and commands.

## License

ASNE framework code is released under the MIT License. See [LICENSE](LICENSE).

This repository does not include upstream model weights or third-party assets. TRIBE v2 and any other upstream models, datasets, checkpoints, or media assets retain their original licenses, which may include restrictions such as CC BY-NC or other non-commercial terms. Users are responsible for checking and complying with those licenses before running experiments or publishing results.
