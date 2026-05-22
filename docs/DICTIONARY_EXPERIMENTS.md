# ASNE Dictionary Experiments

ASNE dictionary experiments compare new text inputs against a controlled stimulus-response dictionary using predicted cortical response similarity. These comparisons are not emotion detection, diagnosis, or measurement of an individual person's mental state.

## Current Default

For `emotion_context_v0`, the default comparison setting is:

```bash
--signature delta_from_neutral --aggregation centroid --scoring full
```

This compares the full neutral-subtracted predicted response vector against category centroids. On the small `emotion_context_v0` evaluation set, this setting outperformed the experimental category-sensitive dimension modes:

```text
full scoring: 7/10 top-1 = 0.70
topk 500: 5/10 top-1 = 0.50
topk 1000: 6/10 top-1 = 0.60
weighted: 6/10 top-1 = 0.60
```

Experiment note: On `emotion_context_v0`, full neutral-subtracted centroid scoring outperformed top-k and weighted category-sensitive scoring. Top-k and weighted modes may require larger dictionaries or ROI-level aggregation before they become useful. They remain available as experimental modes only.

## Experimental Modes

The following modes are available for analysis but are not current defaults:

- `--scoring topk`: selects the dimensions with the largest absolute category delta from neutral for each category.
- `--scoring weighted`: weights dimensions by the absolute category delta from neutral.
- `--scoring binary_axis`: uses a signed projection along the raw centroid axis for a two-category contrast.
- `--scoring centroid_raw`: compares raw mean-response category centroids directly.

Use these modes for ablation studies, not as the primary reported setting until they improve on a held-out evaluation set.

## Frozen v0 Semantic Contrast Suite

Feature growth is paused for the current v0 semantic contrast benchmark. The frozen suite contains:

- `contradiction_vs_consistency_paired`
- `expected_vs_unexpected_paired`
- `approach_vs_static_paired`
- `cause_effect_valid_vs_invalid_paired`

Recommended static scoring for binary text/TTS contrasts:

```bash
--signature mean_response --aggregation centroid --scoring centroid_raw
```

Temporal evaluation uses saved raw segment predictions and an existing static eval summary:

```bash
python scripts/evaluate_asne_temporal_contrast.py \
  --dictionary outputs/asne_dictionaries/<contrast>_tts_macos_say_samantha_180/dictionary_index.json \
  --eval data/stimuli/evals/contrasts/<contrast>_eval.json \
  --eval-summary outputs/asne_evals/contrasts/<contrast>/<timestamp>_summary.json
```

Verify or regenerate the full frozen suite report:

```bash
python scripts/run_asne_semantic_contrast_suite.py --skip-build
```

Report path:

```text
outputs/asne_reports/semantic_contrast_report_v0.md
```

## ASNE v0.1 ROI / Parcel Aggregation

ASNE v0.1 adds ROI/parcel-level aggregation to make predicted response signatures easier to inspect. This layer groups raw cortical dimensions into parcel labels and reports parcel-level contrast deltas and temporal movement.

The raw `20,484`-dimension scoring layer remains the frozen v0 benchmark. ROI/parcel aggregation is an additional reporting and interpretability layer, not a replacement for validated scoring.

Example ROI report command:

```bash
python scripts/analyze_asne_roi_contrast.py \
  --dictionary outputs/asne_dictionaries/<contrast>_tts_macos_say_samantha_180/dictionary_index.json \
  --parcellation data/parcellations/<parcellation>.csv \
  --output outputs/asne_roi_reports/<contrast>/<timestamp>_roi_report.md
```

Current status: ROI tests use mock parcellations. A real TRIBE/fsaverage5-compatible parcellation should be added after checking model output-space compatibility and atlas redistribution terms.

TRIBE v2 outputs are documented locally as fsaverage5 with `10,242` vertices per hemisphere, left hemisphere followed by right hemisphere. ASNE includes a source-aware generator:

```bash
python scripts/create_fsaverage5_parcellation.py --dry-run
```

HCP-MMP can be generated through TRIBE/MNE if external atlas fetching is explicitly allowed. Schaefer fsaverage5 labels are not generated unless a verified source is available; ASNE should not fake scientific parcel labels.

## HCP-MMP ROI Reports

The current real ROI/parcellation layer uses:

```text
data/parcellations/fsaverage5_hcp_mmp.csv
```

This file maps TRIBE's `20,484` fsaverage5 cortical vertices to `362` HCP-MMP parcels.

Current ROI report paths:

- `outputs/asne_roi_reports/contradiction_vs_consistency_paired/roi_report.md`
- `outputs/asne_roi_reports/expected_vs_unexpected_paired/roi_report.md`
- `outputs/asne_roi_reports/approach_vs_static_paired/roi_report.md`
- `outputs/asne_roi_reports/cause_effect_valid_vs_invalid_paired/roi_report.md`

These ROI reports summarize predicted TRIBE response signatures. They are not measured brain activity, diagnosis, or measurement of a person's mental state. In the text/TTS pipeline, temporal movement summaries can be dominated by auditory parcels, so contrast-delta parcels are currently the more useful ROI view for semantic interpretation.

## ASNE v0.2 Parcel-Level Scoring Result

ASNE v0.2 tests whether HCP-MMP parcel vectors can be used directly for scoring instead of only reporting. The experiment compares:

- Vertex benchmark: `feature_space=vertex`, `signature=mean_response`, `aggregation=centroid`, `scoring=centroid_raw`
- Parcel scoring: `feature_space=parcel`, `signature=mean_response`, `aggregation=centroid`, `scoring=centroid_raw`

Report path:

```text
outputs/asne_reports/vertex_vs_parcel_scoring_v0.md
```

Current result on the frozen semantic suite:

| contrast | vertex top1 | parcel top1 |
|---|---:|---:|
| `contradiction_vs_consistency_paired` | 0.83 | 0.83 |
| `expected_vs_unexpected_paired` | 0.83 | 0.83 |
| `approach_vs_static_paired` | 0.50 | 0.50 |
| `cause_effect_valid_vs_invalid_paired` | 0.83 | 0.83 |

Conclusion: HCP-MMP parcel-level centroid scoring preserves the current vertex-level benchmark accuracy on the small frozen suite. It is now a viable interpretable scoring option for experiments, but the raw vertex benchmark remains primary until larger evaluation sets confirm stability.

## ASNE v0.3-lite Contradiction Instability

The larger `semantic_contrast_v0_3_lite` contradiction/consistency eval did not preserve the small v0.2 contradiction result. Using the 20-example v0.3 dictionary and 10-example held-out eval:

| feature space | top1 | top2 | mean rank | consistent | contradictory |
|---|---:|---:|---:|---:|---:|
| vertex | 4/10 = 0.40 | 1.00 | 1.60 | 3/5 | 1/5 |
| HCP-MMP parcel | 5/10 = 0.50 | 1.00 | 1.50 | 4/5 | 1/5 |

Temporal segment scoring also did not recover the contrast:

```text
mean=0.50 early=0.40 late=0.50 final=0.40 majority=0.50
```

Margin analysis shows the expected category is usually rank 2 with small score gaps, especially for contradictory examples:

```text
consistent_information: failures=1/5 avg_gap=0.001139
contradictory_information: failures=4/5 avg_gap=0.005316
```

A stricter held-out subset with stronger concrete color/count/object differences also did not recover the category:

| feature space | top1 | top2 | consistent | contradictory |
|---|---:|---:|---:|---:|
| vertex | 4/10 = 0.40 | 1.00 | 4/5 | 0/5 |
| HCP-MMP parcel | 4/10 = 0.40 | 1.00 | 4/5 | 0/5 |

The stricter subset temporal result was:

```text
mean=0.40 early=0.70 late=0.40 final=0.40 majority=0.40
```

Interpretation: the v0.2 `5/6` contradiction result was prototype-level and should not be treated as stable. The larger eval exposes an unstable binary boundary with a tendency for contradictory examples to flip toward `consistent_information`. Stronger mismatch wording alone did not fix this. Contradiction-style stimuli should be redesigned more deeply or deprioritized in favor of semantic violation contrasts that remain stable on larger evals.

## ASNE v0.3-lite Semantic Contrast Results

The v0.3-lite benchmark is the current stability check for semantic text/TTS contrasts. It uses `10` dictionary pairs and `5` held-out eval pairs per contrast where available.

Report paths:

```text
outputs/asne_reports/semantic_contrast_benchmark_v03.md
outputs/asne_reports/semantic_contrast_benchmark_v03.html
```

Current v0.3-lite static results:

| contrast | classification | v0.2 top1 | vertex top1 | parcel top1 | vertex top2 | parcel top2 | temporal late | temporal final | temporal majority |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `contradiction_vs_consistency_paired` | weak/deprioritized | 0.83 | 0.40 | 0.50 | 1.00 | 1.00 | 0.50 | 0.40 | 0.50 |
| `expected_vs_unexpected_paired` | stable | 0.83 | 0.80 | 0.90 | 1.00 | 1.00 | 0.90 | 0.80 | 0.80 |
| `cause_effect_valid_vs_invalid_paired` | stable | 0.83 | 0.70 | 0.80 | 1.00 | 1.00 | 0.50 | 0.40 | 0.70 |
| `approach_vs_static_paired` | pending | 0.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Current interpretation:

- `expected_vs_unexpected_paired` is the strongest surviving v0.3 semantic contrast. Parcel scoring improves top-1 from `0.80` to `0.90`, and late temporal scoring reaches `0.90`.
- `cause_effect_valid_vs_invalid_paired` remains viable. Vertex scoring reaches `0.70`, parcel scoring reaches `0.80`, and majority temporal scoring reaches `0.70`, although late/final segment scoring is weaker.
- `contradiction_vs_consistency_paired` is retained as a failed/unstable contrast case.
- `approach_vs_static_paired` remains pending in v0.3 and is not a priority for text/TTS; prior smaller runs suggest motion-style contrasts may require video-native stimuli.

Conclusion: v0.3 shows that not all semantic contrasts survive larger evaluation. Expected/unexpected remains stable, cause/effect is stable/promising with parcel scoring, contradiction is unstable, and approach/static is not yet evaluated in v0.3.

## Binary Contrast Scoring

Binary contrast experiments use two concrete stimulus categories, such as a baseline category and a contrast category. In this setting, `delta_from_neutral` or `delta_from_baseline` can make the baseline category compete as a zero vector, which is not ideal for a two-way contrast.

For `contradiction_vs_consistency_paired`, the current paired macOS TTS evaluation produced:

```text
full/delta:   4/6 top-1, consistent_information 1/3, contradictory_information 3/3
binary_axis:  4/6 top-1, consistent_information 1/3, contradictory_information 3/3
centroid_raw: 5/6 top-1, consistent_information 2/3, contradictory_information 3/3
```

Current recommendation: use `centroid_raw` for binary contrast experiments until more evidence suggests a better default:

```bash
--signature mean_response --aggregation centroid --scoring centroid_raw
```

`binary_axis` remains useful as a diagnostic because it removes the zero-vector baseline issue and reports a signed contrast score. However, it did not improve this paired contradiction contrast. The observed margins are small, so this should be treated as prototype predicted response-similarity signal, not a stable classifier.

## Temporal Segment-Level Evaluation

ASNE now supports temporal segment-level diagnostics over saved raw TRIBE segment predictions. These diagnostics compare each segment against category centroids and report early, late, final-segment, and majority segment winners. This can expose predicted response shifts that are hidden by whole-stimulus mean response pooling.

Current paired contrast temporal results using existing macOS TTS outputs:

| contrast | mean_response_acc | early_acc | late_acc | final_segment_acc | majority_segment_acc | avg_switch_count | recovered_mean_failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `contradiction_vs_consistency_paired` | 0.83 | 0.50 | 0.83 | 0.83 | 0.83 | 1.33 | 1 |
| `approach_vs_static_paired` | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| `expected_vs_unexpected_paired` | 0.83 | 0.50 | 0.83 | 1.00 | 0.83 | 0.50 | 1 |
| `cause_effect_valid_vs_invalid_paired` | 0.83 | 0.50 | 0.83 | 0.67 | 0.67 | 1.00 | 1 |

Experiment note: On `expected_vs_unexpected_paired`, final-segment scoring reached `6/6`, recovering one mean-response failure. This suggests temporal analysis is useful for inspecting within-stimulus predicted response shifts. The effect did not generalize to `approach_vs_static_paired`, which remains weak in the current text/TTS pipeline and may require video stimuli or a different stimulus design.

Run temporal evaluation from an existing dictionary evaluation summary:

```bash
python scripts/evaluate_asne_temporal_contrast.py \
  --dictionary outputs/asne_dictionaries/expected_vs_unexpected_paired_tts_macos_say_samantha_180/dictionary_index.json \
  --eval data/stimuli/evals/contrasts/expected_vs_unexpected_paired_eval.json \
  --eval-summary outputs/asne_evals/contrasts/expected_vs_unexpected_paired/<timestamp>_summary.json
```

## v1 Dictionary Expansion Plan

The next dictionary version should expand coverage while keeping labels as stimulus categories, not clinical or diagnostic claims.

Targets:

- `10` dictionary examples per category.
- `5` held-out evaluation examples per category.
- A stimulus validator that checks schema, category counts, duplicate text, and prohibited wording.
- Stimuli should avoid direct category words, such as using the literal category label inside the text.
- Stimuli should avoid clinical, diagnostic, surveillance, hiring, education scoring, law enforcement, or medical decision-making wording.
- Evaluation examples should not duplicate dictionary examples.

TODO: Add `emotion_context_v1.json`, `emotion_context_eval_v1.json`, and a validation command before running larger TRIBE batches.
