# ASNE Semantic Contrast Report v0

## Overview

ASNE is a TRIBE-backed in-silico stimulus-response analysis system that compares predicted cortical response signatures for controlled stimuli.

## Method

- Text/TTS stimuli are generated from controlled paired narratives.
- TRIBE v2 produces predicted cortical responses.
- ASNE stores raw segment predictions with shape `[segments, 20484]` when available.
- Static binary contrast evaluation uses `mean_response [20484]` with `centroid_raw` comparison.
- Temporal evaluation compares segment-level responses to category centroids and reports early, late, final, and majority segment winners.

## Static Contrast Results

| contrast | best/default scoring | top1 accuracy | top2 accuracy | per-category accuracy | summary path |
|---|---|---:|---:|---|---|
| `contradiction_vs_consistency_paired` | `mean_response/centroid/centroid_raw` | 0.83 | 1.00 | consistent_information: 2/3; contradictory_information: 3/3 | `outputs/asne_evals/contrasts/contradiction_vs_consistency_paired/20260521T012445Z_summary.json` |
| `expected_vs_unexpected_paired` | `mean_response/centroid/centroid_raw` | 0.83 | 1.00 | expected_outcome: 2/3; unexpected_outcome: 3/3 | `outputs/asne_evals/contrasts/expected_vs_unexpected_paired/20260521T012803Z_summary.json` |
| `approach_vs_static_paired` | `mean_response/centroid/centroid_raw` | 0.50 | 1.00 | static_scene: 1/3; approaching_agent: 2/3 | `outputs/asne_evals/contrasts/approach_vs_static_paired/20260521T000101Z_summary.json` |
| `cause_effect_valid_vs_invalid_paired` | `mean_response/centroid/centroid_raw` | 0.83 | 1.00 | valid_cause_effect: 3/3; invalid_cause_effect: 2/3 | `outputs/asne_evals/contrasts/cause_effect_valid_vs_invalid_paired/20260521T021339Z_summary.json` |

## Temporal Contrast Results

| contrast | mean_response_acc | early_acc | late_acc | final_segment_acc | majority_segment_acc | avg_switch_count | recovered_mean_failures | temporal summary path |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `contradiction_vs_consistency_paired` | 0.83 | 0.50 | 0.83 | 0.83 | 0.83 | 1.33 | 1 | `outputs/asne_temporal_evals/contradiction_vs_consistency_paired/20260521T015158Z_temporal_summary.json` |
| `expected_vs_unexpected_paired` | 0.83 | 0.50 | 0.83 | 1.00 | 0.83 | 0.50 | 1 | `outputs/asne_temporal_evals/expected_vs_unexpected_paired/20260521T015158Z_temporal_summary.json` |
| `approach_vs_static_paired` | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | `outputs/asne_temporal_evals/approach_vs_static_paired/20260521T015158Z_temporal_summary.json` |
| `cause_effect_valid_vs_invalid_paired` | 0.83 | 0.50 | 0.83 | 0.67 | 0.67 | 1.00 | 1 | `outputs/asne_temporal_evals/cause_effect_valid_vs_invalid_paired/20260521T022151Z_temporal_summary.json` |

## ROI / Parcel-Level Predicted Response Summary

ROI summaries use HCP-MMP parcel aggregation over TRIBE's `20,484` fsaverage5 vertices when `data/parcellations/fsaverage5_hcp_mmp.csv` is used. Temporal movement is often dominated by auditory parcels such as LBelt, PBelt, A1, and A4 because the current pipeline uses text-to-speech audio. For semantic interpretation, contrast-delta parcels are currently more useful than raw temporal movement parcels.

| contrast | ROI report path | top contrast-delta parcels | top late-minus-early parcels |
|---|---|---|---|
| `contradiction_vs_consistency_paired` | `outputs/asne_roi_reports/contradiction_vs_consistency_paired/roi_report.md` | 1. `rh_a5` (0.056104)<br>2. `lh_a5` (0.046817)<br>3. `rh_pbelt` (0.045383)<br>4. `rh_stsdp` (0.045214)<br>5. `lh_stsdp` (0.044308) | 1. `rh_lbelt` (0.410976)<br>2. `rh_pbelt` (0.382879)<br>3. `lh_lbelt` (0.377971)<br>4. `lh_a1` (0.325217)<br>5. `lh_pbelt` (0.307274) |
| `expected_vs_unexpected_paired` | `outputs/asne_roi_reports/expected_vs_unexpected_paired/roi_report.md` | 1. `rh_pos2` (0.085680)<br>2. `lh_pos2` (0.073250)<br>3. `rh_a32pr` (0.066615)<br>4. `rh_pfm` (0.065230)<br>5. `rh_9_46d` (0.063540) | 1. `rh_lbelt` (0.394459)<br>2. `lh_lbelt` (0.366629)<br>3. `rh_pbelt` (0.344196)<br>4. `lh_a1` (0.309769)<br>5. `lh_pbelt` (0.288618) |
| `approach_vs_static_paired` | `outputs/asne_roi_reports/approach_vs_static_paired/roi_report.md` | 1. `lh_a5` (0.038568)<br>2. `rh_a5` (0.034325)<br>3. `lh_a4` (0.029272)<br>4. `lh_vmv2` (0.027005)<br>5. `lh_v3cd` (0.026829) | 1. `rh_lbelt` (0.419540)<br>2. `rh_pbelt` (0.380894)<br>3. `lh_lbelt` (0.380366)<br>4. `lh_a1` (0.321541)<br>5. `lh_pbelt` (0.309302) |
| `cause_effect_valid_vs_invalid_paired` | `outputs/asne_roi_reports/cause_effect_valid_vs_invalid_paired/roi_report.md` | 1. `rh_pcv` (0.108673)<br>2. `rh_pos1` (0.091479)<br>3. `rh_7m` (0.090292)<br>4. `rh_pos2` (0.085859)<br>5. `lh_pcv` (0.085271) | 1. `rh_lbelt` (0.397385)<br>2. `lh_lbelt` (0.371433)<br>3. `rh_pbelt` (0.351172)<br>4. `lh_a1` (0.308079)<br>5. `lh_pbelt` (0.301566) |

## Main Findings

Semantic and logical text contrasts separate better than narrated motion-style contrasts in the current text/TTS ASNE pipeline. `expected_vs_unexpected_paired` showed the strongest temporal result, with final-segment accuracy reaching `1.00`. `approach_vs_static_paired` stayed weak across static and temporal views, suggesting it may require video-native stimuli or a different stimulus design. HCP-MMP ROI reports now summarize which parcels contribute most to predicted contrast deltas; temporal movement parcels should be interpreted cautiously because they can reflect TTS/audio processing dynamics.

## Limitations

- Evaluation sets are tiny and should be treated as smoke-scale experiments.
- Current runs use text/TTS rather than native video stimuli.
- TRIBE outputs are predicted cortical responses, not measured neural activity.
- Some examples have small margins, and ASNE is not a stable classifier yet.
- Results are predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

## Next Recommended Work

- Add more semantic contrasts and increase held-out examples per contrast.
- Test video-native contrasts separately from narrated text/TTS contrasts.
- Refine ROI interpretation and separate semantic contrast deltas from TTS/audio-driven temporal movement.
- Consider ROI-level scoring or ROI-level temporal normalization after the current vertex-level benchmark remains stable.
- Build an HTML or interactive temporal visualizer after the artifact format stabilizes.

## Reproduction Commands

Verify or regenerate the current report without rerunning TRIBE predictions:

```bash
python scripts/run_asne_semantic_contrast_suite.py --skip-build
```

Run missing dictionary builds if needed:

```bash
python scripts/run_asne_semantic_contrast_suite.py --run-build
```

Regenerate only the report from existing summaries:

```bash
python scripts/generate_asne_contrast_report.py
```

## Safety Framing

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.
