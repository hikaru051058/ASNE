# ASNE Semantic Contrast Benchmark v0.3-lite

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

This benchmark expands the frozen semantic contrast suite from tiny six-example evals to ten held-out examples per contrast. It is still small and should be treated as a stability check, not a definitive accuracy claim.

## Static Vertex vs Parcel Results

| contrast | classification | v0.2 top1 | vertex top1 | parcel top1 | vertex top2 | parcel top2 | vertex mean rank | parcel mean rank | mean score margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `contradiction_vs_consistency_paired` | weak/deprioritized | 0.83 | 0.40 | 0.50 | 1.00 | 1.00 | 1.60 | 1.50 | 0.01 |
| `expected_vs_unexpected_paired` | stable | 0.83 | 0.80 | 0.90 | 1.00 | 1.00 | 1.20 | 1.10 | 0.03 |
| `cause_effect_valid_vs_invalid_paired` | stable | 0.83 | 0.70 | 0.80 | 1.00 | 1.00 | 1.30 | 1.20 | 0.01 |
| `approach_vs_static_paired` | pending | 0.50 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Temporal Results

| contrast | late accuracy | final accuracy | majority accuracy | average switches |
|---|---:|---:|---:|---:|
| `contradiction_vs_consistency_paired` | 0.50 | 0.40 | 0.50 | 1.30 |
| `expected_vs_unexpected_paired` | 0.90 | 0.80 | 0.80 | 0.70 |
| `cause_effect_valid_vs_invalid_paired` | 0.50 | 0.40 | 0.70 | 1.30 |
| `approach_vs_static_paired` | n/a | n/a | n/a | n/a |

## Summary Paths

| contrast | vertex summary | parcel summary | temporal summary |
|---|---|---|---|
| `contradiction_vs_consistency_paired` | `outputs/asne_evals/contrasts_v03/contradiction_vs_consistency_paired/20260521T192731Z_summary.json` | `outputs/asne_evals/contrasts_v03/contradiction_vs_consistency_paired/20260521T194142Z_summary.json` | `outputs/asne_temporal_evals/contradiction_vs_consistency_paired/20260521T194515Z_temporal_summary.json` |
| `expected_vs_unexpected_paired` | `outputs/asne_evals/contrasts_v03/expected_vs_unexpected_paired/20260522T000501Z_summary.json` | `outputs/asne_evals/contrasts_v03/expected_vs_unexpected_paired/20260522T001857Z_summary.json` | `outputs/asne_temporal_evals/contrasts_v03/expected_vs_unexpected_paired/20260522T002036Z_temporal_summary.json` |
| `cause_effect_valid_vs_invalid_paired` | `outputs/asne_evals/contrasts_v03/cause_effect_valid_vs_invalid_paired/20260522T004940Z_summary.json` | `outputs/asne_evals/contrasts_v03/cause_effect_valid_vs_invalid_paired/20260522T010250Z_summary.json` | `outputs/asne_temporal_evals/contrasts_v03/cause_effect_valid_vs_invalid_paired/20260522T010416Z_temporal_summary.json` |
| `approach_vs_static_paired` | `missing` | `missing` | `missing` |

## Confidence Warning

Each v0.3-lite contrast has ten held-out examples. This is larger than v0.2 but still too small for stable accuracy claims. Use the results to decide whether the signal is worth expanding, not as a production classifier benchmark.

Classification labels are coarse triage labels: `stable` means both vertex and parcel top-1 are at least 0.70; `promising but unstable` means at least one static feature space reaches 0.60 or temporal late/final reaches 0.70; `weak/deprioritized` means no current v0.3 view clears those thresholds.
