# ASNE Vertex vs Parcel Scoring Report v0

This compares the frozen vertex-level benchmark against HCP-MMP parcel-level centroid scoring.

- Parcel parcellation: `data/parcellations/fsaverage5_hcp_mmp.csv`
- Vertex setting: `feature_space=vertex`, `signature=mean_response`, `aggregation=centroid`, `scoring=centroid_raw`
- Parcel setting: `feature_space=parcel`, `signature=mean_response`, `aggregation=centroid`, `scoring=centroid_raw`

| contrast | vertex_top1 | parcel_top1 | vertex_top2 | parcel_top2 | vertex_mean_rank | parcel_mean_rank | difference | recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `contradiction_vs_consistency_paired` | 0.83 | 0.83 | 1.00 | 1.00 | 1.17 | 1.17 | 0.00 | parcel viable for scoring experiment |
| `expected_vs_unexpected_paired` | 0.83 | 0.83 | 1.00 | 1.00 | 1.17 | 1.17 | 0.00 | parcel viable for scoring experiment |
| `approach_vs_static_paired` | 0.50 | 0.50 | 1.00 | 1.00 | 1.50 | 1.50 | 0.00 | parcel viable for scoring experiment |
| `cause_effect_valid_vs_invalid_paired` | 0.83 | 0.83 | 1.00 | 1.00 | 1.17 | 1.17 | 0.00 | parcel viable for scoring experiment |

## Summary Paths

| contrast | vertex summary | parcel summary |
|---|---|---|
| `contradiction_vs_consistency_paired` | `outputs/asne_evals/contrasts/contradiction_vs_consistency_paired/20260521T012445Z_summary.json` | `outputs/asne_evals/contrasts/contradiction_vs_consistency_paired/20260521T060846Z_summary.json` |
| `expected_vs_unexpected_paired` | `outputs/asne_evals/contrasts/expected_vs_unexpected_paired/20260521T012803Z_summary.json` | `outputs/asne_evals/contrasts/expected_vs_unexpected_paired/20260521T060946Z_summary.json` |
| `approach_vs_static_paired` | `outputs/asne_evals/contrasts/approach_vs_static_paired/20260521T000101Z_summary.json` | `outputs/asne_evals/contrasts/approach_vs_static_paired/20260521T061043Z_summary.json` |
| `cause_effect_valid_vs_invalid_paired` | `outputs/asne_evals/contrasts/cause_effect_valid_vs_invalid_paired/20260521T021339Z_summary.json` | `outputs/asne_evals/contrasts/cause_effect_valid_vs_invalid_paired/20260521T061141Z_summary.json` |

## Safety Framing

This is predicted stimulus-response similarity from TRIBE outputs, not measured brain activity, diagnosis, or measurement of a person's mental state.
