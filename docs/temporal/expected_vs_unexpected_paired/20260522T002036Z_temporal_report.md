# ASNE Temporal Contrast Evaluation

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

- Contrast: `expected_vs_unexpected_paired`
- Dictionary: `outputs/asne_dictionaries/semantic_contrast_v0_3_lite_expected_vs_unexpected_paired_tts_macos_say_samantha_180/dictionary_index.json`
- Eval: `data/stimuli/evals/contrasts_v03/expected_vs_unexpected_paired_eval.json`
- Eval summary: `outputs/asne_evals/contrasts_v03/expected_vs_unexpected_paired/20260522T000501Z_summary.json`

## Summary

| metric | value |
|---|---:|
| total examples | 10 |
| mean response accuracy | 0.8000 |
| early accuracy | 0.5000 |
| late accuracy | 0.9000 |
| final segment accuracy | 0.8000 |
| majority segment accuracy | 0.8000 |
| average switch count | 0.7000 |

## Per-Example Temporal Winners

| id | expected | mean top | early | late | final | majority | switches | largest transition |
|---|---|---|---|---|---|---|---:|---:|
| eval_v03_expected_outcome_01 | expected_outcome | unexpected_outcome | unexpected_outcome | expected_outcome | expected_outcome | unexpected_outcome | 1 | 16.0251 |
| eval_v03_expected_outcome_02 | expected_outcome | expected_outcome | unexpected_outcome | expected_outcome | expected_outcome | expected_outcome | 1 | 13.8441 |
| eval_v03_expected_outcome_03 | expected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | expected_outcome | unexpected_outcome | 1 | 12.3651 |
| eval_v03_expected_outcome_04 | expected_outcome | expected_outcome | unexpected_outcome | expected_outcome | expected_outcome | expected_outcome | 1 | 13.9097 |
| eval_v03_expected_outcome_05 | expected_outcome | expected_outcome | unexpected_outcome | expected_outcome | expected_outcome | expected_outcome | 1 | 13.8548 |
| eval_v03_unexpected_outcome_01 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | 0 | 12.5471 |
| eval_v03_unexpected_outcome_02 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | 0 | 12.3399 |
| eval_v03_unexpected_outcome_03 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | expected_outcome | unexpected_outcome | 1 | 11.6488 |
| eval_v03_unexpected_outcome_04 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | expected_outcome | unexpected_outcome | 1 | 12.9481 |
| eval_v03_unexpected_outcome_05 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | 0 | 10.5750 |

## Segment-Level Differences From Mean Response

No examples differed from the mean-response top category under late/final/majority segment views.
