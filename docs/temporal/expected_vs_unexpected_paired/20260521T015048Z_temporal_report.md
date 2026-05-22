# ASNE Temporal Contrast Evaluation

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

- Contrast: `expected_vs_unexpected_paired`
- Dictionary: `outputs/asne_dictionaries/expected_vs_unexpected_paired_tts_macos_say_samantha_180/dictionary_index.json`
- Eval: `data/stimuli/evals/contrasts/expected_vs_unexpected_paired_eval.json`
- Eval summary: `outputs/asne_evals/contrasts/expected_vs_unexpected_paired/20260521T012803Z_summary.json`

## Summary

| metric | value |
|---|---:|
| total examples | 6 |
| mean response accuracy | 0.8333 |
| early accuracy | 0.5000 |
| late accuracy | 0.8333 |
| final segment accuracy | 1.0000 |
| majority segment accuracy | 0.8333 |
| average switch count | 0.5000 |

## Per-Example Temporal Winners

| id | expected | mean top | early | late | final | majority | switches | largest transition |
|---|---|---|---|---|---|---|---:|---:|
| eval_paired_expected_outcome_01 | expected_outcome | expected_outcome | unexpected_outcome | expected_outcome | expected_outcome | expected_outcome | 1 | 15.6490 |
| eval_paired_expected_outcome_02 | expected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | expected_outcome | unexpected_outcome | 1 | 13.8491 |
| eval_paired_expected_outcome_03 | expected_outcome | expected_outcome | unexpected_outcome | expected_outcome | expected_outcome | expected_outcome | 1 | 12.7926 |
| eval_paired_unexpected_outcome_01 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | 0 | 14.7612 |
| eval_paired_unexpected_outcome_02 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | 0 | 12.7473 |
| eval_paired_unexpected_outcome_03 | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | unexpected_outcome | 0 | 12.3016 |

## Segment-Level Differences From Mean Response

No examples differed from the mean-response top category under late/final/majority segment views.
