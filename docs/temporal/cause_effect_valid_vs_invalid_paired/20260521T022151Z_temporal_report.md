# ASNE Temporal Contrast Evaluation

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

- Contrast: `cause_effect_valid_vs_invalid_paired`
- Dictionary: `outputs/asne_dictionaries/cause_effect_valid_vs_invalid_paired_tts_macos_say_samantha_180/dictionary_index.json`
- Eval: `data/stimuli/evals/contrasts/cause_effect_valid_vs_invalid_paired_eval.json`
- Eval summary: `outputs/asne_evals/contrasts/cause_effect_valid_vs_invalid_paired/20260521T021339Z_summary.json`

## Summary

| metric | value |
|---|---:|
| total examples | 6 |
| mean response accuracy | 0.8333 |
| early accuracy | 0.5000 |
| late accuracy | 0.8333 |
| final segment accuracy | 0.6667 |
| majority segment accuracy | 0.6667 |
| average switch count | 1.0000 |

## Per-Example Temporal Winners

| id | expected | mean top | early | late | final | majority | switches | largest transition |
|---|---|---|---|---|---|---|---:|---:|
| eval_paired_valid_cause_effect_01 | valid_cause_effect | valid_cause_effect | valid_cause_effect | valid_cause_effect | valid_cause_effect | valid_cause_effect | 0 | 15.1840 |
| eval_paired_valid_cause_effect_02 | valid_cause_effect | valid_cause_effect | invalid_cause_effect | valid_cause_effect | valid_cause_effect | valid_cause_effect | 1 | 9.7746 |
| eval_paired_valid_cause_effect_03 | valid_cause_effect | valid_cause_effect | invalid_cause_effect | valid_cause_effect | valid_cause_effect | invalid_cause_effect | 1 | 14.1355 |
| eval_paired_invalid_cause_effect_01 | invalid_cause_effect | valid_cause_effect | valid_cause_effect | valid_cause_effect | invalid_cause_effect | valid_cause_effect | 1 | 16.2836 |
| eval_paired_invalid_cause_effect_02 | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | 2 | 19.2942 |
| eval_paired_invalid_cause_effect_03 | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | 1 | 15.2849 |

## Segment-Level Differences From Mean Response

No examples differed from the mean-response top category under late/final/majority segment views.
