# ASNE Temporal Contrast Evaluation

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

- Contrast: `cause_effect_valid_vs_invalid_paired`
- Dictionary: `outputs/asne_dictionaries/semantic_contrast_v0_3_lite_cause_effect_valid_vs_invalid_paired_tts_macos_say_samantha_180/dictionary_index.json`
- Eval: `data/stimuli/evals/contrasts_v03/cause_effect_valid_vs_invalid_paired_eval.json`
- Eval summary: `outputs/asne_evals/contrasts_v03/cause_effect_valid_vs_invalid_paired/20260522T004940Z_summary.json`

## Summary

| metric | value |
|---|---:|
| total examples | 10 |
| mean response accuracy | 0.7000 |
| early accuracy | 0.6000 |
| late accuracy | 0.5000 |
| final segment accuracy | 0.4000 |
| majority segment accuracy | 0.7000 |
| average switch count | 1.3000 |

## Per-Example Temporal Winners

| id | expected | mean top | early | late | final | majority | switches | largest transition |
|---|---|---|---|---|---|---|---:|---:|
| eval_v03_valid_cause_effect_01 | valid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | 1 | 12.7442 |
| eval_v03_valid_cause_effect_02 | valid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | 2 | 16.7705 |
| eval_v03_valid_cause_effect_03 | valid_cause_effect | valid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | 1 | 11.9472 |
| eval_v03_valid_cause_effect_04 | valid_cause_effect | valid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | 1 | 10.3788 |
| eval_v03_valid_cause_effect_05 | valid_cause_effect | valid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | 1 | 13.2240 |
| eval_v03_invalid_cause_effect_01 | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | 1 | 12.6373 |
| eval_v03_invalid_cause_effect_02 | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | 2 | 21.2268 |
| eval_v03_invalid_cause_effect_03 | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | 1 | 12.8757 |
| eval_v03_invalid_cause_effect_04 | invalid_cause_effect | invalid_cause_effect | valid_cause_effect | invalid_cause_effect | invalid_cause_effect | invalid_cause_effect | 1 | 15.9549 |
| eval_v03_invalid_cause_effect_05 | invalid_cause_effect | valid_cause_effect | valid_cause_effect | invalid_cause_effect | valid_cause_effect | valid_cause_effect | 2 | 13.9251 |

## Segment-Level Differences From Mean Response

No examples differed from the mean-response top category under late/final/majority segment views.
