# ASNE Temporal Contrast Evaluation

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

- Contrast: `contradiction_vs_consistency_paired`
- Dictionary: `outputs/asne_dictionaries/semantic_contrast_v0_3_lite_contradiction_vs_consistency_paired_tts_macos_say_samantha_180/dictionary_index.json`
- Eval: `data/stimuli/evals/contrasts_v03/contradiction_vs_consistency_paired_eval.json`
- Eval summary: `outputs/asne_evals/contrasts_v03/contradiction_vs_consistency_paired/20260521T194142Z_summary.json`

## Summary

| metric | value |
|---|---:|
| total examples | 10 |
| mean response accuracy | 0.5000 |
| early accuracy | 0.4000 |
| late accuracy | 0.5000 |
| final segment accuracy | 0.4000 |
| majority segment accuracy | 0.5000 |
| average switch count | 1.3000 |

## Per-Example Temporal Winners

| id | expected | mean top | early | late | final | majority | switches | largest transition |
|---|---|---|---|---|---|---|---:|---:|
| eval_v03_consistent_information_01 | consistent_information | consistent_information | contradictory_information | consistent_information | consistent_information | consistent_information | 1 | 10.5502 |
| eval_v03_consistent_information_02 | consistent_information | consistent_information | contradictory_information | consistent_information | consistent_information | consistent_information | 3 | 15.4587 |
| eval_v03_consistent_information_03 | consistent_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | 2 | 10.8967 |
| eval_v03_consistent_information_04 | consistent_information | consistent_information | contradictory_information | consistent_information | consistent_information | consistent_information | 1 | 13.0795 |
| eval_v03_consistent_information_05 | consistent_information | consistent_information | consistent_information | consistent_information | contradictory_information | consistent_information | 2 | 12.4496 |
| eval_v03_contradictory_information_01 | contradictory_information | consistent_information | consistent_information | consistent_information | consistent_information | consistent_information | 1 | 11.4224 |
| eval_v03_contradictory_information_02 | contradictory_information | consistent_information | contradictory_information | consistent_information | consistent_information | consistent_information | 1 | 11.4425 |
| eval_v03_contradictory_information_03 | contradictory_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | 0 | 10.8680 |
| eval_v03_contradictory_information_04 | contradictory_information | consistent_information | contradictory_information | consistent_information | consistent_information | consistent_information | 1 | 13.0956 |
| eval_v03_contradictory_information_05 | contradictory_information | consistent_information | consistent_information | consistent_information | consistent_information | consistent_information | 1 | 12.3014 |

## Segment-Level Differences From Mean Response

No examples differed from the mean-response top category under late/final/majority segment views.
