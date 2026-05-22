# ASNE Temporal Contrast Evaluation

This is predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.

- Contrast: `contradiction_vs_consistency_paired`
- Dictionary: `outputs/asne_dictionaries/contradiction_vs_consistency_paired_tts_macos_say_samantha_180/dictionary_index.json`
- Eval: `data/stimuli/evals/contrasts/contradiction_vs_consistency_paired_eval.json`
- Eval summary: `outputs/asne_evals/contrasts/contradiction_vs_consistency_paired/20260521T013130Z_summary.json`

## Summary

| metric | value |
|---|---:|
| total examples | 6 |
| mean response accuracy | 0.5000 |
| early accuracy | 0.5000 |
| late accuracy | 0.8333 |
| final segment accuracy | 0.8333 |
| majority segment accuracy | 0.8333 |
| average switch count | 1.3333 |

## Per-Example Temporal Winners

| id | expected | mean top | early | late | final | majority | switches | largest transition |
|---|---|---|---|---|---|---|---:|---:|
| eval_paired_consistent_information_01 | consistent_information | contradictory_information | contradictory_information | consistent_information | consistent_information | consistent_information | 1 | 13.6519 |
| eval_paired_consistent_information_02 | consistent_information | consistent_information | contradictory_information | consistent_information | contradictory_information | contradictory_information | 2 | 10.1961 |
| eval_paired_consistent_information_03 | consistent_information | consistent_information | contradictory_information | consistent_information | consistent_information | consistent_information | 1 | 13.1071 |
| eval_paired_contradictory_information_01 | contradictory_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | 2 | 12.9513 |
| eval_paired_contradictory_information_02 | contradictory_information | consistent_information | contradictory_information | consistent_information | contradictory_information | contradictory_information | 2 | 10.9760 |
| eval_paired_contradictory_information_03 | contradictory_information | consistent_information | contradictory_information | contradictory_information | contradictory_information | contradictory_information | 0 | 12.2240 |

## Segment-Level Differences From Mean Response

### eval_paired_consistent_information_01

- Input: A person reads that the workshop is in Room 118. The hallway label also says Room 118.
- Expected: consistent_information
- Mean-response top category: contradictory_information
- Segment winners: ['contradictory_information', 'contradictory_information', 'contradictory_information', 'consistent_information', 'consistent_information', 'consistent_information', 'consistent_information']
- Late winner: consistent_information
- Final segment winner: consistent_information
- Majority segment winner: consistent_information

### eval_paired_contradictory_information_03

- Input: A person reads that the bus uses Bay 6. The terminal board says Bay 2.
- Expected: contradictory_information
- Mean-response top category: consistent_information
- Segment winners: ['contradictory_information', 'contradictory_information', 'contradictory_information', 'contradictory_information', 'contradictory_information']
- Late winner: contradictory_information
- Final segment winner: contradictory_information
- Majority segment winner: contradictory_information
