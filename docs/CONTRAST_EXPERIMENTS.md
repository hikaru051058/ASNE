# ASNE Contrast Experiments

ASNE contrast experiments test controlled binary stimulus axes using predicted stimulus-response similarity. They are not emotion detection, diagnosis, or measurement of an individual person's mental state.

## Contrast Set v0

The v0 contrast files are stored under:

- `data/stimuli/dictionaries/contrasts/`
- `data/stimuli/evals/contrasts/`

Each contrast has five dictionary examples per category and three held-out evaluation examples per category.

| contrast | baseline category | comparison category |
| --- | --- | --- |
| `contradiction_vs_consistency` | `consistent_information` | `contradictory_information` |
| `approach_vs_static` | `static_scene` | `approaching_agent` |
| `missing_vs_present` | `object_present` | `object_missing` |
| `unresolved_vs_resolved` | `problem_unresolved` | `problem_resolved` |

The baseline category is recorded in each dictionary JSON as `baseline_category`. ASNE stores that category in the built `dictionary_index.json` and uses it for `delta_from_neutral` comparisons; for these binary contrasts, this is effectively a delta from the configured baseline category.

## Run Commands

Run one contrast dictionary build:

```bash
python scripts/run_asne_dictionary.py \
  --dictionary data/stimuli/dictionaries/contrasts/contradiction_vs_consistency.json \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --debug-traceback
```

Evaluate the built dictionary:

```bash
python scripts/evaluate_asne_dictionary.py \
  --dictionary outputs/asne_dictionaries/contradiction_vs_consistency/dictionary_index.json \
  --eval data/stimuli/evals/contrasts/contradiction_vs_consistency_eval.json \
  --tribev2-package-path ./tribev2 \
  --cache-folder ./cache \
  --feature-device cpu \
  --signature delta_from_neutral \
  --aggregation centroid \
  --scoring full \
  --debug-traceback
```

Repeat those commands for:

- `approach_vs_static`
- `missing_vs_present`
- `unresolved_vs_resolved`

Then summarize the completed contrast evaluation summaries:

```bash
python scripts/summarize_asne_contrasts.py \
  --summaries outputs/asne_evals/contrasts/*/*_summary.json \
  --output outputs/asne_evals/contrasts/contrast_summary.md
```

## Notes

TRIBE v2 text preprocessing may use upstream text-to-speech and speech/text feature extraction tools. If an uncached text stimulus fails during preprocessing, check network/DNS availability, local model cache state, and audio dependencies such as `ffmpeg`.

Generated response signatures and raw segment arrays are experiment outputs and should not be committed. The repository tracks the small controlled stimulus JSON files, not model outputs.
