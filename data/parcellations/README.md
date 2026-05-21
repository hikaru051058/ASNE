# ASNE Parcellation Data

ASNE v0.1 expects a mapping from cortical response dimensions to parcel labels for ROI-level reporting.

The expected CSV format is:

```text
vertex_index,parcel_id,parcel_name
0,parcel_001,Parcel Name
1,parcel_001,Parcel Name
```

For TRIBE v2 outputs used in the current ASNE experiments, a real parcellation should cover `20,484` cortical dimensions. Local TRIBE documentation and utilities identify this output as the `fsaverage5` cortical surface: `10,242` left-hemisphere vertices followed by `10,242` right-hemisphere vertices.

Tests use a small deterministic mock parcellation with `20` vertices only. Mock parcellations are for software validation and should not be interpreted as anatomical regions.

## Real Atlas Generation

The helper script below checks available sources and can generate a CSV when an accepted source is available:

```bash
python scripts/create_fsaverage5_parcellation.py --dry-run
```

Current source status:

- HCP-MMP: available through TRIBE's fsaverage5 helper when `mne` can fetch the HCP-MMP parcellation. Generation requires explicit `--allow-fetch` because it may download external atlas files.
- Schaefer 100/200/400: no verified direct fsaverage5 mapping is available in the current local dependencies. Do not generate placeholder Schaefer labels.

Example HCP-MMP command after confirming source/license terms:

```bash
python scripts/create_fsaverage5_parcellation.py \
  --atlas hcp_mmp \
  --tribev2-package-path ./tribev2 \
  --allow-fetch
```

Do not commit large external atlas files unless their license allows redistribution. ROI reports are predicted TRIBE response summaries, not measured brain activity, diagnosis, or measurement of a person's mental state.
