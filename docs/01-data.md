# 01 · Data

Both MIMIC-IV and AmsterdamUMCdb use WeavEHR and the same steps. Start with
[the complete preparation guide](data-preparation.md) for environments, inputs,
commands, provenance and the differences from the original paper's pipeline.

| Step | Input | Module (`python -m clin_jepa.data.…`) | Output |
|---|---|---|---|
| — | Native MIMIC or AMSTEL OMOP | WeavEHR's own quickstart (Python 3.13), not a Clin-JEPA module | WeavEHR project: MEDS extraction and harmonised concepts |
| 01 | WeavEHR cohort events and shared eligibility | `step01_cohort` | `cohort/cohort.csv` |
| 02 | Shared observation concepts | `step02_observations` | `observations/<stay_id>.parquet` |
| 03 | Shared intervention concepts | `step03_actions` | `actions/<stay_id>.parquet` |
| 04 | Completed event workers | `step04_discretization` | `discretized/<stay_id>.pkl` |
| 05 | Cohort | `step05_split` | `splits/splits.csv` |
| 06 | Bins, splits, shared text/label profile | `step06_trajectories` | `trajectories/{train,val,test}/*.pt` |

`python -m clin_jepa.data.pipeline` runs Steps 01–06 in one go. Steps 01–03 read
the WeavEHR project (`--weavehr_project`, `--data_dir`, `--dataset`); Steps 04–06
are upstream's and take only `--data_dir`. Steps 02–04 support indexed workers;
finish all event workers before binning. Training and embedding read upstream's
trajectory shards unchanged.
`feature_table` exports the shared concept inventory; Steps 02 and 03 log each
concept that WeavEHR does not produce for the dataset.
