# Prepare MIMIC-IV and AmsterdamUMCdb with WeavEHR

The preprocessing path is now the same for both sources:

**Native data → WeavEHR extraction → WeavEHR harmonised concepts → cohort and events (Steps 01–03) → hourly bins → patient splits → trajectories.**

There is no mimic-code SQL/export step and no separate OMOP feature mapper in
Clin-JEPA. The clinical mappings and derived SOFA/AKI/Sepsis concepts come from
WeavEHR. This branch pins commit
`c163e2dd8eb9a389c8b01f1657a635f1118d0279` in `requirements-weavehr.txt`.

## Environments

Current WeavEHR needs Python 3.13. Keep preparation separate from the model's
CUDA environment; Steps 01–06 use the existing Clin-JEPA dependencies and do not
import WeavEHR.
Run commands from the repository root.

```bash
uv venv --python 3.13 .venv-weavehr
uv pip install --python .venv-weavehr/bin/python -r requirements-weavehr.txt
# Install Clin-JEPA and its model dependencies in your usual training environment.
```

## Source configuration

Use different `WEAVEHR_PROJECT` and `CLIN_JEPA_DATA` directories for each dataset.
Stay and patient IDs retain their native values: **do not concatenate directories
from different datasets**, whose integer IDs can collide. The same configuration
and training code can be run on either source.

Every command takes the same three arguments: the WeavEHR project, the output
directory and the dataset's WeavEHR name. There is no paths file.

| Source | WeavEHR `name` / `version` | Native input |
|---|---|---|
| MIMIC-IV | `mimic-iv` / `3.1` | Original `hosp/*.csv.gz`, `icu/*.csv.gz` |
| MIMIC-IV demo | `mimic-iv-demo` / `2.2` | Official demo layout |
| AmsterdamUMCdb | `aumc` / `1.5.0` | AMSTEL OMOP CDM 5.4 Parquet export |

AUMC input is the **AMSTEL OMOP export**, not the older raw Amsterdam CSV layout
and not an arbitrary OMOP database. The pinned mappings depend on AMSTEL's
concept IDs, source values and dose strings. Required cohort files are
`person.parquet`, `visit_occurrence.parquet`, `measurement/*.parquet`; supply the
other clinical tables for their corresponding concepts. The shipped AUMC profile
also reads `observation/*.parquet`, `drug_exposure/*.parquet` and
`procedure_occurrence/*.parquet`; other tables use single Parquet files. WeavEHR's shipped
configuration defines all filenames and source units. MIMIC cohort inputs
require patients, admissions, ICU stays, chart events and their join files.

```bash
export MIMIC_RAW=/path/to/mimic-iv-3.1
export AMSTERDAM_OMOP=/path/to/aumc-omop-parquet
export WEAVEHR_PROJECT=/path/to/weavehr/mimic
export CLIN_JEPA_DATA=/path/to/clin-jepa/mimic
export CLIN_JEPA_OUTPUT=/path/to/clin-jepa-results/mimic
# For Amsterdam, switch the three project/output directories.
```

Each dataset has one folder, `configs/data/<dataset>/`, with a `dataset.yaml`.
These files are the only list of supported datasets: each names the WeavEHR
datasets it serves. Beyond that they say only where the cohort is read from, which record identifiers tie rate and ventilation
events together, and which unharmonised source records to add. Amsterdam selects
AMSTEL `care_site_id=1` (ICU); medium-care visits (`2`) are excluded even though
both use visit concept `32037`.

## Build the WeavEHR project

Clin-JEPA has no preparation step of its own. Build the project exactly as
[WeavEHR's quickstart](https://github.com/aidh-ms/WeavEHR#quickstart) describes,
in the WeavEHR environment: an extraction step and a concept step for your
dataset, with no extra options.

```yaml
# extraction.yml
name: Extraction
version: 1.0.0
config:
  data:
    - name: mimic-iv
      version: "3.1"
      path: /path/to/mimic-iv-3.1
```

```yaml
# concept.yml
name: Concept
version: 1.0.0
config:
  extraction_step: Extraction
  mapping_configs:
    - name: mimic-iv
      version: "3.1"
```

Run both steps into the directory that `WEAVEHR_PROJECT` names (the quickstart
shows the few lines of Python). Clin-JEPA writes nothing into the project.
WeavEHR skips tables whose raw files are absent and only logs a warning. Clin-JEPA
does not check which tables were extracted, so check the WeavEHR log for skipped
tables before relying on sepsis or ventilation labels.

With the pinned WeavEHR, MIMIC-IV phenylephrine and vasopressin rates are not
produced: their mappings do not carry the stay and order identifiers that the
other vasopressor mappings carry, and vasopressin also lacks its duration and
charting time. Both return once those two mapping files carry them.

## Run the pipeline

One command runs Steps 01 to 06:

```bash
python -m clin_jepa.data.pipeline \
    --weavehr_project "$WEAVEHR_PROJECT" --data_dir "$CLIN_JEPA_DATA" --dataset mimic-iv \
    --n_workers 8
```

`--dataset` is the name WeavEHR uses (`mimic-iv`, `mimic-iv-demo`, `aumc`). If a
step fails, fix the cause and continue with `--from_step N`. Pass
`--dataset_version` only when the project holds several versions of the dataset.

Each step is also a command of its own, for example to spread Steps 02 to 04
over several workers. A step takes only what it uses:

```bash
PROJECT="--weavehr_project $WEAVEHR_PROJECT"
DATA="--data_dir $CLIN_JEPA_DATA"
python -m clin_jepa.data.step01_cohort         $PROJECT $DATA --dataset mimic-iv
python -m clin_jepa.data.step02_observations   $PROJECT $DATA --dataset mimic-iv
python -m clin_jepa.data.step03_actions        $PROJECT $DATA --dataset mimic-iv
python -m clin_jepa.data.step04_discretization $DATA
python -m clin_jepa.data.step05_split          $DATA
python -m clin_jepa.data.step06_trajectories   $DATA --n_workers 8
```

Steps 02–04 accept `--chunk_idx I --total_chunks N`. Complete **all** workers of
Steps 02 and 03 before starting Step 04. Steps 04–06 are upstream's; only their
`--data_dir` argument is new. Patient splits keep all admissions of one person in
the same split; where a mortality or sepsis stratum has fewer than 10 patients
(small or partial cohorts), the split is not stratified. Unknown mortality
(AmsterdamUMCdb) is a stratum of its own. Existing embeddings/checkpoints must be regenerated/retrained when
changing the input definitions; do not mix old mimic-code outputs into a new run.

## Event and outcome semantics

`configs/data/features.yaml` is the **single** semantic feature profile (96
observations and 34 actions) and `configs/data/trajectories.yaml` the single
trajectory profile. A concept row carries a native stay ID where WeavEHR's
mapping forwards one. Rows without one are assigned
inside an ICU window; conflicting native IDs never fall back to a nearby
stay. Points use `[admission, discharge)`. An event without a native stay ID
inside two overlapping ICU windows goes to the later stay.

The cohort keeps the original selection rules (`configs/data/cohort.yaml`):
adults, the first ICU stay of each hospital admission, at least 6 h in the ICU
and at least one vital sign. `icu_los_hours` is capped at `max_icu_los_hours`;
`los_icu` keeps the actual length of stay. AmsterdamUMCdb has no hospital
admission identifier, so the first-stay rule keeps all of its stays. Sepsis-3 is
dated per patient: an onset up to `sepsis.prevalent_lookback_hours` before ICU
admission marks the stay as septic on arrival, with its true earlier time.

Steps 02 and 03 check the complete harmonised concept code, including its unit.
It does not infer units from source strings. The only output conversions are
explicit canonical aliases (`C` → `°C`, `insp/min` → `breaths/min`, score units)
and vasopressin `units/min` → the existing Clin-JEPA `units/hour` representation.
Both datasets use the clinical event timestamps supplied by WeavEHR.

### Names, text and unharmonised source records

Events are keyed by WeavEHR concept. An event of a concept in
`configs/data/features.yaml` has `source_table = concept` and the concept name
as its `variable_name` (`name` where two concepts form one variable, as the
paired ventilation markers do). Each entry gives the `display_name` written in
the state and action text, and the order of the entries is their order within
an hour's text. The downstream label and tabular code select events by concept name.

With `raw_actions.enabled` in `features.yaml`, Step 03 also reads unharmonised
source records from the WeavEHR extraction, as declared under `raw_actions` in
the dataset profile (`configs/data/<dataset>/dataset.yaml`). MIMIC-IV declares infusions
(`inputevents`), medication orders (`prescriptions`), administrations (`emar`)
and procedures (`procedureevents`); WeavEHR extracts `emar` with the other tables.
AmsterdamUMCdb declares none. Such an event has its extraction table as
`source_table`, the source's own item name as `variable_name` and the kind of
record as `category`. Its text follows the concepts in a format chosen by
category, by upstream's text templates. Upstream's action filter leaves
procedures and a list of flushes and diluents out of the text and states each
prescription only in its first hour.

A record with a start and an end event (an infusion, a procedure) is never
described by values from its own future. While it runs, its text states what
was entered at its start: for a MIMIC-IV infusion the item, its order category
and `originalrate`. The amount that was delivered comes from the end event and
is a separate event at the later of the end and its recorded availability.
A record without a rate, such as a
bolus, is that later event alone. This differs from the original pipeline,
which stated the delivered amount and rate from the start of every infusion.

The item names of source records are the source's own and are not shared between
datasets. Switch `raw_actions.enabled` off where two datasets must read alike.

The supported vasopressor update streams are joined by native infusion/segment
IDs. Each rate holds until the next update of its record, as one mimic-code
row does; records that overlap stay separate events. A final rate without a
charted stop remains a point instead of being carried indefinitely.
WeavEHR's MIMIC-IV vasopressin is the rate a record ran at, stated when the record
ends; the pipeline takes it as that, a point at the record's end.
Paired ventilation start/end markers form intervals
only with a matching native record ID and a valid end; records without one start
before one end are counted and left out. As upstream, an interval must start inside
the ICU stay, and its end is clamped to discharge. An interval that started before
admission is therefore not in the actions; on the MIMIC-IV demo this leaves out 7
ventilation and 2 norepinephrine intervals that begin up to 36 minutes before
admission, and the original pipeline leaves out the same ventilation episodes.
The labels read whether a stay arrives ventilated from the WeavEHR project, as
upstream reads it from mimic-code's ventilation table. A record's rows are read
in full before its intervals are formed, so read batches never split it.

WeavEHR's six SOFA components are named `respiration_current`,
`cardiovascular_current`, `renal_current`, `liver_current`,
`coagulation_current`, and `cns_current`. The native total `sofa_24hours` retains
its trailing 24-hour definition. Components are not relabelled as 24-hour maxima.
The generic `label_sofa_*` label names refer to these current components, as
specified in the saved configuration. Old embeddings/checkpoints cannot be
compared as though these inputs were unchanged.

**Upgrade:** these changes require rebuilding the WeavEHR project, then
rerunning Steps 01–06.

Steps 02 and 03 keep upstream's structure and its `melt_to_long` and
`melt_to_long_action`: each reads the configured concepts (and, for actions, the
declared source records) for its chunk of stays, assigns them to stays with
upstream's temporal utilities and writes one file per stay that has events. A
concept carries its number or its level, not WeavEHR's source string. They log
each concept WeavEHR does not produce for the dataset and each record they leave out. As upstream, write each
run to a fresh output directory.

This is a new preprocessing definition, **not a reproduction of all 34 original
task definitions**; the cohort rules above are the original ones:

- SOFA, AKI and Sepsis-3 follow the pinned WeavEHR definitions. Sepsis-3 emits
  positive onsets. Where WeavEHR produced the sepsis concept for the dataset, a
  stay without a detected onset is a negative; where it produced none, sepsis
  stays unknown. Ventilation follows the same rule. Whether the tables behind a
  concept were all extracted is WeavEHR's to report, not Clin-JEPA's to check.
- MIMIC hospital mortality is read from WeavEHR's explicit extracted flag. ICU
  death, 7/14/30/90-day mortality, discharge home and readmissions are computed
  by `clin_jepa.downstream.labels`, as upstream, with the original definitions,
  from the WeavEHR events named under `cohort.outcomes` in the dataset file
  (MIMIC-IV only): that declaration asserts follow-up complete enough for a
  missing death record or later admission to be a negative. Amsterdam hospital
  mortality and all of these follow-up outcomes remain unknown.
  Date-only death records can date a death but never place it inside an ICU stay.
- Some original features have no shared mapping, including milrinone, a
  norepinephrine-equivalent dose, renal-replacement status and ventilator
  modes. Invasive-line inventories are not substituted with approximations.
  Discharge diagnoses are not extracted: WeavEHR has them only as unharmonised
  source codes (ICD in MIMIC-IV, OMOP concepts in AUMC), and upstream excludes
  the `diagnosis` category from the state text and the tabular baselines, so no
  model input or label reads them. Broad
  medications and ICU procedures exist only as unharmonised MIMIC-IV source
  records (above). Add/review their mappings in WeavEHR before extending the
  shared feature profile.
- AUMC respiratory support, tracheostomy and specimen selection are not
  definitionally identical to MIMIC. A shared concept label does not by itself
  validate a clinical transfer endpoint. No unsynchronised PaO2/FiO2 ratio is
  derived.
- `clin_jepa.downstream.labels` now reads the canonical cohort and events without
  MIMIC source files. Five measurement-based hourly endpoints are enabled.
  Sepsis onset and IMV onset are enabled where WeavEHR provides the concept. Ventilation
  episodes come from the WeavEHR project; a record without one start before one end
  is left out. Incomplete future
  windows remain excluded. Recorded AKI stage after 24 h requires observed
  baseline and later stages. Prolonged IMV uses exact union duration: a positive
  can be established in a truncated stay, but a negative needs the full episode.
  Remaining unsupported treatment/absence endpoints stay masked.
  `labels/coverage.json` records counts and reasons.

The optional pretrained **CLMBR-T-base benchmark** is still its own MIMIC-specific
input path, with upstream vocabulary requirements. It is not part of WeavEHR
preparation or Clin-JEPA training and has not been converted into an AUMC model.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
# If the preprocessing environment is elsewhere:
WEAVEHR_PYTHON=/path/to/python3.13 python -m pytest -q tests/test_weavehr_native.py
```

The native tests execute the real pinned WeavEHR extractor and concept step on
synthetic MIMIC and AMSTEL-shaped files, then run all Clin-JEPA data stages,
trajectory loading and downstream label generation. They
check independently specified HR, creatinine unit equivalence, norepinephrine
intervals and missing outcomes. The step tests also cover readmissions,
identity conflicts, overlapping stays/pumps, simultaneous changes/stops, batch
boundaries and malformed units. Native tests skip
explicitly if the separate WeavEHR environment is unavailable. These are software
integration checks; full-cohort clinical agreement and GPU training need separate
validation on authorised data.

The [MIMIC-IV demo comparison](demo-comparison.md) now reports an actual run of
both the original mimic-code pipeline and this adaptation on the public v2.2
demo. Basic measurements agree closely, but ventilation, derived scores, sepsis,
some treatments and endpoint coverage differ substantially. It includes pinned
versions, aggregate results and a reproducible comparison script.
The subsequent adapter-fix rerun is documented in
[demo-comparison-fixed.md](demo-comparison-fixed.md), preserving the first report
as the baseline. Both reports predate the concept-keyed event names.

The comparison script lines the original events up with the concepts through
`configs/data/mimic_code_aliases.yaml`, the only file that still names the
original tables and columns; nothing in the pipeline reads it. Besides events,
hourly values and labels, the report's `text` section compares the fragments
both pipelines write for each hour, for the concepts with an alias and for the
source records.
