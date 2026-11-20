"""Reading a WeavEHR project, and filling in what its outputs do not carry yet.

Every function here exists because of how WeavEHR delivers data, not because of anything Clin-JEPA
decides: there is no ICU-stay concept, most rows carry no stay ID, admission outcomes come as raw
extracted events, and rates and ventilation come as updates and markers rather than intervals. Once
WeavEHR delivers one of these, the matching part goes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from clin_jepa.data.helpers.source import load_weavehr_profile, project_dataset
from clin_jepa.data.helpers.temporal_utils import assign_stay_ids_from_subject, clip_to_icu_window

logger = logging.getLogger(__name__)

# Rows read from a WeavEHR parquet file at a time.
READ_BATCH_ROWS = 100_000


def native_ids(values: pd.Series) -> pd.Series:
    """Keep identifiers integral, including null native IDs stored as strings."""
    return values.replace({"": None}).astype("Int64")


def optional_column(frame: pd.DataFrame, name: str, default=None) -> pd.Series:
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


class Project:
    def __init__(self, paths: dict):
        self.paths = paths
        self.root = Path(paths["weavehr"]["project"])
        for step in ("extraction", "concept"):
            if not (self.root / "datasets" / step / ".complete").is_file():
                raise ValueError(f"Incomplete WeavEHR {step} step; build the project as WeavEHR documents: {self.root}")
        self.dataset, self.version = project_dataset(paths)
        self.profile = load_weavehr_profile(paths)

    def extracted_path(self, table: str, event: str) -> Path:
        return self.root / "datasets/extraction/data" / self.dataset / self.version / table / f"{event}.parquet"

    def concept_path(self, name: str, version: str = "1.0.0") -> Path:
        return self.root / "datasets/concept/data" / name / version / self.dataset / f"{self.version}.parquet"

    def batches(self, path: Path, subjects=None):
        dataset = ds.dataset(path, format="parquet")
        predicate = ds.field("subject_id").isin([int(x) for x in subjects]) if subjects is not None else None
        for batch in dataset.scanner(filter=predicate, batch_size=READ_BATCH_ROWS).to_batches():
            frame = batch.to_pandas()
            frame["time"] = pd.to_datetime(frame["time"])
            yield frame

    def event(self, table: str, event: str, subjects=None) -> pd.DataFrame | None:
        """Extracted events, of the given subjects if named; None if WeavEHR extracted no such event."""
        path = self.extracted_path(table, event)
        if not path.exists():
            return None
        frames = list(self.batches(path, subjects))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def required_event(self, spec: dict, event: str | None = None) -> pd.DataFrame:
        frame = self.event(spec["table"], event or spec["event"])
        if frame is None:
            raise FileNotFoundError(f"WeavEHR extracted no {spec['table']} {event or spec['event']} events")
        return frame


# ---------------------------------------------------------------- ICU stays and their patients


def icu_stays(project: Project) -> pd.DataFrame:
    """The dataset's ICU stays, from its stay admission and discharge events paired by stay ID.

    WeavEHR has no ICU-stay concept; the dataset file names the events, and which
    admissions count (AmsterdamUMCdb: ICU, not medium care). Selection applies to the admission event
    only: some sources do not carry the same care-site metadata on the discharge.
    """
    spec = project.profile["cohort"]["stays"]
    start = project.required_event(spec, spec["start"])
    for column, values in spec.get("filters", {}).items():
        start = start[start[column].astype("string").isin([str(v) for v in values])]
    if spec.get("code_regex"):
        start = start[start.code.str.contains(spec["code_regex"], regex=True, na=False)]
    end = project.required_event(spec, spec["end"])
    keys = ["subject_id", spec["id"]]
    for frame in (start, end):
        frame[spec["id"]] = native_ids(frame[spec["id"]])
        frame.drop_duplicates(inplace=True)
    stays = start.merge(end[keys + ["time"]], on=keys, suffixes=("", "_end"), validate="one_to_one")
    stays = stays.rename(columns={spec["id"]: "stay_id", "time": "icu_intime", "time_end": "icu_outtime"})
    admission_id = spec.get("admission_id")
    stays["hadm_id"] = (
        native_ids(stays[admission_id]) if admission_id else pd.Series(pd.NA, index=stays.index, dtype="Int64")
    )
    return stays[["subject_id", "stay_id", "hadm_id", "icu_intime", "icu_outtime"]].reset_index(drop=True)


def icustay_detail(project: Project) -> pd.DataFrame:
    """The ICU stays with the columns of mimic-code's ``icustay_detail`` that the cohort reads.

    Age, sex and race are WeavEHR's harmonised concepts. Hospital mortality has no concept yet; it is
    read from the extracted event the dataset file names, and unknown where it names none.
    """
    stays = icu_stays(project)
    stays["los_icu"] = (stays.icu_outtime - stays.icu_intime).dt.total_seconds() / 86400
    # Ranked over all ICU stays of the admission; a dataset without admissions (AUMC) has only first stays.
    stays["first_icu_stay"] = stays.groupby("hadm_id").icu_intime.rank(method="dense").eq(1) | stays.hadm_id.isna()
    stays["admission_age"] = patient_attribute(project, stays, "patient_age", "numeric_value")
    stays["gender"] = patient_attribute(project, stays, "patient_sex", "text_value").map({"Male": "M", "Female": "F"})
    # UNKNOWN is MIMIC-IV's own word for a race that is not recorded.
    stays["race"] = patient_attribute(project, stays, "patient_race", "text_value").fillna("UNKNOWN")
    mortality = project.profile["cohort"].get("hospital_mortality")
    stays["hospital_expire_flag"] = pd.to_numeric(admission_value(project, stays, mortality)) if mortality else np.nan
    return stays


def assign_stay_ids(rows: pd.DataFrame, windows_df: pd.DataFrame, timestamp_col: str = "time") -> pd.DataFrame:
    """WeavEHR rows with the ICU stay their time falls in, as upstream's temporal utilities assign them.

    WeavEHR keeps the stay ID on some rows only. Those are clipped to their stay, as upstream's stay-keyed
    tables are; the others are matched to a stay of their patient, as its subject-keyed tables are.
    """
    windows_df = windows_df[["subject_id", "stay_id", "icu_intime", "icu_outtime"]]
    stay = native_ids(optional_column(rows, "stay_id"))
    rows = rows.drop(columns="stay_id", errors="ignore")
    return pd.concat([
        clip_to_icu_window(rows[stay.notna()].assign(stay_id=stay[stay.notna()].astype("int64")), windows_df,
                           timestamp_col),
        assign_stay_ids_from_subject(rows[stay.isna()].copy(), windows_df, timestamp_col),
    ], ignore_index=True).astype({"stay_id": "int64"})


def patient_attribute(project: Project, cohort: pd.DataFrame, concept: str, column: str) -> pd.Series:
    """A patient attribute that WeavEHR harmonises, per stay; unknown where WeavEHR has none.

    The concept rows carry no stay or admission ID. A stay takes the latest row of its patient
    that is not later than its discharge: the hospital admission it belongs to, not a later one.
    """
    path = project.concept_path(concept)
    batches = list(project.batches(path, cohort.subject_id.unique())) if path.exists() else []
    if not batches:
        return pd.Series(np.nan, index=cohort.index)
    rows = pd.concat(batches, ignore_index=True)[["subject_id", "time", column]].dropna()
    rows["subject_id"] = rows.subject_id.astype("int64")
    rows["time"] = pd.to_datetime(rows.time).astype("datetime64[ns]")
    stays = pd.DataFrame({
        "stay": cohort.index, "subject_id": cohort.subject_id.astype("int64").to_numpy(),
        "time": cohort.icu_outtime.astype("datetime64[ns]").to_numpy(),
    })
    rows = rows.sort_values("time", kind="stable")
    matched = pd.merge_asof(stays.sort_values("time"), rows, on="time", by="subject_id")
    return matched.set_index("stay")[column].reindex(cohort.index)


def admission_value(project: Project, cohort: pd.DataFrame, spec: dict) -> pd.Series:
    """A value of the stay's admission that WeavEHR has no concept for, read from an extracted event."""
    events = project.required_event(spec)
    keys = list(dict.fromkeys(["subject_id", spec["key"]]))
    for col in keys:
        if col not in events:
            raise ValueError(f"Cohort attribute {spec['event']} lacks {col}")
        events[col] = native_ids(events[col])
    values = events[keys].assign(_value=events[spec["value"]]).drop_duplicates()
    if values.duplicated(keys).any():
        raise ValueError(f"Conflicting cohort attributes for {spec['event']}")
    return cohort[keys].merge(values, on=keys, how="left", validate="many_to_one")["_value"]


def sepsis3(project: Project, cohort: pd.DataFrame, lookback_hours: float) -> pd.DataFrame:
    """Per stay ``sepsis3`` and ``sepsis_sofa_time``, the first onset, as mimic-code's ``sepsis3`` gives them.

    WeavEHR dates detected onsets per patient, without a stay and without negatives. An onset inside a
    stay belongs to it. Any other onset up to ``lookback_hours`` before an admission marks the next stay
    as septic on arrival and keeps its true, earlier time, so it is never read as an onset during the
    stay. Where WeavEHR produced the concept, a stay without an onset is a negative; elsewhere unknown.
    """
    sepsis = cohort[["stay_id"]].copy()
    path = project.concept_path("sepsis3")
    if not path.exists():
        return sepsis.assign(sepsis3=np.nan, sepsis_sofa_time=pd.NaT)
    lookback = pd.Timedelta(hours=float(lookback_hours))
    events = pd.concat(list(project.batches(path, cohort.subject_id.unique())), ignore_index=True)
    events = events[events.numeric_value == 1].reset_index(drop=True).assign(_onset=lambda f: np.arange(len(f)))
    inside = assign_stay_ids(events, cohort)
    merged = events[~events._onset.isin(inside._onset)].drop(columns=["stay_id", "hadm_id"], errors="ignore")
    merged = merged.merge(cohort[["subject_id", "stay_id", "icu_intime"]], on="subject_id")
    merged = merged[(merged.time < merged.icu_intime) & (merged.time >= merged.icu_intime - lookback)]
    before = merged.sort_values(["_onset", "icu_intime"], kind="stable").drop_duplicates("_onset")
    sepsis["sepsis_sofa_time"] = sepsis.stay_id.map(pd.concat([inside, before]).groupby("stay_id").time.min())
    sepsis["sepsis3"] = sepsis.sepsis_sofa_time.notna().astype(float)
    return sepsis


def stays_with_values(project: Project, cohort: pd.DataFrame, rules: list[dict]) -> set:
    """The stays that have a value of any of these concepts."""
    stays = set()
    for rule in rules:
        path = project.concept_path(rule["concept"], rule["version"])
        if path.exists():
            for batch in project.batches(path, cohort.subject_id.unique()):
                stays.update(assign_stay_ids(batch[batch.numeric_value.notna()], cohort).stay_id)
    return stays


# ---------------------------------------------------------------- concepts, rates and ventilation


def _codes(frame: pd.DataFrame, name: str, rule: dict) -> dict:
    """The codes WeavEHR writes for a concept, ``<name>//<unit>[//<level>]``, mapped to their level.

    The unit is part of the code; one other than the configured unit stops the run.
    """
    prefix = name + ("//" + rule["unit"] if rule["unit"] else "")
    codes = {prefix: None, **{prefix + "//" + level: level for level in rule.get("levels", [])}}
    unexpected = set(frame.code.dropna()) - set(codes)
    if unexpected:
        raise ValueError(f"Unexpected WeavEHR codes/units for {name}: {sorted(unexpected)}")
    return codes


def concept_table(project: Project, rule: dict, subjects) -> pd.DataFrame:
    """One concept of the given patients, laid out as a column of a mimic-code concept table.

    The rows hold ``subject_id``, ``stay_id`` where WeavEHR keeps it, ``time`` and the value under the
    concept's name. Empty where WeavEHR produced no such concept for the dataset.
    """
    path = project.concept_path(rule["concept"], rule["version"])
    frames = list(project.batches(path, subjects)) if path.exists() else []
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    _codes(rows, rule["concept"], rule)
    return pd.DataFrame({
        "subject_id": rows.subject_id,
        "stay_id": native_ids(optional_column(rows, "stay_id")),
        "time": rows.time.astype("datetime64[ns]"),
        rule["concept"]: pd.to_numeric(rows.numeric_value).astype("float64"),
    })


def action_table(project: Project, rule: dict, subjects) -> pd.DataFrame:
    """One action concept of the given patients, laid out as a mimic-code action table.

    The rows hold ``subject_id``, ``stay_id`` where WeavEHR keeps it, ``time``, for an interval its
    ``end_time``, and the value under the variable's name: the number, or the level its code names.
    WeavEHR gives infusion rates as updates and ventilation as start and end markers, tied together by
    the record ID the dataset file names; they become the intervals mimic-code has. Empty where WeavEHR
    produced no such concept for the dataset.
    """
    mode = rule.get("mode", "point")
    roles = {"start": rule["concept"], **({"end": rule["end_concept"]} if "end_concept" in rule else {})}
    paths = {role: project.concept_path(name, rule["version"]) for role, name in roles.items()}
    if not all(path.exists() for path in paths.values()):
        return pd.DataFrame()
    record = {"rate_updates": "rate_record_id", "paired_markers": "ventilation_record_id"}.get(mode)
    frames = []
    for role, name in roles.items():
        for batch in project.batches(paths[role], subjects):
            frame = pd.DataFrame({
                "subject_id": batch.subject_id,
                "stay_id": native_ids(optional_column(batch, "stay_id")),
                "time": batch.time.astype("datetime64[ns]"),
                "numeric_value": pd.to_numeric(batch.numeric_value) * float(rule.get("scale", 1)),
                "text_value": batch.code.map(_codes(batch, name, rule)),
            })
            if record:
                frame["_role"] = role
                frame["_record"] = optional_column(batch, project.profile[record]).astype("string").replace("", pd.NA)
            frames.append(frame)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if mode == "rate_updates":
        rows = rate_intervals(rows)
    elif mode == "paired_markers":
        rows = marker_intervals(rows)
    if rows.empty:
        return pd.DataFrame()
    name = rule.get("name", rule["concept"])
    table = pd.DataFrame({"subject_id": rows.subject_id, "stay_id": native_ids(rows.stay_id), "time": rows.time})
    if record:
        table["end_time"] = pd.to_datetime(rows.end_timestamp)
    table[name] = rows.text_value if rule.get("levels") else rows.numeric_value.astype("float64")
    return table.reset_index(drop=True)


def rate_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """Each rate holds until the next update of its infusion record, as one mimic-code row does.

    Records that overlap stay separate events. A rate with no later update of its record (no
    stop was charted) is a point: how long it ran is unknown.
    """
    rows = rows[rows._record.notna()].sort_values(["_record", "time", "numeric_value"], kind="stable")
    # Of simultaneous updates of one record, the highest rate wins (a change charted with its stop).
    rows = rows.drop_duplicates(["_record", "time"], keep="last")
    rows = rows.assign(end_timestamp=rows.groupby("_record").time.shift(-1))
    return rows[rows.numeric_value > 0]


def marker_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """Each record's start and end marker as one interval; records without exactly one of each are left out."""
    if rows._record.isna().any():
        raise ValueError("Ventilation markers require native record identifiers")
    output, unpaired = [], 0
    for _, record in rows.groupby(["_record", "text_value"], dropna=False, sort=False):
        starts = record[record._role == "start"].drop_duplicates("time")
        ends = record[record._role == "end"].drop_duplicates("time")
        if len(starts) != 1 or len(ends) != 1 or ends.time.iloc[0] <= starts.time.iloc[0]:
            unpaired += 1
            continue
        output.append({**starts.iloc[0].to_dict(), "end_timestamp": ends.time.iloc[0]})
    if unpaired:
        logger.warning("%d ventilation records without one start before one end are left out", unpaired)
    return pd.DataFrame(output)
