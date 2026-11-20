"""Hourly and stay-level labels of the downstream tasks, from the cohort, the hourly bins and the WeavEHR project."""

import argparse
import json
import logging
import pickle
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.source import load_data_config
from clin_jepa.data.helpers.weavehr import Project, action_table, icu_stays, native_ids
from clin_jepa.downstream import organ_failure
from clin_jepa.downstream.tasks import CONTEXT, ROLLOUT_HORIZON, TASK_SETS, TASKS
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

EVENT_COLUMNS = ["step_idx", "category", "source_table", "variable_name", "numeric_value", "text_value"]
VASOPRESSOR = re.compile(
    r"norepinephrine|levophed|epinephrine|phenylephrine|neo-synephrine|vasopressin|dopamine|dobutamine|milrinone",
    re.IGNORECASE)
NOT_VASOPRESSOR = re.compile(r"racepinephrine|lidocaine|xylocaine", re.IGNORECASE)
IV_ROUTE = re.compile(r"^route:IV", re.IGNORECASE)
BLOOD_PRODUCT = re.compile(
    r"packed red blood cell|packed rbc|red blood cell|fresh frozen plasma|\bffp\b|platelet|cryoprecipitate|"
    r"autologous blood|whole blood", re.IGNORECASE)
NOT_BLOOD_PRODUCT = re.compile(r"plasma[- ]?lyte|blood cultur|blood_flow|pheresis", re.IGNORECASE)


def onset_label(onset: float, n_hours: int, horizon: int) -> np.ndarray:
    """1 where the onset lies within ``horizon`` hours ahead, NaN from the onset on, 0 elsewhere."""
    t = np.arange(n_hours, dtype=np.float64)
    label = np.zeros(n_hours, dtype=np.float32)
    if np.isfinite(onset):
        label[(t < onset) & (onset <= t + horizon)] = 1.0
        label[t >= onset] = np.nan
    return label


def first_hour(mask: np.ndarray, steps: np.ndarray) -> float:
    return float(steps[mask].min()) if mask.any() else float("inf")


def any_ahead(flag: np.ndarray, horizon: int, n_hours: int) -> np.ndarray:
    """For each hour h: whether ``flag`` holds anywhere in (h, h + horizon]."""
    total = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    h = np.arange(n_hours)
    return (total[np.clip(h + horizon + 1, 0, len(flag))] - total[np.clip(h + 1, 0, len(flag))]) > 0


def extreme_ahead(values: np.ndarray, horizon: int, n_hours: int, largest: bool) -> np.ndarray:
    """For each hour h: the largest (or smallest) finite value in (h, h + horizon]; NaN if there is none."""
    sign = 1.0 if largest else -1.0
    padded = np.concatenate([np.where(np.isfinite(values), sign * values, -np.inf), np.full(horizon, -np.inf)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, horizon)
    best = windows[np.clip(np.arange(n_hours) + 1, 0, len(windows) - 1)].max(axis=1)
    return np.where(np.isfinite(best), sign * best, np.nan)


def next_true(flag: np.ndarray) -> np.ndarray:
    """Index of the first True at or after each position, ``len(flag)`` if none; one entry past the end."""
    n = len(flag)
    index = np.where(flag, np.arange(n), n)
    return np.append(np.minimum.accumulate(index[::-1])[::-1], n).astype(np.int64)


def covered_hours(episodes: tuple[np.ndarray, np.ndarray] | None, n_hours: int) -> np.ndarray:
    """Hours overlapped by an episode ``[start, end)`` given in hours since ICU admission."""
    on = np.zeros(n_hours, dtype=bool)
    for start, end in zip(*episodes) if episodes else ():
        if np.isfinite(start) and np.isfinite(end):
            lo, hi = max(0, int(np.floor(start))), min(n_hours - 1, int(np.ceil(end)) - 1)
            if hi >= lo:
                on[lo : hi + 1] = True
    return on


def starting_hours(episodes: tuple[np.ndarray, np.ndarray] | None, n_hours: int) -> np.ndarray:
    started = np.zeros(n_hours, dtype=bool)
    for start in episodes[0] if episodes else ():
        if np.isfinite(start) and 0 <= start and int(np.floor(start)) < n_hours:
            started[int(np.floor(start))] = True
    return started


def episode_duration(episodes: tuple[np.ndarray, np.ndarray] | None, end_hour: float) -> float:
    """Union duration of recorded intervals, clipped to the observed stay."""
    merged = []
    for start, end in sorted(zip(*episodes)) if episodes is not None else ():
        if not np.isfinite([start, end]).all():
            continue
        start, end = max(0, start), min(end_hour, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return float(sum(end - start for start, end in merged))


def extubation_failure(episodes: tuple[np.ndarray, np.ndarray] | None, n_hours: float) -> tuple[float, float]:
    """Reintubation within 48 h of the first liberation, for stays ventilated at hour 23 and liberated later.

    Returns the label and the liberation hour; the label is NaN for stays that are not at risk.
    """
    if episodes is None:
        return np.nan, np.nan
    start, end = (np.asarray(x, dtype=np.float64) for x in episodes)
    keep = np.isfinite(start) & np.isfinite(end) & (start < n_hours) & (end > 0)
    start, end = np.maximum(start[keep], 0.0), np.minimum(end[keep], n_hours)
    order = np.argsort(start, kind="stable")
    merged: list[list[float]] = []
    for a, b in zip(start[order], end[order]):
        if merged and a - merged[-1][1] < 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([float(a), float(b)])
    if not merged:
        return np.nan, np.nan
    starts, ends = np.array(merged).T
    liberations = ends[ends <= n_hours - 1.0]
    ventilated_at_23 = bool(((starts < CONTEXT) & (ends > CONTEXT - 1)).any())
    if not ventilated_at_23 or len(liberations) == 0:
        return np.nan, (float(liberations[0]) if len(liberations) else np.nan)
    liberation = float(liberations[0])
    if liberation < CONTEXT:
        return np.nan, liberation
    later = starts[starts > liberation]
    return (1.0 if len(later) and later.min() <= liberation + 48 else 0.0), liberation


def label_stay(job: tuple) -> tuple[pd.DataFrame, dict]:
    """Hourly labels of one stay and the per-stay quantities the stay-level labels are built from."""
    path, death_hour, sepsis_onset, arrived_ventilated, ventilation, cohort = job
    with open(path, "rb") as f:
        stay = pickle.load(f)
    n_hours, obs, act = int(stay["n_steps"]), stay["obs_events"], stay["act_events"]
    hours = np.arange(n_hours)
    n_ahead = n_hours + ROLLOUT_HORIZON + 1
    alive = np.arange(n_ahead) < death_hour

    act = act if len(act) else pd.DataFrame(columns=EVENT_COLUMNS)
    obs = obs if len(obs) else pd.DataFrame(columns=EVENT_COLUMNS)
    steps = act["step_idx"].to_numpy(dtype=np.int64)
    category = act["category"].astype(str).to_numpy()
    name, text = act["variable_name"].astype(str), act["text_value"].astype(str)
    value = pd.to_numeric(act["numeric_value"], errors="coerce").to_numpy(dtype=np.float64)

    def hourly_flag(mask: np.ndarray, length: int = n_hours) -> np.ndarray:
        flag = np.zeros(length, dtype=bool)
        flag[steps[mask & (steps >= 0) & (steps < n_hours)]] = True
        return flag

    def hourly_max(mask: np.ndarray, values: np.ndarray, which: np.ndarray) -> np.ndarray:
        out = np.full(n_hours, np.nan)
        ok = mask & (which >= 0) & (which < n_hours) & np.isfinite(values)
        np.fmax.at(out, which[ok], values[ok])
        return out

    infusion = (category == "vasopressor") & (value > 0)
    named = np.isin(category, ["medication_order", "medication_admin"]) & name.str.contains(VASOPRESSOR).to_numpy() \
        & ~name.str.contains(NOT_VASOPRESSOR).to_numpy()
    order = named & (text.str.contains(IV_ROUTE).to_numpy() | ~text.str.startswith("route:").to_numpy())
    vasopressor_onset = min(first_hour(infusion, steps), first_hour(order, steps))
    ventilated_charted = (category == "ventilation") & (text == "invasive").to_numpy()

    labels = organ_failure.labels(obs, act, n_hours)
    labels["sepsis3_8h"] = onset_label(sepsis_onset, n_hours, 8) if sepsis_onset is not None \
        else np.full(n_hours, np.nan, dtype=np.float32)
    death = onset_label(death_hour, n_hours, 24)
    intubation = np.full(n_hours, np.nan, dtype=np.float32) if arrived_ventilated \
        else onset_label(first_hour(ventilated_charted, steps), n_hours, 24)
    components = np.stack([death, onset_label(vasopressor_onset, n_hours, 24), intubation])
    composite = ((components == 1).sum(axis=0) > 0).astype(np.float32)
    composite[np.isnan(components).all(axis=0) | (hours >= death_hour)] = np.nan
    labels["decompensation_24h"], labels["composite_deterioration_24h"] = death, composite

    labels["vaso_onset_8h"] = onset_label(vasopressor_onset, n_hours, 8)
    labels["vaso_onset_24h"] = onset_label(vasopressor_onset, n_hours, 24)
    on_vasopressor = hourly_flag(infusion)
    dose = hourly_max(infusion & (name == "norepinephrine_equivalent_dose").to_numpy(), value, steps)
    dose_ahead = extreme_ahead(np.concatenate([dose, np.full(n_ahead - n_hours, np.nan)]), 24, n_hours, largest=True)
    escalated = (np.isfinite(dose_ahead) & (dose_ahead >= 2.0 * dose)).astype(np.float32)
    labels["vaso_escalation_24h"] = np.where(on_vasopressor & np.isfinite(dose) & (dose > 0), escalated, np.nan)
    off = ~hourly_flag(infusion, n_ahead) & alive
    run_length = np.where(off, next_true(~off)[:-1] - np.arange(n_ahead), 0)
    labels["vaso_liberation_24h"] = np.where(
        on_vasopressor, any_ahead(run_length >= 24, 24, n_hours).astype(np.float32), np.nan)

    ventilated_ahead = covered_hours(ventilation.get("invasive"), n_ahead)
    ventilated = ventilated_ahead[:n_hours]
    intubated_ahead = any_ahead(starting_hours(ventilation.get("invasive"), n_ahead), 24, n_hours)
    labels["imv_onset_24h"] = np.where(ventilated, np.nan, intubated_ahead.astype(np.float32))
    first_off = next_true(~ventilated_ahead)[np.minimum(hours + 1, n_ahead)]
    liberated = (first_off <= hours + 48) & (next_true(ventilated_ahead)[np.minimum(first_off, n_ahead)] > hours + 48)
    liberated &= death_hour > hours + 48
    labels["imv_liberation_48h"] = np.where(ventilated, liberated.astype(np.float32), np.nan)

    # WeavEHR has no renal-replacement status; a charted CRRT setting marks an hour of it.
    on_rrt = hourly_flag(category == "dialysis")
    first_rrt = float(np.argmax(on_rrt)) if on_rrt.any() else float("inf")
    labels["rrt_onset_48h"] = onset_label(first_rrt, n_hours, 48)

    obs_steps = obs["step_idx"].to_numpy(dtype=np.int64)
    obs_value = pd.to_numeric(obs["numeric_value"], errors="coerce").to_numpy(dtype=np.float64)

    def observed(obs_category: str, variable: str) -> np.ndarray:
        mask = (obs["category"].astype(str) == obs_category) & (obs["variable_name"].astype(str) == variable)
        return hourly_max(mask.to_numpy(), obs_value, obs_steps)

    sofa = observed("score", "sofa")
    sofa_ahead = np.concatenate([sofa, np.full(n_ahead - n_hours, np.nan)])
    highest, lowest = (extreme_ahead(sofa_ahead, 48, n_hours, largest) for largest in (True, False))
    at_risk = np.isfinite(sofa) & np.isfinite(highest)
    labels["sofa_worsen_2_48h"] = np.where(at_risk, (highest - sofa >= 2).astype(np.float32), np.nan)
    labels["sofa_improve_2_48h"] = np.where(at_risk, (lowest - sofa <= -2).astype(np.float32), np.nan)

    transfused = hourly_flag((category == "fluid_med_input") & name.str.contains(BLOOD_PRODUCT).to_numpy()
                             & ~name.str.contains(NOT_BLOOD_PRODUCT).to_numpy() & (value > 0), n_ahead)
    started = transfused & ~np.concatenate([[False], transfused[:-1]])
    labels["transfusion_24h"] = any_ahead(started, 24, n_hours).astype(np.float32)

    frame = pd.DataFrame({"stay_id": np.full(n_hours, int(stay["stay_id"]), dtype=np.int64),
                          "hour": hours.astype(np.int32)})
    for task in TASKS:
        if task.hourly:
            frame[task.name] = labels[task.name].astype(np.float32)

    tracheostomy = np.flatnonzero(covered_hours(ventilation.get("tracheostomy"), n_hours))
    stage = observed("organ_failure", "aki")
    failure, liberation = extubation_failure(ventilation.get("invasive"), float(n_hours))
    if failure == 0 and np.isfinite(death_hour) and death_hour <= liberation + 48:
        failure = np.nan
    vasopressor_free = np.nan if n_hours <= CONTEXT else float(
        not on_vasopressor[CONTEXT:192].any() and not death_hour < 192)
    record = {
        "stay_id": int(stay["stay_id"]),
        "prolonged_imv_96h": float(ventilated.sum() >= 96),
        "tracheostomy_ever": (np.nan if len(tracheostomy) and tracheostomy[0] < CONTEXT
                              else float(len(tracheostomy) > 0)),
        "extubation_failure_48h": failure,
        "new_rrt_after_24h": np.nan if on_rrt[:CONTEXT].any() else float(on_rrt.any()),
        "aki_kdigo23_after_24h": np.nan if not np.isfinite(stage).any() or (stage[:CONTEXT] >= 2).any()
        else float((stage >= 2).any()),
        "vasopressor_free_7d": vasopressor_free,
    }
    supported_stay = set()
    if cohort["ventilation_assessable"]:
        # The cohort caps icu_los_hours, so truncation needs the actual stay.
        los = cohort["los_icu"] * 24
        duration = episode_duration(ventilation.get("invasive"), min(n_hours, los))
        # A truncated episode can establish a positive, but not a negative.
        if duration >= 96 or los <= n_hours:
            record["prolonged_imv_96h"] = float(duration >= 96)
            supported_stay.add("prolonged_imv_96h")
    if np.isfinite(stage[:CONTEXT]).any() and np.isfinite(stage[CONTEXT:]).any():
        supported_stay.add("aki_kdigo23_after_24h")
    mask_weavehr_labels(frame, record, n_hours, cohort, supported_stay)
    return frame, record


def ventilation_episodes(project: Project, intime: pd.Series) -> tuple[dict[int, dict], set[int]]:
    """Per stay the ventilation episodes in hours since ICU admission, and the stays that arrive ventilated."""
    features = load_data_config(project.paths, "features")
    rule = next(rule for rule in features["actions"] if rule.get("name") == "ventilation")
    table = action_table(project, rule, None)
    if table.empty:
        return {}, set()
    table = table.rename(columns={"time": "starttime", "end_time": "endtime", "ventilation": "ventilation_status"})
    table = table[table["ventilation_status"].isin(["invasive", "tracheostomy"])]
    table = table.merge(intime.rename("intime"), left_on="stay_id", right_index=True)
    invasive = table["ventilation_status"] == "invasive"
    arrived = table[invasive & (table["starttime"] < table["intime"]) & (table["endtime"] >= table["intime"])]
    for column in ("starttime", "endtime"):
        table[column] = (table[column] - table["intime"]).dt.total_seconds() / 3600.0
    episodes: dict[int, dict] = {}
    for (stay_id, status), rows in table.groupby(["stay_id", "ventilation_status"]):
        episodes.setdefault(int(stay_id), {})[status] = (rows["starttime"].to_numpy(), rows["endtime"].to_numpy())
    return episodes, set(arrived["stay_id"].astype(int))


def sepsis_onset_hour(septic: float, onset: float) -> float | None:
    """Onset hour for ``onset_label``: None if unknown, inf without sepsis."""
    if pd.isna(septic):
        return None
    if septic == 0:
        return float("inf")
    # Whole hours, as for every other onset: the hour in which sepsis begins
    # already contains it and must be masked, not labelled positive.
    return None if np.isnan(onset) else float(np.floor(onset))


def stay_outcomes(cohort: pd.DataFrame, project: Project) -> pd.DataFrame:
    """Outcomes read from the cohort and the WeavEHR project's death, admission and ICU-stay events, one row per
    cohort stay."""
    out = cohort[["stay_id", "subject_id", "hadm_id", "icu_intime", "icu_outtime"]].copy()
    out["intime"] = cohort.icu_intime
    out["hospital_mortality"] = pd.to_numeric(cohort.hospital_expire_flag, errors="raise")
    follow_up = follow_up_outcomes(project, cohort)
    for column in follow_up:
        out[column] = follow_up[column]
    out.loc[out.death_in_icu.eq(0), "death_hour"] = np.inf
    # icu_los_hours is capped in the cohort; los_icu (days) is the actual stay.
    los = cohort.los_icu * 24
    for n in (7, 14):
        out[f"los_gt_{n}d"] = np.where(los.notna(), los > n * 24, np.nan)
    out["sepsis_ever"] = pd.to_numeric(cohort.sepsis3, errors="raise")
    onset = (pd.to_datetime(cohort.sepsis_sofa_time) - cohort.icu_intime).dt.total_seconds() / 3600
    out["sepsis_onset_after_24h"] = np.where(
        out.sepsis_ever.eq(1) & onset.gt(24), 1.0,
        np.where(out.sepsis_ever.eq(0), 0.0, np.nan),
    )
    return out


FOLLOW_UP_OUTCOMES = [
    "death_in_icu", "death_hour", "icu_mortality", "mortality_7d", "mortality_14d", "mortality_30d", "mortality_90d",
    "discharge_home", "hospital_readmission_30d", "hospital_readmission_60d", "icu_readmission_same_hosp",
]


def follow_up_outcomes(project: Project, cohort: pd.DataFrame) -> pd.DataFrame:
    """Death, discharge and readmission outcomes of each cohort stay, as the original pipeline defined them.

    Only for a dataset whose definition declares ``cohort.outcomes``: that names the WeavEHR events
    and asserts follow-up complete enough for a missing death record or a missing later admission to
    count as a negative. Everywhere else these stay unknown.
    """
    out = pd.DataFrame(np.nan, index=cohort.index, columns=FOLLOW_UP_OUTCOMES)
    spec = project.profile["cohort"].get("outcomes") or {}
    missing = sorted(
        f"{source['table']}/{source['event']}"
        for source in spec.values()
        if not project.extracted_path(source["table"], source["event"]).exists()
    )
    if not spec or missing:
        if missing:
            logger.warning("Follow-up outcomes stay unknown: the WeavEHR project lacks %s", ", ".join(missing))
        return out
    hour = pd.Timedelta(hours=1)
    # No death event file means that no death was recorded in a supplied table.
    deaths = project.event(spec["death"]["table"], spec["death"]["event"])
    first = deaths.groupby("subject_id").time.min() if deaths is not None and len(deaths) else pd.Series()
    died = pd.to_datetime(cohort.subject_id.map(first))
    # A death known by date only carries a fixed time of day; it can date the
    # death but cannot place it inside an ICU stay.
    timed = died.notna() & (died.dt.strftime("%H:%M:%S") != spec["death"].get("date_only_time"))
    in_icu = timed & (cohort.icu_intime <= died) & (died <= cohort.icu_outtime + hour)
    out["death_in_icu"] = in_icu.astype(float)
    out.loc[in_icu, "death_hour"] = np.floor((died - cohort.icu_intime) / hour)[in_icu]
    day, admitted = died.dt.normalize(), cohort.icu_intime.dt.normalize()
    out["icu_mortality"] = (died.notna() & (day >= admitted) & (day <= cohort.icu_outtime.dt.normalize())).astype(
        float
    )
    days = (day - admitted).dt.days
    for n in (7, 14, 30, 90):
        out[f"mortality_{n}d"] = (died.notna() & (days >= 0) & (days <= n)).astype(float)

    admissions = project.required_event(spec["hospital_admission"])[["subject_id", "hadm_id", "time"]]
    discharge = project.required_event(spec["hospital_discharge"])
    for frame in (admissions, discharge):
        frame["hadm_id"] = native_ids(frame["hadm_id"])
    discharge["location"] = discharge.code.str.split("//").str[-1]
    discharge = discharge[["hadm_id", "time", "location"]].drop_duplicates("hadm_id")
    admissions = admissions.drop_duplicates("hadm_id").merge(
        discharge, on="hadm_id", how="left", suffixes=("", "_discharge")
    )
    admissions = admissions.sort_values(["subject_id", "time"]).reset_index(drop=True)
    gap = (admissions.groupby("subject_id").time.shift(-1) - admissions.time_discharge) / pd.Timedelta(days=1)
    for n in (30, 60):
        admissions[f"hospital_readmission_{n}d"] = ((gap >= 0) & (gap <= n)).astype(float)
    home = set(spec["hospital_discharge"]["home"])
    known = admissions.location.notna() & admissions.location.ne(spec["hospital_discharge"]["event"])
    admissions["discharge_home"] = np.where(known, admissions.location.isin(home), np.nan).astype(float)
    columns = ["discharge_home", "hospital_readmission_30d", "hospital_readmission_60d"]
    matched = pd.DataFrame({"hadm_id": native_ids(cohort.hadm_id)}).merge(
        admissions[["hadm_id"] + columns], on="hadm_id", how="left"
    )
    for column in columns:
        out[column] = matched[column].to_numpy()
    for column in columns[1:]:
        out[column] = out[column].fillna(0.0)
    # A later ICU stay in the same hospital admission, among all of its ICU stays.
    stays = icu_stays(project)
    order = stays.groupby("hadm_id").icu_intime.rank(method="dense")
    later = (order < order.groupby(stays.hadm_id).transform("max")).astype(float).where(stays.hadm_id.notna())
    out["icu_readmission_same_hosp"] = cohort.stay_id.map(dict(zip(stays.stay_id, later))).to_numpy()
    return out


# These tasks use observed, harmonised measurements. Treatment-onset, absence,
# ventilation-liberation and RRT definitions need additional coverage validation;
# the original MIMIC benchmark's implicit negatives cannot be transferred safely.
WEAVEHR_HOURLY_TASKS = {
    "circ_failure_8h", "liver_failure_48h", "hyperglycemia_8h",
    "sofa_worsen_2_48h", "sofa_improve_2_48h",
}


def mask_weavehr_labels(
    frame: pd.DataFrame, record: dict, n_hours: int,
    metadata: dict | None = None, supported_stay: set | None = None,
) -> None:
    metadata, supported_stay = metadata or {}, supported_stay or set()
    supported_hourly = set(WEAVEHR_HOURLY_TASKS)
    if metadata.get("sepsis_assessable", False):
        supported_hourly.add("sepsis3_8h")
    if metadata.get("ventilation_assessable", False):
        supported_hourly.add("imv_onset_24h")
    for task in TASKS:
        if task.hourly:
            if task.name not in supported_hourly:
                frame[task.name] = np.nan
            else:
                # The full target window must lie inside the observed episode,
                # not beyond discharge or the trajectory truncation boundary.
                frame.loc[frame.hour + task.horizon >= n_hours, task.name] = np.nan
    for name in record:
        if name != "stay_id" and name not in supported_stay:
            record[name] = np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream.yaml")
    parser.add_argument("--weavehr_project", required=True, help="WeavEHR project the cohort was built from")
    parser.add_argument("--dataset", required=True, help="WeavEHR name of the dataset, e.g. mimic-iv or aumc")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    data_dir = Path(config["data_dir"])
    cohort = pd.read_csv(data_dir / "cohort" / "cohort.csv",
                         parse_dates=["icu_intime", "icu_outtime", "sepsis_sofa_time"])
    project = Project({"weavehr": {"project": args.weavehr_project}, "dataset": args.dataset})
    outcomes = stay_outcomes(cohort, project)
    episodes, arrived = ventilation_episodes(project, outcomes.set_index("stay_id")["intime"])

    sepsis_hours = (cohort["sepsis_sofa_time"] - cohort["icu_intime"]).dt.total_seconds() / 3600.0
    jobs = []
    # Which outcomes WeavEHR can establish (sepsis and ventilation where it produced their concepts),
    # and each stay's actual length.
    ventilation = all(project.concept_path(name).exists() for name in ("ventilation_start", "ventilation_end"))
    flags = cohort[["los_icu"]].assign(
        sepsis_assessable=cohort.sepsis3.notna(), ventilation_assessable=ventilation
    ).to_dict("records")
    for stay_id, death_hour, septic, onset, stay_flags in zip(cohort["stay_id"], outcomes["death_hour"],
                                                              cohort["sepsis3"], sepsis_hours, flags):
        path = data_dir / "discretized" / f"{int(stay_id)}.pkl"
        if path.exists():
            sepsis_onset = sepsis_onset_hour(septic, onset)
            jobs.append((path, float(death_hour), sepsis_onset, int(stay_id) in arrived,
                         episodes.get(int(stay_id), {}), stay_flags))
    with Pool(config["n_workers"]) as pool:
        results = pool.map(label_stay, jobs, chunksize=200)

    output_dir = ensure_dir(Path(config["output_dir"]) / "labels")
    if not results:
        raise ValueError("No discretized stays available for downstream labels")
    hourly = pd.concat([frame for frame, _ in results], ignore_index=True)
    hourly.to_parquet(output_dir / "hourly.parquet", index=False)
    stay = outcomes.merge(pd.DataFrame([record for _, record in results]), on="stay_id", how="left")
    names = list(dict.fromkeys(n for task_set in TASK_SETS.values() for n in task_set if n not in hourly.columns))
    stay[["stay_id"] + names].astype({name: np.float32 for name in names}).to_parquet(
        output_dir / "stay.parquet", index=False)
    coverage = {
        "hourly": {c: int(hourly[c].notna().sum()) for c in hourly if c not in {"stay_id", "hour"}},
        "stay": {c: int(stay[c].notna().sum()) for c in names},
        "definition": (
            "Pinned WeavEHR definitions. Negative sepsis/ventilation labels mean no algorithm-detected "
            "event in the supplied source tables, not proof of clinical absence. Missing source coverage "
            "and incomplete horizons remain NaN; unpaired ventilation markers are left out. AKI after 24 h requires "
            "observed stages before and after 24 h. Prolonged IMV uses exact union duration."
        ),
        "remaining_masks": {
            "vasopressor_tasks": "Incomplete drug catalogue",
            "rrt_tasks": "No unified RRT status mapping",
            "ventilation_liberation_and_extubation": "Need validated death timing and post-liberation follow-up",
            "fixed_horizon_mortality_and_readmission": (
                "Only for sources whose WeavEHR profile declares cohort.outcomes (MIMIC-IV); unknown elsewhere"
            ),
            "transfusion_and_tracheostomy": "No complete shared event mapping",
        },
    }
    (output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    logger.info("%d stays, %d stay-hours", len(stay), len(hourly))


if __name__ == "__main__":
    main()
