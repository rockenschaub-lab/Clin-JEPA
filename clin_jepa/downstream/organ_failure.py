"""Hourly organ-failure states of a stay and their early-event labels."""

import warnings

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.schema import CONCEPT_SOURCE
from clin_jepa.downstream.tasks import BY_NAME


def hourly_means(obs: pd.DataFrame, n_hours: int) -> dict[str, np.ndarray]:
    """Mean numeric value per concept and hour; NaN where nothing was charted."""
    if obs.empty:
        return {}
    rows = obs.loc[(obs["source_table"] == CONCEPT_SOURCE) & obs["numeric_value"].notna()
                   & obs["step_idx"].between(0, n_hours - 1)]
    means = rows.groupby(["variable_name", "step_idx"])["numeric_value"].mean()
    out: dict[str, np.ndarray] = {}
    for (concept, step), value in means.items():
        out.setdefault(concept, np.full(n_hours, np.nan, dtype=np.float32))[int(step)] = value
    return out


def neighbours(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the last observed hour at or before each hour (-1 if none) and of the next one (n if none)."""
    hours = np.arange(len(values))
    known = ~np.isnan(values)
    last = np.maximum.accumulate(np.where(known, hours, -1))
    following = np.minimum.accumulate(np.where(known, hours, len(values))[::-1])[::-1]
    return last, following


def fill(values: np.ndarray, max_hours: np.ndarray | int, out: np.ndarray | None = None) -> np.ndarray:
    """Carry the last observation forward, then the next one backward, each for at most ``max_hours``."""
    hours, (last, following) = np.arange(len(values)), neighbours(values)
    out = values.astype(np.float32) if out is None else out
    for source, distance in ((last, hours - last), (following, following - hours)):
        inside = (source >= 0) & (source < len(values))
        limit = max_hours if np.isscalar(max_hours) else max_hours[np.clip(source, 0, len(values) - 1)]
        ok = np.isnan(out) & inside & (distance <= limit)
        out[ok] = values[source[ok]]
    return out


def interpolate(values: np.ndarray, max_gap: int, out: np.ndarray) -> np.ndarray:
    """Linear interpolation between two observations at most ``max_gap`` hours apart."""
    hours, (last, following) = np.arange(len(values)), neighbours(values)
    between = np.isnan(values) & (last >= 0) & (following < len(values))
    between &= (following - last) <= max_gap
    t, t0, t1 = hours[between], last[between], following[between]
    v0, v1 = values[t0].astype(np.float64), values[t1].astype(np.float64)
    out[t] = v0 + (v1 - v0) * ((t - t0) / (t1 - t0))
    return out


def fill_lactate(values: np.ndarray) -> np.ndarray:
    """Interpolate gaps of at most 6 h; then carry a value below 2 mmol/L indefinitely and any other for 6 h."""
    out = interpolate(values, 6, values.astype(np.float32))
    reach = np.where(values < 2.0, 10 ** 9, 6)
    return fill(values, reach, out)


def early_event_label(state: np.ndarray, horizon: int) -> np.ndarray:
    """1 if the event starts within ``horizon`` hours, 0 if only non-events are confirmed there; NaN during an event."""
    n = len(state)
    events = np.concatenate([[0], np.cumsum(state == 1)])
    stable = np.concatenate([[0], np.cumsum(state == 0)])
    start = np.arange(n) + 1
    end = np.minimum(np.arange(n) + horizon, n - 1) + 1
    end = np.maximum(end, start)
    label = np.full(n, np.nan, dtype=np.float32)
    label[(stable[end] - stable[start]) > 0] = 0.0
    label[(events[end] - events[start]) > 0] = 1.0
    label[state == 1] = np.nan
    return label


def active_hours(steps: np.ndarray, n_hours: int) -> np.ndarray:
    active = np.zeros(n_hours, dtype=bool)
    active[steps[(steps >= 0) & (steps < n_hours)]] = True
    return active


def renal_replacement(act: pd.DataFrame, n_hours: int) -> np.ndarray:
    """Hours on renal replacement therapy, each charted hour carried forward for 48 h."""
    if act.empty:
        return np.zeros(n_hours, dtype=bool)
    name = act["variable_name"].astype(str)
    procedure = (act["category"] == "procedure") & name.str.contains(r"CRRT|Hemodialy", case=False) \
        & ~name.str.contains("Catheter", case=False)
    charted = active_hours(act.loc[procedure, "step_idx"].to_numpy(dtype=np.int64), n_hours)
    hours = np.arange(n_hours)
    last = np.maximum.accumulate(np.where(charted, hours, -(10 ** 9)))
    return (hours - last) <= 48


def states(obs: pd.DataFrame, act: pd.DataFrame, n_hours: int) -> dict[str, np.ndarray]:
    """Per task: 1 where the organ is failing, 0 where that is ruled out, NaN where it cannot be told."""
    means = hourly_means(obs, n_hours)

    def get(concept: str) -> np.ndarray:
        return means.get(concept, np.full(n_hours, np.nan, dtype=np.float32))

    def state(event: np.ndarray, non_event: np.ndarray) -> np.ndarray:
        out = np.full(n_hours, np.nan, dtype=np.float32)
        out[non_event] = 0.0
        out[event] = 1.0
        return out

    out = {}
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pressure = get("mean_arterial_pressure")
        lactate = fill_lactate(get("lactate"))
        out["circ_failure_8h"] = state((pressure < 65) & (lactate > 2.0), (pressure >= 65) & (lactate <= 2.0))

        # Respiratory failure is a PaO2/FiO2 ratio below 100 while ventilated.
        # WeavEHR has no such ratio, and none is derived here from specimens
        # and settings charted at different times, so the state stays unknown.
        out["resp_failure_24h"] = np.full(n_hours, np.nan, dtype=np.float32)

        creatinine_raw = get("creatinine")
        creatinine = interpolate(creatinine_raw, 48, fill(creatinine_raw, 48))
        renal_sofa = fill(get("sofa_renal"), 24)
        on_rrt = renal_replacement(act, n_hours)
        high = creatinine >= 4.0
        if not np.isnan(creatinine_raw).all():
            high |= creatinine >= 3.0 * float(np.nanmin(creatinine_raw))
        event = on_rrt | high | (renal_sofa >= 3)
        out["kidney_failure_48h"] = state(event, ~event & ~np.isnan(creatinine) & ~np.isnan(renal_sofa))

        inr = get("prothrombin_time_international_normalized_ratio")
        meld = (9.57 * np.log(np.clip(fill(get("creatinine"), 1), 1.0, None))
                + 3.78 * np.log(np.clip(fill(get("total_bilirubin"), 1), 1.0, None))
                + 11.2 * np.log(np.clip(fill(inr, 1), 1.0, None)) + 6.43)
        out["liver_failure_48h"] = state(meld > 30.0, meld <= 30.0)

        glucose = get("glucose")
        out["hyperglycemia_8h"] = state(glucose > 180.0, glucose <= 180.0)
    return out


def labels(obs: pd.DataFrame, act: pd.DataFrame, n_hours: int) -> dict[str, np.ndarray]:
    return {task: early_event_label(state, BY_NAME[task].horizon) for task, state in states(obs, act, n_hours).items()}
