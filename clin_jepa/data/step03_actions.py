"""Step 03: extract intervention events from the WeavEHR concepts and source records, one parquet per stay."""

import gc
import logging

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.io_utils import (
    add_events,
    extraction_args,
    load_chunk_cohort,
    resolve_table_path,
    write_stay_events,
)
from clin_jepa.data.helpers.schema import CONCEPT_SOURCE, OUTPUT_COLUMNS, OUTPUT_DTYPES, empty_events_df
from clin_jepa.data.helpers.source import data_paths
from clin_jepa.data.helpers.temporal_utils import (
    build_icu_windows,
    build_intime_map,
    build_outtime_map,
    compute_hours_since_admission,
)
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.data.helpers.weavehr import Project, action_table, assign_stay_ids, optional_column
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def melt_to_long_action(
    df: pd.DataFrame,
    source_name: str,
    category: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_col: str,
    intime_map: dict,
    outtime_map: dict,
    end_col: str | None,
    is_continuous: bool,
) -> pd.DataFrame:
    """Melt wide-format action table to long-format event rows."""
    frames = []

    has_end = end_col is not None and end_col in df.columns
    id_vars = ["stay_id", timestamp_col]
    if has_end:
        id_vars.append(end_col)

    if numeric_cols:
        melted = df.melt(
            id_vars=id_vars,
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="numeric_value",
        )
        melted = melted.dropna(subset=["numeric_value"])
        melted["text_value"] = ""
        frames.append(melted)

    for col in categorical_cols:
        keep_cols = ["stay_id", timestamp_col]
        if has_end:
            keep_cols.append(end_col)
        keep_cols.append(col)
        subset = df[keep_cols].copy()
        subset = subset.dropna(subset=[col])
        subset = subset[subset[col].astype(str).str.strip() != ""]
        subset = subset.rename(columns={col: "text_value"})
        subset["variable_name"] = col
        subset["numeric_value"] = np.nan
        subset["text_value"] = subset["text_value"].astype(str)
        frames.append(subset)

    if not frames:
        return empty_events_df()

    result = pd.concat(frames, ignore_index=True)

    result["timestamp"] = pd.to_datetime(result[timestamp_col])
    if timestamp_col != "timestamp":
        result = result.drop(columns=[timestamp_col])

    result["timestamp_hours"] = compute_hours_since_admission(
        result["timestamp"], result["stay_id"], intime_map,
    )

    if has_end:
        result["end_timestamp"] = pd.to_datetime(result[end_col])
        outtimes = result["stay_id"].map(outtime_map)
        exceed_mask = result["end_timestamp"] > outtimes
        if exceed_mask.any():
            logger.info(
                "  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum(),
            )
            result.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
        result["end_timestamp_hours"] = compute_hours_since_admission(
            result["end_timestamp"], result["stay_id"], intime_map,
        )
        if end_col != "end_timestamp":
            result = result.drop(columns=[end_col])
    else:
        result["end_timestamp"] = pd.NaT
        result["end_timestamp_hours"] = np.nan

    result["domain"] = "action"
    result["source_table"] = source_name
    result["category"] = category
    result["is_continuous"] = is_continuous

    result["unit"] = result["variable_name"].map(
        lambda vn: get_unit(source_name, vn)
    )

    return result[OUTPUT_COLUMNS]


def process_action_concept(
    rule: dict,
    project: Project,
    chunk_subject_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process a single WeavEHR action concept."""
    concept = rule["concept"]
    is_continuous = rule.get("mode", "point") != "point"
    end_col = "end_time" if is_continuous else None

    logger.info("Processing %s (category=%s, continuous=%s)", concept, rule["category"], is_continuous)

    df = action_table(project, rule, chunk_subject_ids_set)

    if len(df) == 0:
        logger.warning("  No data found for %s", concept)
        return empty_events_df()

    df = assign_stay_ids(df, windows_df, "time")
    if len(df) == 0:
        logger.warning("  No rows within ICU window for %s", concept)
        return empty_events_df()

    name = rule.get("name", concept)
    numeric_cols, categorical_cols = ([], [name]) if rule.get("levels") else ([name], [])

    events = melt_to_long_action(
        df, CONCEPT_SOURCE, rule["category"], numeric_cols, categorical_cols,
        "time", intime_map, outtime_map, end_col, is_continuous,
    )

    logger.info("  Produced %d event rows", len(events))

    del df
    gc.collect()

    return events


def _item_names(codes: pd.Series) -> pd.Series:
    """Item of an extracted event code: ``INFUSION_END//221906//Norepinephrine`` is ``Norepinephrine``."""

    def name(code) -> str:
        parts = str(code).split("//") if pd.notna(code) else []
        label = parts[-1] if len(parts) > 1 else ""
        # The name is kept exactly as the source wrote it, including any spaces.
        if label.strip() and label != "None":
            return label
        return "unknown_item" + (f"_{parts[-2]}" if len(parts) > 2 else "")

    return codes.map(name)


def _rate_text(rows: pd.DataFrame, text: dict) -> pd.Series:
    """``<rate> <unit>`` of the records that have a rate, otherwise an empty string."""
    if "rate" not in text:
        return pd.Series("", index=rows.index, dtype=object)
    rate = pd.to_numeric(rows[text["rate"]], errors="coerce").astype("float32").astype(str).replace("nan", "")
    if "rate_unit" not in text:
        return rate.astype(object)
    # A source may fill its rate column for records that have none, such as a
    # bolus; only a rate with a unit is one.
    unit = rows[text["rate_unit"]].astype("string").fillna("").astype(object)
    return (rate + " " + unit).where((rate != "") & (unit != ""), "").astype(object)


def _packed_text(rows: pd.DataFrame, text: dict) -> pd.Series:
    """Details of a source record in the packed form the text templates read.

    ``rate:<rate> <unit> | <detail>`` for an infusion, ``route:<route> |
    <detail>`` for an order, otherwise the detail alone.
    """

    def column(key: str) -> pd.Series:
        if key not in text:
            return pd.Series("", index=rows.index)
        return rows[text[key]].astype("string").fillna("").astype(object)

    detail = column("detail")
    if "rate" in text:
        rate = _rate_text(rows, text)
        packed = np.where(rate != "", "rate:" + rate + " | " + detail, detail)
    elif "route" in text:
        route = column("route")
        packed = np.where(route != "", "route:" + route + " | " + detail, detail)
    else:
        packed = detail.to_numpy()
    return pd.Series(packed, index=rows.index, dtype=object).str.strip(" |")


def source_records(
    project: Project,
    name: str,
    spec: dict,
    chunk_subject_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Unharmonised records of one extraction table as canonical action events.

    ``spec`` is the table's entry under ``raw_actions`` in the dataset profile.
    It names the extracted event with one row per source record and the
    columns to read. Item names and values are the source's own. A record
    belongs to the ICU stay in which it starts.

    ``start`` names a second event of the same record that carries its start
    time. What that start event says (such as the rate that was set) is stated
    for as long as the record runs. The value of the end event (such as the
    amount that was delivered) is only known afterwards: it is a separate
    event at the later of the end and its recorded availability, so that no
    hour reads a value from its own future. A record with nothing to state
    from its start is that later event alone.

    WeavEHR extracts every table whose raw file it finds; a table it did not extract gives no events.
    """
    if name == CONCEPT_SOURCE:
        raise ValueError(f"{CONCEPT_SOURCE!r} is reserved for harmonised concepts")
    text_columns = spec.get("text", {})
    if "start" in spec and any(not str(column).startswith("start.") for column in text_columns.values()):
        raise ValueError(f"raw_actions.{name}: the text of a record with a start event may only read start.<column>")
    rows = project.event(spec["table"], spec["event"], chunk_subject_ids_set)
    start = project.event(spec["table"], spec["start"]["event"], chunk_subject_ids_set) if "start" in spec else None
    if rows is None or ("start" in spec and start is None):
        logger.info("%s: not extracted by WeavEHR", name)
        return empty_events_df()
    if rows.empty or (start is not None and start.empty):
        return empty_events_df()
    rows["_end"] = pd.NaT
    if start is not None:
        # Both events of a record carry its item and the listed record IDs.
        keys = ["subject_id", "_item", *spec["start"]["ids"]]
        for frame in (rows, start):
            frame["_item"] = frame.code.astype("string").str.split("//", n=1).str[1]
            frame.sort_values("time", kind="stable", inplace=True)
            frame["_n"] = frame.groupby(keys, dropna=False).cumcount()
        start = start.rename(columns=lambda c: c if c in {*keys, "_n"} else f"start.{c}")
        paired = rows.merge(start, on=[*keys, "_n"], how="inner")
        if len(rows) > len(paired):
            logger.warning("%s: %d records without a start event are left out", name, len(rows) - len(paired))
        rows = paired.assign(_end=paired.time, time=paired["start.time"])
    elif "time" in spec:
        rows["time"] = pd.to_datetime(rows[spec["time"]])
        if "end_time" in spec:
            rows["_end"] = pd.to_datetime(rows[spec["end_time"]])
    for column, values in spec.get("keep", {}).items():
        rows = rows[rows[column].isin(values)]
    # A record whose documented end precedes its start has no usable interval.
    rows = rows[rows.time.notna() & ~(rows._end < rows.time)]
    rows = assign_stay_ids(rows, windows_df, "time")
    if rows.empty:
        return empty_events_df()
    numeric = pd.to_numeric(rows[spec["numeric_value"]], errors="coerce") if "numeric_value" in spec else np.nan
    unit = rows[spec["unit"]].astype("string").fillna("").astype(object) if "unit" in spec else ""
    text = _packed_text(rows, text_columns)
    outtime = rows["stay_id"].map(outtime_map)
    end = pd.concat([rows._end, outtime], axis=1).min(axis=1).where(rows._end.notna())

    def events(keep, time, end, numeric, text, unit) -> pd.DataFrame:
        def hours(moment) -> pd.Series:
            return compute_hours_since_admission(pd.to_datetime(moment), rows["stay_id"], intime_map)

        end = pd.Series(end, index=rows.index, dtype="datetime64[ns]")
        interval = end.notna() & (end > time)
        frame = pd.DataFrame({
            "stay_id": rows.stay_id,
            "timestamp": time,
            "timestamp_hours": hours(time),
            "domain": "action",
            "source_table": name,
            "category": spec["category"],
            "variable_name": _item_names(rows.code),
            "numeric_value": numeric,
            "text_value": text,
            "unit": unit,
            "end_timestamp": end.where(interval),
            "end_timestamp_hours": hours(end).where(interval),
            "is_continuous": interval,
        })
        return frame.loc[keep].astype(OUTPUT_DTYPES)

    if start is None:
        return events(slice(None), rows.time, end, numeric, text, unit)
    stated = _rate_text(rows, text_columns).ne("") if "rate" in text_columns else pd.Series(True, index=rows.index)
    running = events(stated, rows.time, end, np.nan, text, "")
    if "numeric_value" not in spec:
        return running
    known = pd.concat([rows._end, pd.to_datetime(optional_column(rows, "available_time"))], axis=1).max(axis=1)
    result = events(numeric.notna() & (known < outtime), known, pd.NaT, numeric, text.where(~stated, ""), unit)
    return pd.concat([running, result], ignore_index=True)


def main() -> None:
    args = extraction_args(__doc__, weavehr=True)
    setup_logging()
    paths_config = data_paths(args)
    features_config = load_config(args.features_config)

    stay_ids, cohort = load_chunk_cohort(paths_config, args.chunk_idx, args.total_chunks)
    subject_id_set = set(cohort["subject_id"].tolist())
    intime_map = build_intime_map(cohort)
    outtime_map = build_outtime_map(cohort)
    windows_df = build_icu_windows(cohort)
    project = Project(paths_config)

    output_dir = ensure_dir(resolve_table_path(paths_config, "output.actions"))

    all_events: dict[int, list[pd.DataFrame]] = {}
    for rule in features_config["actions"]:
        events = process_action_concept(
            rule, project, subject_id_set, windows_df, intime_map, outtime_map,
        )
        add_events(all_events, events)
        del events
        gc.collect()
    # Unharmonised source records the dataset file declares, unless the feature profile switches them off.
    if features_config.get("raw_actions", {}).get("enabled", True):
        for name, spec in project.profile.get("raw_actions", {}).items():
            events = source_records(
                project, name, spec, subject_id_set, windows_df, intime_map, outtime_map,
            )
            add_events(all_events, events)

    write_stay_events(all_events, stay_ids, output_dir)


if __name__ == "__main__":
    main()
