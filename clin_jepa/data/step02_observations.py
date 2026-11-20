"""Step 02: extract observation events from the WeavEHR concepts, one parquet per stay."""

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
from clin_jepa.data.helpers.schema import CONCEPT_SOURCE, OUTPUT_COLUMNS, empty_events_df
from clin_jepa.data.helpers.source import data_paths
from clin_jepa.data.helpers.temporal_utils import build_icu_windows, build_intime_map, compute_hours_since_admission
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.data.helpers.weavehr import Project, assign_stay_ids, concept_table
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def melt_to_long(
    df: pd.DataFrame,
    source_name: str,
    category: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_col: str,
    intime_map: dict,
) -> pd.DataFrame:
    """Melt wide-format table to long-format event rows."""
    frames = []

    if numeric_cols:
        melted = df.melt(
            id_vars=["stay_id", timestamp_col],
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="numeric_value",
        )
        melted = melted.dropna(subset=["numeric_value"])
        melted["text_value"] = ""
        frames.append(melted)

    for col in categorical_cols:
        subset = df[["stay_id", timestamp_col, col]].copy()
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
        result["timestamp"], result["stay_id"], intime_map
    )

    result["domain"] = "observation"
    result["source_table"] = source_name
    result["category"] = category
    result["end_timestamp"] = pd.NaT
    result["end_timestamp_hours"] = np.nan
    result["is_continuous"] = False

    result["unit"] = result["variable_name"].map(
        lambda vn: get_unit(source_name, vn)
    )

    return result[OUTPUT_COLUMNS]


def process_observation_concept(
    rule: dict,
    project: Project,
    chunk_subject_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
) -> pd.DataFrame:
    """Process a single WeavEHR concept."""
    concept = rule["concept"]
    logger.info("Processing %s (category=%s)", concept, rule["category"])

    df = concept_table(project, rule, chunk_subject_ids_set)

    if len(df) == 0:
        logger.warning("  No data found for %s", concept)
        return empty_events_df()

    df = assign_stay_ids(df, windows_df, "time")
    if len(df) == 0:
        logger.warning("  No rows matched ICU windows for %s", concept)
        return empty_events_df()
    logger.info("  After ICU window matching: %d rows", len(df))

    events = melt_to_long(df, CONCEPT_SOURCE, rule["category"], [concept], [], "time", intime_map)

    logger.info("  Produced %d event rows", len(events))

    del df
    gc.collect()

    return events


def main() -> None:
    args = extraction_args(__doc__, weavehr=True)
    setup_logging()
    paths_config = data_paths(args)
    features_config = load_config(args.features_config)

    stay_ids, cohort = load_chunk_cohort(paths_config, args.chunk_idx, args.total_chunks)
    subject_id_set = set(cohort["subject_id"].tolist())
    intime_map = build_intime_map(cohort)
    windows_df = build_icu_windows(cohort)
    project = Project(paths_config)

    output_dir = ensure_dir(resolve_table_path(paths_config, "output.observations"))

    all_events: dict[int, list[pd.DataFrame]] = {}
    for rule in features_config["observations"]:
        events = process_observation_concept(rule, project, subject_id_set, windows_df, intime_map)
        add_events(all_events, events)
        del events
        gc.collect()

    write_stay_events(all_events, stay_ids, output_dir)


if __name__ == "__main__":
    main()
