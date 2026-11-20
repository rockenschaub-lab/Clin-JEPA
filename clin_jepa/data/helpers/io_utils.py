"""Table loading, cohort chunking and per-stay event writing for the data pipeline."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.schema import validate_schema
from clin_jepa.data.helpers.source import add_data_arguments
from clin_jepa.utils import expand_path as resolve_path

logger = logging.getLogger(__name__)

def resolve_table_path(paths_config: dict, dotpath: str) -> Path:
    """Resolve a dot-notation path to an absolute file path."""
    node = paths_config
    for key in dotpath.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(
                f"Key '{key}' not found in config at path '{dotpath}'. "
                f"Available keys: {list(node.keys()) if isinstance(node, dict) else 'N/A'}"
            )
        node = node[key]
    if not isinstance(node, str):
        raise ValueError(
            f"Dotpath '{dotpath}' did not resolve to a file path string. Got {type(node).__name__}: {node}"
        )
    return resolve_path(node)


def chunk_stay_ids(
    stay_ids: np.ndarray,
    chunk_idx: int,
    total_chunks: int,
) -> np.ndarray:
    """Split stay_ids into chunks for SLURM array parallelization."""
    if total_chunks < 1:
        raise ValueError(f"total_chunks must be >= 1, got {total_chunks}")
    if not 0 <= chunk_idx < total_chunks:
        raise ValueError(f"chunk_idx ({chunk_idx}) must be in [0, {total_chunks})")

    chunks = np.array_split(stay_ids, total_chunks)
    selected = chunks[chunk_idx]
    logger.info(
        "Chunk %d/%d: %d stay_ids (total: %d)",
        chunk_idx,
        total_chunks,
        len(selected),
        len(stay_ids),
    )
    return selected



def extraction_args(description: str, *, weavehr: bool = False) -> argparse.Namespace:
    """Arguments of an extraction step; ``weavehr`` adds the project and dataset that Steps 02 and 03 read."""
    parser = argparse.ArgumentParser(description=description)
    add_data_arguments(parser, project=weavehr, dataset=weavehr)
    parser.add_argument("--features_config", default="configs/data/features.yaml")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    return parser.parse_args()


def load_chunk_cohort(
    paths_config: dict,
    chunk_idx: int,
    total_chunks: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return this chunk's sorted stay_ids and the cohort rows of those stays."""
    cohort = pd.read_csv(resolve_table_path(paths_config, "output.cohort") / "cohort.csv")
    cohort["icu_intime"] = pd.to_datetime(cohort["icu_intime"])
    cohort["icu_outtime"] = pd.to_datetime(cohort["icu_outtime"])
    stay_ids = chunk_stay_ids(np.sort(cohort["stay_id"].values), chunk_idx, total_chunks)
    return stay_ids, cohort[cohort["stay_id"].isin(set(stay_ids.tolist()))].copy()


def add_events(all_events: dict[int, list[pd.DataFrame]], events: pd.DataFrame) -> None:
    if len(events) > 0:
        for stay_id, group in events.groupby("stay_id"):
            all_events.setdefault(stay_id, []).append(group)


def write_stay_events(
    all_events: dict[int, list[pd.DataFrame]], stay_ids: np.ndarray, output_dir: Path,
) -> None:
    """Write one time-sorted parquet per stay that has events."""
    n_written = n_events = 0
    for stay_id in stay_ids:
        stay_id = int(stay_id)
        if stay_id not in all_events:
            continue
        stay_df = pd.concat(all_events[stay_id], ignore_index=True)
        stay_df = stay_df.sort_values("timestamp_hours").reset_index(drop=True)
        validate_schema(stay_df)
        stay_df.to_parquet(output_dir / f"{stay_id}.parquet", index=False, engine="pyarrow")
        n_written += 1
        n_events += len(stay_df)
    logger.info("wrote %d of %d stays, %d events, to %s", n_written, len(stay_ids), n_events, output_dir)
