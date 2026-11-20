"""Step 01: select the ICU-stay cohort and write ``cohort/cohort.csv``."""

import argparse
import logging

import pandas as pd

from clin_jepa.data.helpers.io_utils import resolve_table_path
from clin_jepa.data.helpers.source import add_data_arguments, data_paths, load_data_config
from clin_jepa.data.helpers.weavehr import Project, icustay_detail, sepsis3, stays_with_values
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "stay_id",
    "subject_id",
    "hadm_id",
    "gender",
    "admission_age",
    "race",
    "icu_intime",
    "icu_outtime",
    "los_icu",
    "icu_los_hours",
    "hospital_expire_flag",
    "sepsis3",
    "sepsis_sofa_time",
]


def _log_filter(name: str, before: int, after: int) -> None:
    logger.info("filter [%s]: %d -> %d", name, before, after)


def _stays_with_vitals(project: Project, stays: pd.DataFrame) -> set:
    features = load_data_config(project.paths, "features")
    vitals = [rule for rule in features["observations"] if rule["category"] == "vital"]
    return stays_with_values(project, stays, vitals)


def build_cohort(paths_config: dict, cohort_config: dict) -> pd.DataFrame:
    """Apply the inclusion criteria to the ICU stays of the WeavEHR project and join the Sepsis-3 flag."""
    project = Project(paths_config)
    df = icustay_detail(project)
    n_initial = len(df)

    n = len(df)
    inclusion = cohort_config["inclusion"]
    min_age = inclusion["min_age"]
    df = df[df["admission_age"] >= min_age]
    _log_filter(f"age >= {min_age}", n, len(df))

    n = len(df)
    df = df.dropna(subset=["icu_intime", "icu_outtime"])
    _log_filter("valid ICU times", n, len(df))

    n = len(df)
    df = df[df["first_icu_stay"]]
    _log_filter("first ICU stay of the admission", n, len(df))

    n = len(df)
    min_los_hours = inclusion["min_icu_los_hours"]
    df = df[(df["los_icu"] * 24) >= min_los_hours]
    _log_filter(f"ICU LOS >= {min_los_hours} h", n, len(df))

    n = len(df)
    df = df[df["stay_id"].isin(_stays_with_vitals(project, df))]
    _log_filter("at least one vital sign", n, len(df))

    df["icu_los_hours"] = (df["los_icu"] * 24).clip(upper=cohort_config["max_icu_los_hours"])

    sepsis = sepsis3(project, df, cohort_config["sepsis"]["prevalent_lookback_hours"])
    df = df.merge(sepsis, on="stay_id", how="left")

    df = df[OUTPUT_COLUMNS].sort_values("stay_id").reset_index(drop=True)
    logger.info("cohort: %d of %d ICU stays", len(df), n_initial)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--cohort_config", default="configs/data/cohort.yaml")
    args = parser.parse_args()

    setup_logging()
    paths_config = data_paths(args)
    cohort = build_cohort(paths_config, load_config(args.cohort_config))

    output_dir = ensure_dir(resolve_table_path(paths_config, "output.cohort"))
    cohort.to_csv(output_dir / "cohort.csv", index=False)
    logger.info("wrote %s", output_dir / "cohort.csv")


if __name__ == "__main__":
    main()
