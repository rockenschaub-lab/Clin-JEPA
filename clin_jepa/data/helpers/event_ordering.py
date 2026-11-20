"""Fixed within-hour ordering of clinical variables for the state and action texts."""

import pandas as pd

from clin_jepa.data.helpers.schema import CONCEPT_SOURCE
from clin_jepa.data.helpers.source import CONFIG_ROOT
from clin_jepa.utils import load_config


def _concept_order() -> dict[tuple[str, str], int]:
    """The order of the WeavEHR concepts in configs/data/features.yaml."""
    features = load_config(CONFIG_ROOT / "features.yaml")
    rules = features["observations"] + features["actions"]
    return {(CONCEPT_SOURCE, rule.get("name", rule["concept"])): i for i, rule in enumerate(rules)}


CANONICAL_VARIABLE_ORDER: dict[tuple[str, str], int] = _concept_order()


def _sort_key(source_table: str, variable_name: str) -> int:
    """Get canonical sort priority for a variable. Unmapped variables get 9999."""
    return CANONICAL_VARIABLE_ORDER.get((source_table, variable_name), 9999)


def sort_events(events: pd.DataFrame) -> pd.DataFrame:
    """Events in the fixed variable order; ties keep their input order."""
    if len(events) == 0:
        return events
    priorities = events.apply(
        lambda row: _sort_key(row["source_table"], row["variable_name"]),
        axis=1,
    )
    return events.assign(_priority=priorities).sort_values(
        "_priority", kind="stable",
    ).drop(columns=["_priority"])
