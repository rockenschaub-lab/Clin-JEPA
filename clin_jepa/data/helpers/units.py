"""Units of the extracted variables, keyed by (source_table, variable_name)."""

from clin_jepa.data.helpers.schema import CONCEPT_SOURCE
from clin_jepa.data.helpers.source import CONFIG_ROOT
from clin_jepa.utils import load_config


def _units() -> dict[tuple[str, str], str]:
    """Units of the WeavEHR concepts, as configs/data/features.yaml lists them."""
    features = load_config(CONFIG_ROOT / "features.yaml")
    rules = features["observations"] + features["actions"]
    return {
        (CONCEPT_SOURCE, rule.get("name", rule["concept"])): rule.get("output_unit", rule["unit"]) for rule in rules
    }


VARIABLE_UNITS: dict[tuple[str, str], str] = _units()


def get_unit(source_table: str, variable_name: str) -> str:
    """Look up the unit for a given (source_table, variable_name) pair."""
    return VARIABLE_UNITS.get((source_table, variable_name), "")
