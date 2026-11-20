"""Upstream's text templates, with the names and order of the WeavEHR concepts from features.yaml."""

import pandas as pd

from clin_jepa.data.helpers.event_ordering import CANONICAL_VARIABLE_ORDER
from clin_jepa.data.helpers.source import load_data_config
from clin_jepa.data.helpers.text_templates import DISPLAY_NAMES, format_event
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.data.step06_trajectories import hour_text


def test_every_concept_has_a_display_name_and_a_position():
    features = load_data_config({"dataset": "mimic-iv"}, "features")
    names = [rule.get("name", rule["concept"]) for rule in features["observations"] + features["actions"]]
    assert len(names) == len(set(names)) == 130
    assert set(DISPLAY_NAMES) == set(CANONICAL_VARIABLE_ORDER) == {("concept", name) for name in names}
    assert sorted(CANONICAL_VARIABLE_ORDER.values()) == list(range(130))
    # Two concepts form the ventilation intervals.
    assert DISPLAY_NAMES["concept", "ventilation"] == "Ventilation status"


def test_concept_units_are_the_configured_ones():
    assert get_unit("concept", "heart_rate") == "bpm"
    # output_unit is written in place of the unit in WeavEHR's code.
    assert get_unit("concept", "sofa") == "" and get_unit("concept", "vasopressin_rate") == "units/hour"
    assert get_unit("concept", "new_concept") == ""


def test_concepts_are_named_by_the_configuration():
    assert format_event("concept", "mean_arterial_pressure", 56.0, None, "mmHg") == "MAP: 56 mmHg"
    assert format_event("concept", "gcs", 14.0, None, "") == "GCS total: 14"
    assert format_event("concept", "ventilation", None, "invasive", "") == "Ventilation status: invasive"
    assert format_event("concept", "antibiotics", 1.0, "1g Frozen Bag", "") == "Antibiotics: 1 (1g Frozen Bag)"
    assert format_event("concept", "lactate", 2.456, None, "mmol/L") == "Lactate: 2.46 mmol/L"
    # A concept that the configuration does not name is still written.
    assert format_event("concept", "new_concept", 1.0, None, "") == "new concept: 1"
    assert format_event("concept", "heart_rate", None, None, "bpm") is None


def test_concepts_follow_the_configured_order_then_the_source_records():
    def event(variable, value=None, text=None, unit="", table="concept"):
        return {"source_table": table, "variable_name": variable, "numeric_value": value, "text_value": text,
                "unit": unit}

    events = pd.DataFrame([
        event("Propofol", 200.0, "rate:20.0 mcg/kg/min | Continuous Med", "mg", "inputevents"),
        event("lactate", 2.0, unit="mmol/L"),
        event("heart_rate", 88.0, unit="bpm"),
        event("Aspirin", text="Administered", table="emar"),
        event("sofa", 5.0),
        event("heart_rate", 90.0, unit="bpm"),
    ])
    assert hour_text(events, 3, "none") == (
        "t=3h | SOFA total: 5. Heart rate: 88 bpm. Heart rate: 90 bpm. Lactate: 2 mmol/L. "
        "Propofol: 200 mg, 20 mcg/kg/min Continuous Med. Aspirin: Administered"
    )
