"""Comparison with the original mimic-code pipeline: its names are aliased, its values never changed."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_mimic_demo", ROOT / "scripts/compare_mimic_demo.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def original_events():
    return pd.DataFrame({
        "stay_id": 1, "step_idx": [0, 0, 0, 0, 1, 1, 1],
        "source_table": ["vitalsign", "vitalsign", "rhythm", "ventilation", "inputevents", "coagulation", "bg"],
        "variable_name": ["heart_rate", "mbp", "heart_rhythm", "ventilation_status", "Propofol", "d_dimer", "calcium"],
        "category": ["vitals", "vitals", "vitals", "ventilation", "fluid_med_input", "labs", "blood_gas"],
        "numeric_value": [80.0, 60.0, np.nan, np.nan, 200.0, 500.0, 1.1],
        "text_value": [None, None, "SR", "InvasiveVent", "rate:20.0 mcg/kg/min | Continuous Med", None, None],
        "unit": ["bpm", "mmHg", None, "", "mg", "ng/mL", "mg/dL"],
    })


def test_every_alias_names_a_configured_concept_and_one_original_variable():
    naming = compare.load_naming()
    assert set(naming["concepts"]) <= set(naming["category"])
    pairs = [(alias["source_table"], alias["variable_name"]) for alias in naming["concepts"].values()]
    assert len(set(pairs)) == len(pairs)
    # The alias file is the only place that still names the original tables.
    features = yaml.safe_load((ROOT / "configs/data/features.yaml").read_text())
    assert not any({"source_table", "variable_name"} & set(rule)
                   for domain in ("observations", "actions") for rule in features[domain])


def test_original_events_are_renamed_without_changing_values():
    naming = compare.load_naming()
    original = original_events()
    renamed = compare.canonical_events(original, "original", naming)
    assert renamed.source_table.tolist() == ["concept", "concept", "rhythm", "concept", "inputevents", "concept",
                                             "concept"]
    assert renamed.variable_name.tolist() == ["heart_rate", "mean_arterial_pressure", "heart_rhythm", "ventilation",
                                              "Propofol", "d_dimer", "calcium_ionized"]
    assert renamed.category.tolist() == ["vital", "vital", "vitals", "ventilation", "fluid_med_input", "lab", "lab"]
    assert renamed.text_value[3] == "invasive" and renamed.unit.tolist()[-1] == "mmol/L"  # Annotations only.
    np.testing.assert_array_equal(renamed.numeric_value, original.numeric_value)
    adapted = compare.canonical_events(original, "adapted", naming)
    assert adapted.source_table.tolist() == original.source_table.tolist()


def test_text_agreement_counts_fragments_per_hour_within_the_shared_vocabulary():
    naming = compare.load_naming()
    original = compare.canonical_events(original_events(), "original", naming)
    adapted = original[original.source_table.isin(["concept", "inputevents"])].copy()
    adapted.loc[adapted.variable_name == "mean_arterial_pressure", "numeric_value"] = 61.0
    adapted = pd.concat([adapted, adapted[adapted.variable_name == "heart_rate"].assign(variable_name="phosphate")])
    result = {}
    for domain in ("observations", "actions"):
        fragments = [compare.text_fragments(events, domain) for events in (original, adapted)]
        result[domain] = compare.text_agreement(*fragments, set(naming["concepts"]), {"inputevents"})
    rows = pd.DataFrame(result["observations"]["by_variable"]).set_index("variable")
    assert rows.loc["mean_arterial_pressure", ["original", "adapted", "equal"]].tolist() == [1, 1, 0]
    assert rows.loc["heart_rate", "equal"] == rows.loc["d_dimer", "equal"] == rows.loc["ventilation", "equal"] == 1
    assert rows.loc["(all items)", ["original", "adapted", "equal"]].tolist() == [1, 1, 1]
    assert result["observations"]["hours_with_text"] == 2
    assert result["observations"]["hours_with_identical_fragments"] == 1
    # Heart rhythm exists only in the original vocabulary, phosphate only in the adapted one.
    assert result["observations"]["original_outside_shared_vocabulary"] == 1
    assert result["observations"]["adapted_outside_shared_vocabulary"] == 1
    assert result["actions"]["fragments"] == result["observations"]["fragments"]
