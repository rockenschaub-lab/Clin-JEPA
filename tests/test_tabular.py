"""Hourly tabular view: concept columns by name, medication classes from the source records."""

import numpy as np
import pandas as pd

from clin_jepa.downstream.tabular import OBSERVATIONS, Layout, MedicationClasses, census_chunk, stay_matrix

STATE_TEXT = {"exclude_categories": [], "exclude_variables": []}


def action(step, variable, value=np.nan, text=None, table="concept", category="vasopressor"):
    return {"step_idx": step, "source_table": table, "category": category, "variable_name": variable,
            "numeric_value": value, "text_value": text, "unit": ""}


def stay():
    obs = pd.DataFrame({
        "step_idx": [0, 0, 1], "source_table": "concept", "category": "vital",
        "variable_name": ["heart_rate", "heart_rate", "mean_arterial_pressure"],
        "numeric_value": [80.0, 90.0, 60.0], "text_value": None, "unit": "",
    })
    act = pd.DataFrame([
        action(0, "norepinephrine_rate", 0.1),
        action(0, "ventilation", text="invasive", category="ventilation"),
        action(1, "positive_end_expiratory_pressure", 8.0, category="ventilation_settings"),
        action(1, "crrt_blood_flow", 150.0, category="dialysis"),
        action(1, "antibiotics", 1.0, category="medication_admin"),
        action(0, "Propofol", 200.0, "rate:20.0 mcg/kg/min | Continuous Med", "inputevents", "fluid_med_input"),
        action(1, "Vancomycin", 1000.0, "route:IV | 1g Frozen Bag", "prescriptions", "medication_order"),
        action(2, "Vancomycin", 1000.0, "route:IV | 1g Frozen Bag", "prescriptions", "medication_order"),
        action(1, "Dialysis Catheter", 1.0, table="procedureevents", category="procedure"),
    ])
    return {"stay_id": 1, "n_steps": 3, "obs_events": obs, "act_events": act}


def test_columns_are_named_after_concepts_and_classes_come_from_source_records(tmp_path):
    path = tmp_path / "1.pkl"
    pd.to_pickle(stay(), path)
    events, stays = census_chunk([path])
    # The census sees what the action text states: no excluded categories. Items a
    # dataset excludes never reach a stay; they are left out when its records are read.
    assert ("fluid_med_input", "inputevents", "propofol") in events
    assert not any(category == "procedure" for category, _, name in events)
    assert events["medication_order", "prescriptions", "vancomycin"] == stays[
        "medication_order", "prescriptions", "vancomycin"] == 1
    items = [{"source_table": "inputevents", "name": "propofol", "column": "item_inputevents__propofol"}]
    layout = Layout(MedicationClasses("configs/medication_classes.yaml"), {"items": items})
    assert layout.columns[: len(OBSERVATIONS)] == OBSERVATIONS and len(set(layout.columns)) == len(layout.columns)
    x = pd.DataFrame(stay_matrix(stay(), layout, STATE_TEXT), columns=layout.columns)
    assert x.heart_rate.tolist()[:1] == [85.0] and x.mean_arterial_pressure[1] == 60.0 and np.isnan(x.lactate).all()
    assert x.norepinephrine_rate[0] == np.float32(0.1) and np.isnan(x.norepinephrine_rate[1])
    assert x.ventilation_invasive.tolist() == [1, 0, 0] and x.ventilation_noninvasive.tolist() == [0, 0, 0]
    assert x.positive_end_expiratory_pressure[1] == 8.0 and x.crrt_blood_flow[1] == 150.0
    assert x.crrt_any.tolist() == [0, 1, 0] and x.antibiotic_active.tolist() == [0, 1, 0]
    assert x.class_sedative.tolist() == [1, 0, 0]
    assert x.class_sedative_amount[0] == 200 and x.class_sedative_rate[0] == 20
    assert x.order_placed_antibiotic.tolist() == [0, 1, 0]  # An order is stated in its first hour only.
    assert x.item_inputevents__propofol.tolist() == [1, 0, 0]
    assert x.n_obs_statements.tolist() == [2, 1, 0] and x.no_active_interventions.tolist() == [0, 0, 1]
