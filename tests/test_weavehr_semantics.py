"""Clinical definitions stay native; the pipeline's summaries must preserve their meaning."""

import pickle

import numpy as np
import pandas as pd
import pytest
from test_weavehr_steps import make_project, write_meds

from clin_jepa.data.helpers.source import load_data_config
from clin_jepa.data.helpers.weavehr import Project
from clin_jepa.data.step01_cohort import build_cohort
from clin_jepa.downstream.labels import episode_duration, label_stay, mask_weavehr_labels
from clin_jepa.downstream.tasks import TASKS


def test_sofa_components_are_named_current_and_total_keeps_native_window(tmp_path):
    paths = make_project(tmp_path)
    features = load_data_config(paths, "features")
    names = {r["concept"]: r["display_name"] for r in features["observations"] if r["concept"].startswith("sofa")}
    assert names.pop("sofa") == "SOFA total"
    assert len(names) == 6 and all(name.endswith("(current)") for name in names.values())


def test_sepsis_is_unknown_only_where_weavehr_provides_no_sepsis_concept(tmp_path):
    paths = make_project(tmp_path)
    project = Project(paths)
    config = load_data_config(paths, "cohort")
    cohort = build_cohort(paths, config)
    assert cohort.sepsis3.isna().all()
    write_meds(project.concept_path("sepsis3"), [])
    cohort = build_cohort(paths, config)
    assert cohort.sepsis3.eq(0).all()


def test_supported_outcomes_are_enabled_only_with_coverage_and_full_horizon():
    frame = pd.DataFrame({"hour": np.arange(60), **{t.name: 0.0 for t in TASKS if t.hourly}})
    record = {"stay_id": 1, "prolonged_imv_96h": 0, "new_rrt_after_24h": 0}
    mask_weavehr_labels(frame, record, 60, {"sepsis_assessable": True, "ventilation_assessable": True},
                       {"prolonged_imv_96h"})
    assert frame.sepsis3_8h[:52].eq(0).all() and frame.sepsis3_8h[52:].isna().all()
    assert frame.imv_onset_24h[:36].eq(0).all() and frame.imv_onset_24h[36:].isna().all()
    assert frame.rrt_onset_48h.isna().all() and np.isnan(record["new_rrt_after_24h"])
    assert record["prolonged_imv_96h"] == 0


@pytest.mark.parametrize("los,stage_hours,expected", [(60, [1, 30], 0), (100, [1, 30], 0), (60, [30], np.nan)])
def test_observed_aki_and_exact_ventilation_duration(tmp_path, los, stage_hours, expected):
    assert episode_duration((np.array([0, 1, 3]), np.array([2, 4, 5])), 4.5) == 4.5
    assert episode_duration((np.array([np.nan, 1]), np.array([2, np.nan])), 4.5) == 0
    obs = pd.DataFrame({"step_idx": stage_hours, "category": "organ_failure", "source_table": "concept",
                        "variable_name": "aki", "numeric_value": 0, "text_value": None})
    path = tmp_path / "stay.pkl"
    with path.open("wb") as stream:
        pickle.dump({"stay_id": 1, "n_steps": 60, "obs_events": obs, "act_events": pd.DataFrame()}, stream)
    flags = {"sepsis_assessable": False, "ventilation_assessable": True, "los_icu": los / 24}
    _, record = label_stay((path, np.nan, None, False, {}, flags))
    assert (np.isnan(record["aki_kdigo23_after_24h"]) if np.isnan(expected)
            else record["aki_kdigo23_after_24h"] == expected)
    assert (record["prolonged_imv_96h"] == 0 if los == 60 else np.isnan(record["prolonged_imv_96h"]))
