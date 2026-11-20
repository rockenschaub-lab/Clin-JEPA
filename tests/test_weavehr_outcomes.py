"""Do not turn absent follow-up or absent mappings into negative outcomes."""


import numpy as np
import pandas as pd
from test_weavehr_steps import make_project, write_meds

from clin_jepa.data.helpers.source import load_data_config
from clin_jepa.data.helpers.weavehr import Project
from clin_jepa.data.step01_cohort import build_cohort
from clin_jepa.downstream import organ_failure
from clin_jepa.downstream.labels import mask_weavehr_labels, onset_label, sepsis_onset_hour, stay_outcomes
from clin_jepa.downstream.tasks import TASKS


def test_follow_up_stays_unknown_where_the_dataset_declares_none(tmp_path):
    # AmsterdamUMCdb declares no follow-up outcomes; known values pass through and nothing becomes a negative.
    paths = make_project(tmp_path, "aumc")
    cohort = build_cohort(paths, load_data_config(paths, "cohort")).iloc[:2].copy()
    cohort["hospital_expire_flag"] = [0, np.nan]
    cohort["sepsis3"] = [1, np.nan]
    cohort["sepsis_sofa_time"] = cohort.icu_intime + pd.Timedelta(hours=30)
    out = stay_outcomes(cohort, Project(paths))
    assert out.hospital_mortality.iloc[0] == 0 and np.isnan(out.hospital_mortality.iloc[1])
    assert out.mortality_7d.isna().all() and out.death_in_icu.isna().all() and out.death_hour.isna().all()
    assert out.sepsis_ever.iloc[0] == 1 and np.isnan(out.sepsis_ever.iloc[1])
    assert out.sepsis_onset_after_24h.iloc[0] == 1
    assert out.hospital_readmission_30d.isna().all() and out.icu_readmission_same_hosp.isna().all()


def test_unvalidated_tasks_and_unobserved_horizons_are_masked():
    frame = pd.DataFrame({"hour": np.arange(60), **{t.name: 0.0 for t in TASKS if t.hourly}})
    record = {"stay_id": 1, "new_rrt_after_24h": 0, "prolonged_imv_96h": 0}
    mask_weavehr_labels(frame, record, 60)
    assert frame.rrt_onset_48h.isna().all() and frame.imv_onset_24h.isna().all()
    assert frame.sofa_worsen_2_48h.iloc[:12].eq(0).all()
    assert frame.sofa_worsen_2_48h.iloc[12:].isna().all()
    assert record["stay_id"] == 1 and np.isnan(record["new_rrt_after_24h"])


def test_declared_follow_up_outcomes_are_read_from_the_weavehr_project(tmp_path):
    paths = make_project(tmp_path)
    project = Project(paths)
    stay = pd.Timestamp("2013-01-01")
    # Patient 1 dies 5 h into stay 1; patient 2 has a death known by date only, two weeks later.
    write_meds(project.extracted_path("patients", "MEDS_DEATH"), [
        {"subject_id": 1, "code": "MEDS_DEATH", "time": stay + pd.Timedelta(days=1, hours=5)},
        {"subject_id": 2, "code": "MEDS_DEATH", "time": stay + pd.Timedelta(days=17, hours=23, minutes=59, seconds=59)},
    ])
    write_meds(project.extracted_path("admissions", "HOSPITAL_ADMISSION"), [
        {"subject_id": (n + 1) // 2, "hadm_id": 100 + n, "code": "HOSPITAL_ADMISSION",
         "time": stay + pd.Timedelta(days=n)}
        for n in range(1, 5)
    ])
    write_meds(project.extracted_path("admissions", "HOSPITAL_DISCHARGE"), [
        {"subject_id": (n + 1) // 2, "hadm_id": 100 + n, "code": f"HOSPITAL_DISCHARGE//{location}",
         "time": stay + pd.Timedelta(days=n, hours=8)}
        for n, location in zip(range(1, 5), ["DIED", "HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY"])
    ])
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    out = stay_outcomes(cohort, project)
    assert out.death_in_icu.tolist() == [1, 0, 0, 0] and out.death_hour.tolist() == [5] + [np.inf] * 3
    # A date-only death dates mortality but is never placed inside a stay.
    assert out.mortality_7d.tolist() == [1, 0, 0, 0] and out.mortality_30d.tolist() == [1, 0, 1, 1]
    assert out.discharge_home.tolist() == [0, 1, 1, 0]
    # Each patient's second admission follows the first within a day.
    assert out.hospital_readmission_30d.tolist() == [1, 0, 1, 0]
    assert out.icu_readmission_same_hosp.tolist() == [0, 0, 0, 0]


def test_sepsis_onset_hour_is_masked_not_labelled_positive():
    assert sepsis_onset_hour(np.nan, 3.0) is None and sepsis_onset_hour(0, np.nan) == np.inf
    onset = sepsis_onset_hour(1, 5.5)
    assert onset == 5
    label = onset_label(onset, 10, 8)
    # Hour 5 holds the events of 5-6 h, which already include the onset.
    assert label[:5].tolist() == [1] * 5 and np.isnan(label[5:]).all()
    # Sepsis present before admission never yields an in-stay onset.
    assert np.isnan(onset_label(sepsis_onset_hour(1, -10.5), 10, 8)).all()


def test_kidney_failure_reads_the_renal_sofa_concept():
    def events(variable):
        return pd.DataFrame({
            "step_idx": [0, 0], "category": ["lab", "score"], "source_table": "concept",
            "variable_name": ["creatinine", variable], "numeric_value": [1.0, 3.0], "text_value": None,
        })

    state = organ_failure.states(events("sofa_renal"), pd.DataFrame(columns=events("sofa_renal").columns), 4)
    assert state["kidney_failure_48h"][0] == 1
    assert np.isnan(organ_failure.states(events("other"), pd.DataFrame(columns=events("other").columns), 4)[
        "kidney_failure_48h"]).all()
