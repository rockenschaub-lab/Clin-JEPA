"""Steps 01 to 03 on synthetic WeavEHR projects: cohort, stay alignment, intervals and source records."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from clin_jepa.data import step02_observations, step03_actions
from clin_jepa.data.helpers.action_filter import filter_act_events
from clin_jepa.data.helpers.source import load_data_config
from clin_jepa.data.helpers.temporal_utils import build_icu_windows, build_intime_map, build_outtime_map
from clin_jepa.data.helpers.weavehr import Project, assign_stay_ids, rate_intervals
from clin_jepa.data.step01_cohort import build_cohort
from clin_jepa.data.step03_actions import source_records
from clin_jepa.data.step04_discretization import build_stay_timeline, discretize_stay
from clin_jepa.data.step06_trajectories import hourly_texts


def write_events(paths, domain, features=None):
    """Run Step 02 (observations) or Step 03 (actions) as its command does."""
    module = step02_observations if domain == "observation" else step03_actions
    argv = [module.__name__, "--data_dir", paths["data_dir"], "--dataset", paths["dataset"],
            "--weavehr_project", paths["weavehr"]["project"]]
    if features is not None:
        path = Path(paths["data_dir"]) / "features.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(features))
        argv += ["--features_config", str(path)]
    with patch.object(sys, "argv", argv):
        module.main()


def all_actions(paths):
    return pd.concat([pd.read_parquet(p) for p in sorted(Path(paths["output"]["actions"]).glob("*.parquet"))])


def write_meds(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "time": pd.Timestamp("2013-01-02"),
                "code": "",
                "numeric_value": np.nan,
                "text_value": None,
                **r,
            }
            for r in rows
        ]
    )
    if not len(frame):
        frame = pd.DataFrame(columns=["subject_id", "time", "code", "numeric_value", "text_value"])
    frame["time"] = pd.to_datetime(frame.time)
    frame.to_parquet(path, index=False)


def make_project(root, dataset="mimic-iv", n=4):
    root.mkdir(parents=True, exist_ok=True)
    name, version = ("aumc", "1.5.0") if dataset == "aumc" else ("mimic-iv", "3.1")
    # What the command-line arguments of a step amount to.
    paths = {
        "dataset": dataset,
        "weavehr": {"project": str(root / "weavehr")},
        "output": {
            k: str(root / "processed" / k)
            for k in ["cohort", "observations", "actions", "discretized", "splits", "trajectories"]
        },
    }
    paths["data_dir"] = str(root / "processed")
    project = Path(paths["weavehr"]["project"])
    # What WeavEHR itself leaves in a project: a completion marker and its metadata per step.
    for step in ["extraction", "concept"]:
        (project / "datasets" / step / "metadata").mkdir(parents=True)
        (project / "datasets" / step / ".complete").touch()
        (project / "datasets" / step / "metadata" / "dataset.json").write_text(
            json.dumps({"etl_name": "WeavEHR", "etl_version": "0.0.1", "created_at": "2026-01-01T00:00:00"})
        )
    (project / "datasets/extraction/data" / name / version).mkdir(parents=True)
    inputs = {}
    concepts = {}

    def event(table, event, **row):
        inputs.setdefault((table, event), []).append(row)

    for subject in range(1, n // 2 + 1):
        concepts.setdefault("patient_sex", []).append(
            {"subject_id": subject, "time": "1968-01-01", "code": "patient_sex", "text_value": "Male"}
        )
    for stay in range(1, n + 1):
        subject = (stay + 1) // 2
        t = pd.Timestamp("2013-01-01") + pd.Timedelta(days=stay)
        if dataset == "mimic-iv":
            for ev, time in [("icu_admission", t), ("icu_discharge", t + pd.Timedelta(hours=8))]:
                event("icustay", ev, subject_id=subject, stay_id=stay, hadm_id=100 + stay, time=time, code=ev)
            event("admissions", "IN_HOSPITAL_MORTALITY", subject_id=subject, hadm_id=100 + stay, time=t,
                  code="IN_HOSPITAL_MORTALITY", numeric_value=int(stay % 4 == 0))
            # WeavEHR has a race for the first patient only.
            if subject == 1:
                concepts.setdefault("patient_race", []).append(
                    {"subject_id": subject, "time": t, "code": "patient_race//white", "text_value": "WHITE"}
                )
        else:
            for ev, time in [("VISIT_START", t), ("VISIT_END", t + pd.Timedelta(hours=8))]:
                event(
                    "visit_occurrence",
                    ev,
                    subject_id=subject,
                    visit_occurrence_id=stay,
                    care_site_id="1",
                    time=time,
                    code=ev + "//32037//ICU",
                )
        # One age per hospital admission (MIMIC-IV) or ICU visit (AUMC), at its start.
        concepts.setdefault("patient_age", []).append(
            {"subject_id": subject, "time": t, "code": "patient_age//years", "numeric_value": 45 + stay}
        )
        for h in range(8):
            concepts.setdefault("heart_rate", []).append(
                {
                    "subject_id": subject,
                    "stay_id": str(stay),
                    "time": t + pd.Timedelta(hours=h, minutes=10),
                    "code": "heart_rate//bpm",
                    "numeric_value": 70 + h,
                }
            )
        for h, value in [(1, 0.1), (4, 0)]:
            concepts.setdefault("norepinephrine_rate", []).append(
                {
                    "subject_id": subject,
                    "stay_id": str(stay),
                    "time": t + pd.Timedelta(hours=h),
                    "code": "norepinephrine_rate//mcg/kg/min",
                    "numeric_value": value,
                    ("linkorderid" if dataset == "mimic-iv" else "drug_exposure_id"): str(stay),
                }
            )
    for (table, ev), rows in inputs.items():
        write_meds(project / "datasets/extraction/data" / name / version / table / (ev + ".parquet"), rows)
    for concept, rows in concepts.items():
        write_meds(project / "datasets/concept/data" / concept / "1.0.0" / name / (version + ".parquet"), rows)
    return paths


@pytest.mark.parametrize("dataset", ["mimic-iv", "aumc"])
def test_cohort_and_intervals_preserve_dataset_contract(tmp_path, dataset):
    paths = make_project(tmp_path, dataset)
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    # WeavEHR's age, sex and race concepts. Each stay takes the age of its own admission,
    # not that of the patient's other stay; a patient without a race row has none.
    assert cohort.admission_age.tolist() == [46, 47, 48, 49]
    assert cohort.gender.tolist() == ["M"] * 4
    assert cohort.race.tolist() == (["WHITE", "WHITE"] if dataset == "mimic-iv" else ["UNKNOWN"] * 2) + ["UNKNOWN"] * 2
    assert cohort.subject_id.tolist() == [1, 1, 2, 2]
    assert cohort.sepsis3.isna().all()
    if dataset == "aumc":
        assert cohort.hadm_id.isna().all() and cohort.hospital_expire_flag.isna().all()
    output = Path(paths["output"]["cohort"])
    output.mkdir(parents=True)
    cohort.to_csv(output / "cohort.csv", index=False)
    write_events(paths, "observation")
    write_events(paths, "action")
    obs = pd.read_parquet(Path(paths["output"]["observations"]) / "1.parquet")
    act = pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet")
    assert obs.numeric_value.tolist() == list(range(70, 78))
    # One interval from the rate to its stop; the stop is no event of its own.
    assert act.timestamp_hours.tolist() == [1] and act.end_timestamp_hours.tolist() == [4]
    assert act.numeric_value.tolist() == [0.1]
    # A concept WeavEHR does not produce for this project has no events.
    assert not all_actions(paths).variable_name.eq("phenylephrine_rate").any()


def test_a_row_keeps_its_own_stay_and_one_without_is_matched_by_time(tmp_path):
    paths = make_project(tmp_path)
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))  # Stay 1: 2013-01-02 00:00 to 08:00.
    events = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "stay_id": ["2", "1", None],
            "time": pd.to_datetime(["2013-01-02 01:00", "2013-01-02 08:00", "2013-01-02 01:00"]),
            "value": [1, 2, 3],
        }
    )
    result = assign_stay_ids(events, cohort)
    # Stay 2's row lies in stay 1 and is not reassigned; stay 1's own row lies at its discharge.
    assert result.stay_id.tolist() == [1] and result.value.tolist() == [3]
    assert result.stay_id.dtype == "int64"


def test_each_rate_holds_until_the_next_update_of_its_record():
    rows = rate_rows([(0, 0.1, "a"), (1, 0.2, "b"), (2, 0, "a"), (2, 0.3, "a"), (3, 0, "b"), (4, 0, "a")])
    actual = rate_intervals(rows)
    # Overlapping records stay separate; a stop ends its record and is no event itself.
    spans = sorted(zip(hours(actual.time), hours(actual.end_timestamp), actual.numeric_value))
    assert spans == [(0, 2, 0.1), (1, 3, 0.2), (2, 4, 0.3)]
    for seed in (1, 8, 20):
        pd.testing.assert_frame_equal(actual, rate_intervals(rows.sample(frac=1, random_state=seed)))


def test_a_rate_without_a_later_update_is_a_point():
    actual = rate_intervals(rate_rows([(0, 0.1, "a"), (1, 0.2, "a"), (0, 0.3, "b"), (4, 0, "b")]))
    spans = sorted(zip(hours(actual.time), actual.numeric_value, actual.end_timestamp.isna()))
    assert spans == [(0, 0.1, False), (0, 0.3, False), (1, 0.2, True)]
    # Of simultaneous updates of one record, the highest rate wins.
    actual = rate_intervals(rate_rows([(0, 0.1, "a"), (0, 0.2, "a"), (2, 0, "a")]))
    assert actual.numeric_value.tolist() == [0.2] and hours(actual.end_timestamp) == [2]
    # Updates without a record cannot be paired and are left out.
    assert rate_intervals(rate_rows([(0, 0.1, None), (1, 0, None)])).empty


def test_malformed_concept_units_fail_explicitly(tmp_path):
    paths = make_project(tmp_path)
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    out = Path(paths["output"]["cohort"])
    out.mkdir(parents=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    path = Project(paths).concept_path("heart_rate")
    frame = pd.read_parquet(path)
    frame["code"] = "heart_rate//wrong_unit"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="codes/units"):
        write_events(paths, "observation")


def test_unfinished_project_fails_before_cohort(tmp_path):
    paths = make_project(tmp_path)
    (Path(paths["weavehr"]["project"]) / "datasets/concept/.complete").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        build_cohort(paths, load_data_config(paths, "cohort"))


def rate_rows(updates):
    t = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        [{"time": t + pd.Timedelta(hours=h), "numeric_value": v, "_record": p, **extra} for h, v, p, *rest in updates
         for extra in [rest[0] if rest else {}]]
    )


def hours(times):
    return [(time - pd.Timestamp("2020-01-01")) / pd.Timedelta(hours=1) for time in times]


def test_invalid_ventilation_does_not_abort(tmp_path):
    paths = make_project(tmp_path)
    project = Project(paths)
    for name, time in [("ventilation_start", "2013-01-02 03:00"), ("ventilation_end", "2013-01-02 03:00")]:
        write_meds(project.concept_path(name), [{"stay_id": "1", "orderid": "10", "code": name + "//invasive",
                                                 "time": time}])
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    out = Path(paths["output"]["cohort"])
    out.mkdir(parents=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    write_events(paths, "action")
    # An end at its own start is no interval, and the other events are still written.
    actions = all_actions(paths)
    assert not actions.variable_name.eq("ventilation").any() and actions.variable_name.eq("norepinephrine_rate").any()


def test_cohort_keeps_the_original_selection_rules(tmp_path):
    paths = make_project(tmp_path, n=6)
    project = Project(paths)
    path = project.extracted_path("icustay", "icu_admission")
    start = pd.read_parquet(path)
    end_path = project.extracted_path("icustay", "icu_discharge")
    end = pd.read_parquet(end_path)
    for frame in (start, end):
        frame.loc[frame.stay_id == 2, "hadm_id"] = 101  # A second ICU stay of stay 1's admission.
    end.loc[end.stay_id == 3, "time"] += pd.Timedelta(hours=400)  # Longer than the cap.
    end.loc[end.stay_id == 5, "time"] = start.loc[start.stay_id == 5, "time"].iloc[0]  # Malformed interval.
    start.to_parquet(path, index=False)
    end.to_parquet(end_path, index=False)
    path = project.concept_path("heart_rate")
    vitals = pd.read_parquet(path)
    vitals[vitals.stay_id != "4"].to_parquet(path, index=False)  # Stay 4 has no vital sign.
    config = load_data_config(paths, "cohort")
    cohort = build_cohort(paths, config).set_index("stay_id")
    assert cohort.index.tolist() == [1, 3, 6]
    assert cohort.loc[3, "icu_los_hours"] == 336 and cohort.loc[3, "los_icu"] == pytest.approx(17)


def test_sepsis_before_icu_admission_is_present_on_admission(tmp_path):
    paths = make_project(tmp_path)
    project = Project(paths)
    # WeavEHR dates sepsis per subject, without a stay: 10 h before stay 1, in
    # stay 3, and too long before stay 4 to be attributed to it.
    write_meds(project.concept_path("sepsis3"), [
        {"subject_id": 1, "code": "sepsis3", "time": "2013-01-01 14:00", "numeric_value": 1},
        {"subject_id": 2, "code": "sepsis3", "time": "2013-01-04 05:00", "numeric_value": 1},
        {"subject_id": 2, "code": "sepsis3", "time": "2013-01-04 03:00", "numeric_value": 1},
    ])
    cohort = build_cohort(paths, load_data_config(paths, "cohort")).set_index("stay_id")
    assert cohort.sepsis3.eq(1).tolist() == [True, False, True, False]
    assert cohort.loc[1, "sepsis_sofa_time"] == pd.Timestamp("2013-01-01 14:00")
    assert cohort.loc[3, "sepsis_sofa_time"] == pd.Timestamp("2013-01-04 03:00")
    config = load_data_config(paths, "cohort")
    config["sepsis"]["prevalent_lookback_hours"] = 0
    assert build_cohort(paths, config).sepsis3.eq(1).tolist() == [False, False, True, False]


def test_amsterdam_cohort_excludes_medium_care(tmp_path):
    paths = make_project(tmp_path, "aumc")
    project = Project(paths)
    path = project.extracted_path("visit_occurrence", "VISIT_START")
    frame = pd.read_parquet(path)
    frame.loc[frame.visit_occurrence_id == 2, "care_site_id"] = "2"
    frame.to_parquet(path, index=False)
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    assert cohort.stay_id.tolist() == [1, 3, 4]


def test_stay_without_detected_sepsis_is_negative_where_weavehr_provides_the_concept(tmp_path):
    paths = make_project(tmp_path)
    project = Project(paths)
    write_meds(
        project.concept_path("sepsis3"),
        [{"stay_id": "1", "code": "sepsis3", "time": "2013-01-02 03:00", "numeric_value": 1}],
    )
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    assert cohort.sepsis3.iloc[0] == 1 and cohort.sepsis3.iloc[1:].eq(0).all()
    assert cohort.sepsis_sofa_time.iloc[0] == pd.Timestamp("2013-01-02 03:00")


def test_paired_ventilation_is_an_interval_that_starts_in_the_stay(tmp_path):
    paths = make_project(tmp_path)  # Stay 1: 2013-01-02 00:00 to 08:00.
    project = Project(paths)
    write_meds(project.concept_path("ventilation_start"), [
        {"stay_id": "1", "orderid": "10", "code": "ventilation_start//invasive", "time": "2013-01-02 01:00"},
        # Started before ICU admission: as upstream, the interval is left out of the actions.
        {"stay_id": "1", "orderid": "11", "code": "ventilation_start//noninvasive", "time": "2013-01-01 23:00"},
    ])
    write_meds(project.concept_path("ventilation_end"), [
        {"stay_id": "1", "orderid": "10", "code": "ventilation_end//invasive", "time": "2013-01-02 09:30"},
        {"stay_id": "1", "orderid": "11", "code": "ventilation_end//noninvasive", "time": "2013-01-02 03:30"},
    ])
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    out = Path(paths["output"]["cohort"])
    out.mkdir(parents=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    write_events(paths, "action")
    actions = pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet")
    vent = actions[actions.variable_name == "ventilation"]
    # It ends with the stay.
    assert vent.timestamp_hours.tolist() == [1.0] and vent.end_timestamp_hours.tolist() == [8.0]
    assert vent.text_value.tolist() == ["invasive"] and vent.source_table.tolist() == ["concept"]
    # Do not invent an interval if its native end marker is missing.
    p = project.concept_path("ventilation_end")
    frame = pd.read_parquet(p)
    frame.iloc[:0].to_parquet(p, index=False)
    write_events(paths, "action")
    actions = pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet")
    assert not actions.variable_name.eq("ventilation").any()


def source_record_project(tmp_path):
    """Stay 1 (subject 1, admission 101) lasts from 2013-01-02 00:00 to 08:00."""
    paths = make_project(tmp_path)
    root = Path(paths["weavehr"]["project"]) / "datasets/extraction/data/mimic-iv/3.1"
    t = pd.Timestamp("2013-01-02")

    def hour(h):
        return t + pd.Timedelta(hours=h)

    def infusion(orderid, item, begin, end, *, status="FinishedRunning", planned, delivered, amount, known=None,
                 rateuom="mcg/kg/min", amountuom="mg", category="Continuous Med"):
        ids = {"stay_id": 1, "hadm_id": 101, "orderid": orderid, "rateuom": rateuom, "amountuom": amountuom}
        started = {**ids, "time": hour(begin), "code": f"INFUSION_START//{item}", "available_time": hour(begin),
                   "originalrate": np.float32(planned), "ordercategorydescription": category}
        ended = {**ids, "time": hour(end), "code": f"INFUSION_END//{item}", "text_value": status,
                 "available_time": hour(end if known is None else known),
                 "rate": np.float32(delivered), "amount": np.float32(amount)}
        return started, ended

    records = [
        # Set to 20, delivered at 20.1; the amount is charted two hours after the end.
        infusion(1, "222168//Propofol", 1, 3, planned=20, delivered=20.1, amount=200, known=5),
        # A rewritten record is not an administration.
        infusion(2, "222168//Propofol", 3, 4, status="Rewritten", planned=30, delivered=30, amount=100),
        # Started before ICU admission: it belongs to no stay.
        infusion(3, "225158//NaCl 0.9%", -2, 1, planned=500, delivered=np.nan, amount=500, rateuom=None,
                 amountuom="mL", category="Bolus"),
        # A push has no rate; the source repeats its amount in the rate column.
        infusion(5, "221744//Fentanyl", 2, 2 + 1 / 60, planned=50, delivered=np.nan, amount=50, rateuom=None,
                 amountuom="mcg", category="Drug Push"),
        # Charted only after ICU discharge: its amount never reaches this stay.
        infusion(6, "222168//Propofol", 6, 7, planned=10, delivered=10, amount=40, known=9),
    ]
    write_meds(root / "inputevents/INFUSION_START.parquet", [started for started, _ in records])
    write_meds(root / "inputevents/INFUSION_END.parquet", [ended for _, ended in records] + [
        # No start event: the record cannot be dated.
        {"stay_id": 1, "hadm_id": 101, "orderid": 4, "time": hour(5), "code": "INFUSION_END//222168//Propofol",
         "text_value": "FinishedRunning", "rate": np.float32(10), "amount": np.float32(50)},
    ])
    order = {"hadm_id": 101, "code": "MEDICATION_ORDER//Vancomycin", "text_value": "1g Frozen Bag",
             "dose": "1000", "dose_uom": "mg", "route": "IV", "time": hour(0)}
    write_meds(root / "prescriptions/MEDICATION_ORDER.parquet", [
        {**order, "prescribed_start": hour(2), "prescribed_end": hour(30)},
        # An order that ends before it starts is dropped; one outside the stay as well.
        {**order, "prescribed_start": hour(3), "prescribed_end": hour(2)},
        {**order, "prescribed_start": hour(9), "prescribed_end": hour(12)},
        {**order, "code": "MEDICATION_ORDER//Aspirin ", "dose": "81-325", "route": None, "text_value": None,
         "prescribed_start": hour(4), "prescribed_end": pd.NaT},
    ])
    write_meds(root / "emar/MEDICATION_ADMINISTRATION.parquet", [
        {"hadm_id": 101, "time": hour(2.5), "code": "MEDICATION_ADMINISTRATION//Vancomycin", "text_value": status}
        for status in ("Administered", "Not Given")
    ])
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    out = Path(paths["output"]["cohort"])
    out.mkdir(parents=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    return paths


def test_source_records_are_read_from_the_extraction_as_declared(tmp_path):
    paths = source_record_project(tmp_path)
    features = load_data_config(paths, "features")
    write_events(paths, "action")
    tables = all_actions(paths).source_table.value_counts().to_dict()
    # One norepinephrine interval per stay; the project holds no procedureevents extraction.
    assert tables == {"concept": 4, "inputevents": 4, "prescriptions": 2, "emar": 1}
    # A record without its start event cannot be dated and is left out.
    infusions = all_actions(paths).query("source_table == 'inputevents'")
    assert not (infusions.variable_name.eq("Propofol") & infusions.numeric_value.eq(50)).any()
    actions = pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet")
    records = actions[actions.source_table != "concept"].sort_values(["source_table", "timestamp_hours"])
    orders = records[records.source_table != "inputevents"]
    assert orders.source_table.tolist() == ["emar", "prescriptions", "prescriptions"]
    # The item name is the source's own, including its trailing space.
    assert orders.variable_name.tolist() == ["Vancomycin", "Vancomycin", "Aspirin "]
    assert orders.category.tolist() == ["medication_admin", "medication_order", "medication_order"]
    assert orders.text_value.tolist() == ["Administered", "route:IV | 1g Frozen Bag", ""]
    np.testing.assert_allclose(orders.numeric_value, [np.nan, 1000, np.nan])
    assert orders.timestamp_hours.tolist() == [2.5, 2, 4]
    # Intervals end with the stay; an order without an end is a point.
    np.testing.assert_array_equal(orders.end_timestamp_hours, [np.nan, 8, np.nan])
    assert orders.is_continuous.tolist() == [False, True, False]
    assert set(actions.loc[actions.source_table == "concept", "variable_name"]) == {"norepinephrine_rate"}
    # Other stays have none, and the switch in the feature profile removes them all.
    others = pd.read_parquet(Path(paths["output"]["actions"]) / "2.parquet")
    assert set(others.source_table) == {"concept"}
    features["raw_actions"]["enabled"] = False
    write_events(paths, "action", features)
    assert set(pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet").source_table) == {"concept"}


def test_infusion_states_the_rate_that_was_set_and_the_amount_only_once_it_is_known(tmp_path):
    paths = source_record_project(tmp_path)
    write_events(paths, "observation")
    write_events(paths, "action")
    actions = pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet")
    infusions = actions[actions.source_table == "inputevents"].sort_values("timestamp_hours")
    assert infusions.variable_name.tolist() == ["Propofol", "Fentanyl", "Propofol", "Propofol"]
    assert infusions.is_continuous.tolist() == [True, False, False, True]
    # While it runs: the rate entered at the start, not the 20.1 that was delivered, and no amount.
    assert infusions.text_value.tolist() == [
        "rate:20.0 mcg/kg/min | Continuous Med", "Drug Push", "", "rate:10.0 mcg/kg/min | Continuous Med"]
    np.testing.assert_allclose(infusions.numeric_value, [np.nan, 50, 200, np.nan])
    assert infusions.unit.tolist() == ["", "mcg", "mg", ""]
    np.testing.assert_allclose(infusions.timestamp_hours, [1, 2 + 1 / 60, 5, 6])
    np.testing.assert_array_equal(infusions.end_timestamp_hours, [3, np.nan, np.nan, 7])

    timeline = build_stay_timeline(1, Path(paths["output"]["observations"]), Path(paths["output"]["actions"]))
    stay = discretize_stay(timeline, 1, {"icu_los_hours": 8.0}, 1, 336, 1)
    texts = hourly_texts(filter_act_events(stay["act_events"]), 8, "none")
    assert texts[1] == "t=1h | Norepinephrine: 0.1 mcg/kg/min. Propofol, 20 mcg/kg/min Continuous Med"
    assert "Propofol, 20 mcg/kg/min Continuous Med" in texts[2] and "Fentanyl: 50 mcg" in texts[2]
    # No hour before the amount was charted mentions it.
    assert not any("200 mg" in text for text in texts[:5])
    assert texts[5] == "t=5h | Propofol: 200 mg"
    assert texts[6] == "t=6h | Propofol, 10 mcg/kg/min Continuous Med" and not any("40 mg" in text for text in texts)


def test_text_of_a_record_with_a_start_event_must_come_from_that_event(tmp_path):
    paths = source_record_project(tmp_path)
    project = Project(paths)
    cohort = pd.read_csv(Path(paths["output"]["cohort"]) / "cohort.csv", parse_dates=["icu_intime", "icu_outtime"])
    spec = dict(project.profile["raw_actions"]["inputevents"], text={"rate": "rate", "rate_unit": "rateuom"})
    with pytest.raises(ValueError, match="start.<column>"):
        source_records(project, "inputevents", spec, set(cohort.subject_id), build_icu_windows(cohort),
                       build_intime_map(cohort), build_outtime_map(cohort))


def test_dataset_without_declared_source_records_has_none(tmp_path):
    paths = make_project(tmp_path, "aumc")
    cohort = build_cohort(paths, load_data_config(paths, "cohort"))
    out = Path(paths["output"]["cohort"])
    out.mkdir(parents=True)
    cohort.to_csv(out / "cohort.csv", index=False)
    write_events(paths, "action")
    assert set(pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet").source_table) == {"concept"}
