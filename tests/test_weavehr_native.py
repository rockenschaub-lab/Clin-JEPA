"""Actual pinned WeavEHR, run as its quickstart documents, followed by the Clin-JEPA CLI.

Set WEAVEHR_PYTHON or install requirements-weavehr.txt in .venv-weavehr to run.
No MIMIC credentials, OMOP vocabulary download or DuckDB required.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from test_weavehr_steps import make_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["mimic-iv", "aumc"])
def test_native_weavehr_to_trajectories_and_evaluation(tmp_path, dataset):
    python = Path(os.environ.get("WEAVEHR_PYTHON", ROOT / ".venv-weavehr/bin/python"))
    if not python.exists():
        pytest.skip("Install requirements-weavehr.txt in Python 3.13; set WEAVEHR_PYTHON")
    # 12 patients with 2 stays each: enough for upstream's 70/15/15 patient split.
    paths = make_project(tmp_path, dataset, n=24)
    shutil.rmtree(paths["weavehr"]["project"])
    raw = tmp_path / "raw"
    env = {
        **os.environ,
        "CLIN_JEPA_DATA": paths["data_dir"],
        "POLARS_MAX_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    def run(command):
        result = subprocess.run(
            [str(x) for x in command], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-8000:]

    # Raw files, then WeavEHR's own quickstart: Clin-JEPA adds nothing to the project.
    run([python, "tests/weavehr_native_fixture.py", raw, dataset, 24, paths["weavehr"]["project"]])
    # One command runs Steps 01 to 06; the same three arguments for both datasets.
    run([sys.executable, "-m", "clin_jepa.data.pipeline", "--data_dir", paths["data_dir"], "--dataset", dataset,
         "--weavehr_project", paths["weavehr"]["project"], "--n_workers", 1])
    # Age and sex are WeavEHR concepts for both datasets; the fixture records no race.
    cohort = pd.read_csv(Path(paths["output"]["cohort"]) / "cohort.csv")
    assert cohort.admission_age.eq(45).all() and cohort.gender.eq("M").all()
    assert cohort.race.eq("UNKNOWN").all()
    obs = pd.read_parquet(Path(paths["output"]["observations"]) / "1.parquet")
    assert obs.loc[obs.variable_name == "heart_rate", "numeric_value"].tolist() == list(range(70, 78))
    # Independently known unit equivalence: AUMC 88.4 umol/L -> 1 mg/dL;
    # MIMIC 1 mg/dL is unchanged by its own WeavEHR mapping.
    assert set(obs.source_table) == {"concept"}
    assert obs.loc[obs.variable_name == "creatinine", "numeric_value"].iloc[0] == pytest.approx(1.0, rel=1e-3)
    act = pd.read_parquet(Path(paths["output"]["actions"]) / "1.parquet")
    concepts = act[act.source_table == "concept"]
    ne = concepts[concepts.variable_name == "norepinephrine_rate"]
    assert ne.timestamp_hours.tolist() == [1.0] and ne.end_timestamp_hours.tolist() == [4.0]
    assert ne.numeric_value.tolist() == pytest.approx([0.1])
    ventilation = concepts[concepts.variable_name == "ventilation"]
    assert ventilation.timestamp_hours.tolist() == [0.5]
    assert ventilation.end_timestamp_hours.tolist() == [2.5]
    assert ventilation.text_value.tolist() == ["invasive"]
    shards = sorted(Path(paths["output"]["trajectories"]).glob("*/*.pt"))
    assert shards
    texts = {}
    for path in shards:
        shard = torch.load(path, weights_only=False)
        assert len(shard["windows"]) > 0
        hours = shard["per_hour"]
        for key in ("state_texts", "action_texts"):
            texts.update({(key, s, h): t for s, h, t in zip(hours["stay_ids"], hours["hours"], hours[key])})
    assert "Heart rate: 71 bpm" in texts["state_texts", 1, 1] and "Creatinine: 1 mg/dL" in texts["state_texts", 1, 2]
    raw = act[act.source_table != "concept"].set_index("source_table")
    if dataset == "aumc":
        # No unharmonised source records are declared for this dataset.
        assert raw.empty
        assert texts["action_texts", 1, 1] == "t=1h | Norepinephrine: 0.1 mcg/kg/min. Ventilation status: invasive"
    else:
        assert sorted(set(raw.index)) == ["emar", "inputevents", "prescriptions", "procedureevents"]
        # While the infusion runs: the rate entered at its start. Its amount is known at the end.
        infusion = raw.loc["inputevents"].sort_values("timestamp_hours")
        assert infusion.variable_name.tolist() == ["Norepinephrine"] * 2
        assert infusion.is_continuous.tolist() == [True, False]
        assert infusion.text_value.tolist() == ["rate:0.1 mcg/kg/min | Continuous Med", ""]
        assert infusion.numeric_value.tolist()[1] == pytest.approx(1.44) and infusion.unit.tolist() == ["", "mg"]
        assert infusion.timestamp_hours.tolist() == [1.0, 4.0] and infusion.end_timestamp_hours.tolist()[0] == 4.0
        order = raw.loc["prescriptions"]
        assert (order.variable_name, order.numeric_value, order.unit) == ("Vancomycin", 1000.0, "mg")
        assert order.text_value == "route:IV | 1g Frozen Bag" and order.category == "medication_order"
        assert (order.timestamp_hours, order.end_timestamp_hours) == (1.5, 5.0)
        # The dose that was not given is no administration.
        given = raw.loc["emar"]
        assert (given.variable_name, given.text_value, given.timestamp_hours) == ("Vancomycin", "Administered", 2.0)
        procedure = raw.loc["procedureevents"].sort_values("timestamp_hours")
        assert procedure.variable_name.tolist() == ["Invasive Ventilation"] * 2
        assert procedure.timestamp_hours.tolist() == [0.5, 2.5] and procedure.numeric_value.tolist()[1] == 120.0
        # Concepts first (the order is also WeavEHR's antibiotic marker, which carries the
        # product strength, not the drug), then the source records; the order appears once
        # and procedures never.
        assert texts["action_texts", 1, 1] == (
            "t=1h | Norepinephrine: 0.1 mcg/kg/min. Ventilation status: invasive. Antibiotics: 1. "
            "Norepinephrine, 0.1 mcg/kg/min Continuous Med. Vancomycin: 1000 mg 1g Frozen Bag IV"
        )
        assert "Vancomycin: Administered" in texts["action_texts", 1, 2]
        assert not any(text in texts["action_texts", 1, 2] for text in ("Frozen Bag", "Invasive Ventilation"))
        # The delivered amount is stated once the infusion has ended, never before.
        assert not any("1.44" in texts["action_texts", 1, hour] for hour in range(4))
        assert "Norepinephrine: 1.44 mg" in texts["action_texts", 1, 4]
    downstream = tmp_path / "downstream.yaml"
    downstream.write_text(
        yaml.safe_dump({"data_dir": paths["data_dir"], "output_dir": str(tmp_path / "downstream"), "n_workers": 1})
    )
    run([sys.executable, "-m", "clin_jepa.downstream.labels", "--config", downstream,
         "--weavehr_project", paths["weavehr"]["project"], "--dataset", dataset])
    labels = pd.read_parquet(tmp_path / "downstream/labels/stay.parquet").set_index("stay_id").sort_index()
    if dataset == "aumc":
        # No validated follow-up: nothing is turned into a negative outcome.
        assert labels.hospital_mortality.isna().all() and labels.mortality_7d.isna().all()
        assert labels.discharge_home.isna().all() and labels.hospital_readmission_30d.isna().all()
    else:
        # Stay 4 dies at 7.5 h, on the day after its patient's stay 3 began.
        cohort = pd.read_csv(Path(paths["output"]["cohort"]) / "cohort.csv").set_index("stay_id")
        labels = labels.loc[[1, 2, 3, 4]]
        assert labels.death_in_icu.tolist() == [0, 0, 0, 1]
        assert labels.icu_mortality.tolist() == [0, 0, 0, 1]
        assert labels.mortality_7d.tolist() == [0, 0, 1, 1]
        assert labels.discharge_home.tolist() == [1, 1, 1, 0]
        # Each patient's second admission follows the first within a day.
        assert labels.hospital_readmission_30d.tolist() == [1, 0, 1, 0]
        assert labels.icu_readmission_same_hosp.tolist() == [0, 0, 0, 0]
    project_contents = sorted(path.name for path in Path(paths["weavehr"]["project"]).iterdir())
    assert project_contents == ["configs", "datasets", "workspace"]
