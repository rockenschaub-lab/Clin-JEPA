"""Exercise both actual CLI pipelines using synthetic, repository-local exports."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_weavehr_steps import make_project

from clin_jepa.data.helpers.source import dataset_name, load_data_config
from clin_jepa.training.trajectory_dataset import TrajectoryWindowDataset

REPO = Path(__file__).resolve().parents[1]


def cli(module, *args, env):
    result = subprocess.run(
        [sys.executable, "-m", f"clin_jepa.data.{module}", *map(str, args)],
        cwd=REPO, env=env, text=True, capture_output=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def synthetic_inputs(root, dataset):
    paths = make_project(root, dataset, n=24)
    output = Path(paths["data_dir"])
    features_file, cohort_file = [root / name for name in ("features.yaml", "cohort.yaml")]
    for file, cfg in ((features_file, load_data_config(paths, "features")),
                      (cohort_file, load_data_config(paths, "cohort"))):
        file.write_text(yaml.safe_dump(cfg))
    return output, paths["weavehr"]["project"], features_file, cohort_file


@pytest.mark.parametrize("dataset", ["mimic-iv", "aumc"])
def test_actual_six_step_pipeline(tmp_path, dataset):
    output, project, features, cohort_config = synthetic_inputs(tmp_path / dataset, dataset)
    env = {**os.environ, "CLIN_JEPA_DATA": str(output)}
    data = ("--data_dir", output)
    located = (*data, "--dataset", dataset, "--weavehr_project", project)
    cli("step01_cohort", *located, "--cohort_config", cohort_config, env=env)
    cohort = pd.read_csv(output / "cohort/cohort.csv")
    assert cohort.sepsis3.isna().all()
    if dataset == "aumc":
        assert cohort.hospital_expire_flag.isna().all()
    for module in ("step02_observations", "step03_actions"):
        cli(module, *located, "--features_config", features, env=env)
    # Steps 04 to 06 are upstream's: they read only this pipeline's own output.
    cli("step04_discretization", *data, "--features_config", features, env=env)
    cli("step05_split", *data, env=env)
    cli("step06_trajectories", *data, "--n_workers", 1, env=env)
    splits = pd.read_csv(output / "splits/splits.csv")
    assert len(splits) == 24
    assert splits.groupby("subject_id")["split"].nunique().max() == 1
    assert set(splits["split"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        ds = TrajectoryWindowDataset(output / "trajectories", split=split)
        assert len(ds)
        item = ds[0]
        assert item["length"] == 8
        assert "Heart rate: 70 bpm" in item["state_texts"][0]
        assert "Norepinephrine: 0.1" in item["action_texts"][1]
        assert "Gender: Male. Race: " in item["demographics_text"]


def test_dataset_is_never_assumed():
    with pytest.raises(ValueError, match="No dataset named"):
        dataset_name({})
    assert dataset_name({"dataset": "mimic-iv"}) == "mimic-iv"
    # The demo shares MIMIC-IV's definition.
    assert dataset_name({"dataset": "mimic-iv-demo"}) == "mimic-iv"
    with pytest.raises(ValueError, match="Unsupported dataset"):
        dataset_name({"dataset": "unexpected"})
