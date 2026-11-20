"""Both datasets share the same steps and feature profile; each is located by its arguments."""

import argparse

import pytest
from test_weavehr_steps import make_project

from clin_jepa.data.helpers.source import (
    add_data_arguments,
    data_paths,
    dataset_name,
    load_data_config,
    load_weavehr_profile,
)
from clin_jepa.data.helpers.weavehr import Project


def paths(dataset):
    return {"dataset": dataset, "weavehr": {}}


def test_one_feature_profile_for_both_datasets():
    m, a = paths("mimic-iv"), paths("aumc")
    for kind in ("features", "cohort"):
        assert load_data_config(m, kind) == load_data_config(a, kind)
    assert load_weavehr_profile(a)["cohort"]["stays"]["filters"] == {"care_site_id": ["1"]}


@pytest.mark.parametrize("dataset", ["mimic-iv", "aumc"])
def test_three_arguments_locate_everything(tmp_path, dataset):
    project = make_project(tmp_path, dataset)["weavehr"]["project"]
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    located = data_paths(parser.parse_args(
        ["--weavehr_project", project, "--data_dir", str(tmp_path / "processed"), "--dataset", dataset]
    ))
    assert located["output"]["cohort"] == str(tmp_path / "processed" / "cohort")
    assert located["output"]["trajectories"] == str(tmp_path / "processed" / "trajectories")
    held = Project(located)
    assert (held.dataset, held.version) == ((dataset, "1.5.0") if dataset == "aumc" else (dataset, "3.1"))
    assert held.profile["dataset"] == dataset_name(located) == dataset
    with pytest.raises(SystemExit):
        parser.parse_args(["--weavehr_project", project, "--data_dir", str(tmp_path)])


def test_rejects_missing_project_and_a_dataset_the_project_lacks(tmp_path):
    with pytest.raises(ValueError, match="requires a WeavEHR project"):
        load_weavehr_profile({"dataset": "mimic-iv"})
    with pytest.raises(ValueError, match="Incomplete WeavEHR extraction step"):
        Project({"dataset": "mimic-iv", "weavehr": {"project": str(tmp_path / "missing")}})
    held = make_project(tmp_path, "mimic-iv")
    with pytest.raises(ValueError, match="Expected one extraction of aumc .* found .'mimic-iv 3.1'."):
        Project({**held, "dataset": "aumc"})
    with pytest.raises(ValueError, match="Expected one extraction of mimic-iv 2.2"):
        Project({**held, "dataset_version": "2.2"})
    # The demo is read from its own extraction with MIMIC-IV's definition.
    assert dataset_name({"dataset": "mimic-iv-demo"}) == "mimic-iv"
