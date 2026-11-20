"""Dataset definitions, and the command-line arguments that locate a run."""

from pathlib import Path

from clin_jepa.utils import expand_path, load_config

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "data"
OUTPUTS = ("cohort", "observations", "actions", "discretized", "splits", "trajectories")


def add_data_arguments(parser, *, project: bool = True, dataset: bool = True) -> None:
    """The arguments that say what a step reads and where it writes."""
    if project:
        parser.add_argument(
            "--weavehr_project", required=True, help="WeavEHR project with completed extraction and concept steps"
        )
    parser.add_argument(
        "--data_dir", required=True, help="Directory this pipeline writes to (cohort, observations, ..., trajectories)"
    )
    if dataset:
        parser.add_argument("--dataset", required=True, help="WeavEHR name of the dataset, e.g. mimic-iv or aumc")
    if project:
        parser.add_argument(
            "--dataset_version", default=None, help="Only needed when the project holds several versions of the dataset"
        )


def data_paths(args) -> dict:
    """What a step reads and where it writes, from its command-line arguments."""
    base = expand_path(args.data_dir)
    paths = {"data_dir": str(base), "output": {"base": str(base), **{name: str(base / name) for name in OUTPUTS}}}
    for key in ("dataset", "dataset_version"):
        if getattr(args, key, None):
            paths[key] = getattr(args, key)
    if getattr(args, "weavehr_project", None):
        paths["weavehr"] = {"project": str(expand_path(args.weavehr_project))}
    return paths


def project_dataset(paths: dict) -> tuple[str, str]:
    """The dataset asked for and its version, as extracted in the WeavEHR project.

    WeavEHR stores an extraction under ``datasets/extraction/data/<dataset>/<version>``.
    """
    if "weavehr" not in paths:
        raise ValueError("This pipeline requires a WeavEHR project (--weavehr_project)")
    name = paths["dataset"]
    root = Path(paths["weavehr"]["project"]) / "datasets" / "extraction" / "data"
    versions = sorted(path.name for path in (root / name).glob("*") if path.is_dir())
    wanted = paths.get("dataset_version")
    chosen = [version for version in versions if wanted in (None, version)]
    if len(chosen) != 1:
        held = sorted(f"{path.parent.name} {path.name}" for path in root.glob("*/*") if path.is_dir())
        raise ValueError(
            f"Expected one extraction of {name}{' ' + str(wanted) if wanted else ''} in {root}, found {held}. "
            "Build the project as WeavEHR documents; if it holds several versions, pass --dataset_version"
        )
    return name, chosen[0]


def _dataset(name: str) -> dict:
    """The definition of a dataset: ``configs/data/<name>/dataset.yaml``, where the folder is its WeavEHR name.

    A dataset that shares another's definition, such as a demo subset, holds only ``extends: <name>``.
    The returned definition carries the name of the folder it was read from as ``dataset``.
    """
    path = CONFIG_ROOT / name / "dataset.yaml"
    if not path.is_file():
        known = sorted(found.parent.name for found in CONFIG_ROOT.glob("*/dataset.yaml"))
        raise ValueError(f"Unsupported dataset {name!r}: there is no {path} (available: {known})")
    definition = load_config(path)
    return _dataset(definition["extends"]) if "extends" in definition else {"dataset": name, **definition}


def dataset_name(config: dict) -> str:
    """The dataset a config is for, as the name of the folder under ``configs/data`` that defines it."""
    if not config.get("dataset"):
        raise ValueError("No dataset named (--dataset)")
    return _dataset(config["dataset"])["dataset"]


def load_weavehr_profile(paths: dict) -> dict:
    """The definition of the dataset to read from the WeavEHR project."""
    if "weavehr" not in paths:
        raise ValueError("This pipeline requires a WeavEHR project (--weavehr_project)")
    return _dataset(paths["dataset"])


def load_data_config(paths_config: dict, kind: str, override: str | None = None) -> dict:
    if kind not in {"cohort", "features"}:
        raise ValueError(f"Unsupported data config kind: {kind}")
    path = override or CONFIG_ROOT / f"{kind}.yaml"
    cfg = load_config(path)
    if cfg.get("dataset") and dataset_name(cfg) != dataset_name(paths_config):
        raise ValueError(f"{kind} profile {path} does not match {dataset_name(paths_config)}")
    return cfg
