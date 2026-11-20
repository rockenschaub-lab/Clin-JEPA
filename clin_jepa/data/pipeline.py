"""Run the data pipeline, Steps 01 to 06, on a WeavEHR project.

    python -m clin_jepa.data.pipeline \\
        --weavehr_project /path/to/weavehr-project --data_dir /path/to/output --dataset mimic-iv

Each step is the same command you can run on its own (``python -m clin_jepa.data.step01_cohort``, ...),
for example to spread Steps 02 to 04 over several workers. ``--from_step`` resumes after a failure.
"""

import argparse
import logging
import subprocess
import sys

from clin_jepa.data.helpers.source import add_data_arguments
from clin_jepa.utils import setup_logging

logger = logging.getLogger(__name__)

STEPS = [
    "step01_cohort",
    "step02_observations",
    "step03_actions",
    "step04_discretization",
    "step05_split",
    "step06_trajectories",
]


def step_arguments(number: int, args: argparse.Namespace) -> list[str]:
    """What one step is told: Steps 01 to 03 read the WeavEHR project, the others only this pipeline's output."""
    arguments = ["--data_dir", args.data_dir]
    if number <= 3:
        arguments += ["--dataset", args.dataset, "--weavehr_project", args.weavehr_project]
        if args.dataset_version:
            arguments += ["--dataset_version", args.dataset_version]
    if number == 6 and args.n_workers is not None:
        arguments += ["--n_workers", str(args.n_workers)]
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_arguments(parser)
    parser.add_argument("--from_step", type=int, default=1, choices=range(1, 7), help="First step to run")
    parser.add_argument("--to_step", type=int, default=6, choices=range(1, 7), help="Last step to run")
    parser.add_argument("--n_workers", type=int, default=None, help="Processes for Step 06")
    args = parser.parse_args()
    setup_logging()
    for number in range(args.from_step, args.to_step + 1):
        module = f"clin_jepa.data.{STEPS[number - 1]}"
        logger.info("Running %s", module)
        result = subprocess.run([sys.executable, "-m", module, *step_arguments(number, args)], check=False)
        if result.returncode:
            sys.exit(f"{module} failed; fix the cause and continue with --from_step {number}")


if __name__ == "__main__":
    main()
