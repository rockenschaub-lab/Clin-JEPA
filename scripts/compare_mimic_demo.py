"""Run and audit original Clin-JEPA versus the WeavEHR-based pipeline on MIMIC-IV demo.

The original pipeline names its events after mimic-code tables and columns.
``configs/data/mimic_code_aliases.yaml`` lines them up with the WeavEHR concept
names; this script is the only reader of that file.

No patient-level data are written outside --work-dir. The report contains only
aggregates. See docs/demo-comparison.md for versions, commands and limitations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ALIASES = REPO / "configs/data/mimic_code_aliases.yaml"
CONCEPT = "concept"  # source_table of harmonised events (clin_jepa.data.helpers.schema.CONCEPT_SOURCE)
STEPS = [
    "step01_cohort",
    "step02_observations",
    "step03_actions",
    "step04_discretization",
    "step05_split",
    "step06_trajectories",
]


def command(args, cmd, *, cwd=REPO, name, env=None):
    log = args.work_dir / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    run_env = dict(
        os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", POLARS_MAX_THREADS="2"
    )
    # Import from the selected checkout, never from another editable install.
    run_env["PYTHONPATH"] = str(cwd)
    run_env.update(env or {})
    with log.open("w") as stream:
        completed = subprocess.run([str(c) for c in cmd], cwd=cwd, env=run_env, stdout=stream, stderr=subprocess.STDOUT)
    entry = {
        "name": name,
        "command": [str(c) for c in cmd],
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "seconds": round(time.monotonic() - started, 3),
    }
    with (args.work_dir / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    if completed.returncode:
        print(log.read_text()[-6000:], file=sys.stderr)
        raise SystemExit(completed.returncode)


def prepare(args):
    import importlib.metadata as metadata

    raw = args.work_dir / "mimic-iv-clinical-database-demo-2.2"
    hashes = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        actual = hashlib.sha256((raw / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        hashes[relative] = actual
    (args.work_dir / "verified-inputs.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for variant in ("original", "adapted"):
        output = args.work_dir / variant
        if variant == "original":
            # Only the original pipeline has a paths file; this one takes its locations as arguments.
            text = (args.original_repo / "configs/data/paths.yaml").read_text()
            text = text.replace("${MIMIC_RAW}", str(raw)).replace("${CLIN_JEPA_DATA}", str(output))
            (args.work_dir / "original-paths.yaml").write_text(yaml.safe_dump(yaml.safe_load(text), sort_keys=False))
        downstream = {"data_dir": str(output), "output_dir": str(output / "downstream"), "n_workers": 1}
        if variant == "original":
            downstream["mimic"] = {
                "icustays": str(raw / "icu/icustays.csv.gz"),
                "admissions": str(raw / "hosp/admissions.csv.gz"),
                "patients": str(raw / "hosp/patients.csv.gz"),
                "ventilation": str(raw / "concepts/treatment/ventilation.csv"),
            }
        (args.work_dir / f"{variant}-downstream.yaml").write_text(yaml.safe_dump(downstream))
    pins = {
        "clin_jepa_original": args.original_repo,
        "clin_jepa_adapted": REPO,
        "mimic_code": args.mimic_code,
        "weavehr": REPO.parent / "weavehr",
    }
    provenance = {
        k: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p, text=True).strip() for k, p in pins.items()
    }
    provenance["dataset_url"] = "https://physionet.org/content/mimic-iv-demo/2.2/"
    provenance["duckdb_cli"] = subprocess.check_output([args.duckdb_cli, "--version"], text=True).strip()
    (args.work_dir / "versions.json").write_text(json.dumps(provenance, indent=2) + "\n")
    environments = {
        "clin_jepa": {
            "python": sys.version,
            "packages": {
                name: metadata.version(name)
                for name in ["numpy", "pandas", "pyarrow", "torch", "scikit-learn", "pyyaml"]
            },
        }
    }
    code = (
        "import sys,json,importlib.metadata as m; "
        "print(json.dumps({'python':sys.version,'packages':{n:m.version(n) "
        "for n in ['weavehr','polars','pyarrow','numpy','pandas','pyyaml']}}))"
    )
    environments["weavehr"] = json.loads(subprocess.check_output([args.weavehr_python, "-c", code], text=True))
    (args.work_dir / "environments.json").write_text(json.dumps(environments, indent=2) + "\n")
    print(f"Verified {len(hashes)} input files; prepared isolated configs.")


def reference(args):
    raw = args.work_dir / "mimic-iv-clinical-database-demo-2.2"
    db = args.work_dir / "mimic4.db"
    command(
        args,
        ["bash", args.mimic_code / "mimic-iv/buildmimic/duckdb/build_mimic.sh", raw, db],
        name="mimic-code-build",
        env={
            "PATH": str(args.duckdb_cli.parent) + os.pathsep + os.environ["PATH"],
            "MIMIC_VALIDATE": "true",
            "MIMIC_MAKE_CONCEPTS": "true",
        },
    )
    # Use the same CLI version for the export as for the build. Boolean t/f
    # exports match the format accepted by the unmodified upstream cohort step.
    sql = []
    for line in (args.mimic_code / "mimic-iv/concepts_duckdb/duckdb.sql").read_text().splitlines():
        if not line.strip().startswith(".read "):
            continue
        relative = Path(shlex.split(line)[1])
        table = relative.stem
        target = raw / "concepts" / relative.with_suffix(".csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        query = (
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema='mimiciv_derived' AND table_name='{table}' ORDER BY ordinal_position"
        )
        cols = json.loads(subprocess.check_output([args.duckdb_cli, db, "-json", "-c", query], text=True))
        expressions = []
        for col in cols:
            name = '"' + col["column_name"].replace('"', '""') + '"'
            expressions.append(
                f"CASE WHEN {name} THEN 't' WHEN NOT {name} THEN 'f' END AS {name}"
                if col["data_type"] == "BOOLEAN"
                else name
            )
        escaped = str(target).replace("'", "''")
        sql.append(
            f'COPY (SELECT {", ".join(expressions)} FROM mimiciv_derived."{table}") '
            f"TO '{escaped}' (HEADER, DELIMITER ',');"
        )
    sql_path = args.work_dir / "export-concepts.sql"
    sql_path.write_text("\n".join(sql) + "\n")
    command(args, [args.duckdb_cli, db, "-f", sql_path], name="mimic-code-export")
    print(f"Exported {len(sql)} reference concept tables.")


def pipelines(args, variant):
    repo = args.original_repo if variant == "original" else REPO
    env = {
        "CLIN_JEPA_DATA": str(args.work_dir / variant),
        "CLIN_JEPA_OUTPUT": str(args.work_dir / variant / "model-output"),
    }
    project = ["--weavehr_project", args.work_dir / "weavehr-project"]
    data = ["--data_dir", args.work_dir / variant]
    dataset = ["--dataset", "mimic-iv-demo"]
    for number, step in enumerate(STEPS[args.start_step - 1 :], start=args.start_step):
        if variant == "original":
            located = ["--paths_config", args.work_dir / "original-paths.yaml"]
        else:
            # Steps 01 to 03 read the WeavEHR project; Steps 04 to 06 only this pipeline's output.
            located = data + (dataset + project if number <= 3 else [])
        cmd = [args.python, "-m", f"clin_jepa.data.{step}", *located]
        if step == "step06_trajectories":
            cmd += ["--n_workers", "1"]
        command(args, cmd, cwd=repo, name=f"{variant}-{step}", env=env)
    command(
        args,
        [args.python, "-m", "clin_jepa.downstream.labels", "--config", args.work_dir / f"{variant}-downstream.yaml"]
        + ([] if variant == "original" else [*project, *dataset]),
        cwd=repo,
        name=f"{variant}-labels",
    )


def clean_json(value):
    """Strict JSON: nonfinite summary values are null, never JSON NaN."""
    import math

    import numpy as np

    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def paired_stats(left, right):
    """Series indexed by stay/time; missing data are never imputed."""
    import numpy as np
    import pandas as pd

    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError("Comparison keys must be unique after explicit aggregation")
    paired = pd.concat([left.rename("original"), right.rename("adapted")], axis=1)
    paired = paired.replace([np.inf, -np.inf], np.nan)
    a, b = paired.original, paired.adapted
    overlap = a.notna() & b.notna()
    valid = paired.loc[overlap]
    diff = (valid.original - valid.adapted).abs()
    agree = np.isclose(valid.original, valid.adapted, atol=1e-6, rtol=1e-5)
    n_a, n_b, n_both = int(a.notna().sum()), int(b.notna().sum()), int(overlap.sum())
    n_union = n_a + n_b - n_both
    return {
        "original": n_a,
        "adapted": n_b,
        "paired": n_both,
        "original_only": n_a - n_both,
        "adapted_only": n_b - n_both,
        "original_coverage_pct": 100 * n_both / n_a if n_a else None,
        "overlap_jaccard_pct": 100 * n_both / n_union if n_union else None,
        "equal": int(agree.sum()),
        "equal_pct": 100 * agree.mean() if n_both else None,
        "mae": diff.mean(),
        "median_abs_difference": diff.median(),
        "p95_abs_difference": diff.quantile(0.95) if n_both else None,
        "max_abs_difference": diff.max() if n_both else None,
    }


def load_naming():
    """Original names per concept, and each concept's category in the adapted profile."""
    naming = yaml.safe_load(ALIASES.read_text())
    features = yaml.safe_load((REPO / "configs/data/features.yaml").read_text())
    naming["category"] = {
        rule.get("name", rule["concept"]): rule["category"]
        for domain in ("observations", "actions") for rule in features[domain]
    }
    return naming


def canonical_events(frame, variant, naming):
    """Original events under the WeavEHR concept names; values are never changed.

    An original event without an alias keeps its mimic-code names, so that it
    is reported as covered by the original pipeline only.
    """
    import pandas as pd

    frame = frame.copy()
    frame["unit"] = frame["unit"].fillna("")
    if variant != "original":
        return frame
    concepts = naming["concepts"]
    lookup = {(alias["source_table"], alias["variable_name"]): concept for concept, alias in concepts.items()}
    concept = pd.Series(
        [lookup.get(pair) for pair in zip(frame.source_table, frame.variable_name)], index=frame.index, dtype=object
    )
    known = concept.notna()
    # Unit labels verified against pinned mimic-code SQL: the values already
    # have the aliased unit, so only the annotation is corrected.
    unit = concept.map({name: alias["unit"] for name, alias in concepts.items() if "unit" in alias})
    frame.loc[unit.notna(), "unit"] = unit[unit.notna()]
    for variable, levels in naming.get("levels", {}).items():
        rows = concept == variable
        frame.loc[rows, "text_value"] = frame.loc[rows, "text_value"].replace(levels)
    frame.loc[known, "category"] = concept[known].map(naming["category"])
    frame.loc[known, "variable_name"] = concept[known]
    frame.loc[known, "source_table"] = CONCEPT
    return frame


def text_fragments(events, domain):
    """Fragments the adapted templates write for these binned events, one row per fragment."""
    import pandas as pd

    sys.path.insert(0, str(REPO))
    from clin_jepa.data.helpers.action_filter import filter_act_events
    from clin_jepa.data.helpers.observation_filter import filter_obs_events
    from clin_jepa.data.helpers.text_templates import format_event

    config = yaml.safe_load((REPO / "configs/data/trajectories.yaml").read_text())
    if domain == "observations":
        events = filter_obs_events(events, config["state_text"])
    else:
        events = pd.concat(
            [filter_act_events(rows) for _, rows in events.groupby("stay_id")] or [events.iloc[:0]], ignore_index=True
        )
    fragments = [
        format_event(row["source_table"], row["variable_name"], row.get("numeric_value"), row.get("text_value"),
                     row.get("unit", ""))
        for row in events.to_dict("records")
    ]
    out = events[["stay_id", "step_idx", "source_table", "variable_name"]].assign(fragment=fragments)
    return out.dropna(subset=["fragment"])


def text_agreement(original, adapted, concepts, record_tables):
    """Compare the fragments of both pipelines hour by hour, as multisets.

    Both pipelines can state the aliased ``concepts`` and the source records
    of the ``record_tables``. Everything else exists in one vocabulary only
    and is counted, not compared.
    """
    import pandas as pd

    keys = ["stay_id", "step_idx", "source_table", "variable_name", "fragment"]
    counts = {}
    for name, frame in (("original", original), ("adapted", adapted)):
        shared = ((frame.source_table == CONCEPT) & frame.variable_name.isin(concepts)) | frame.source_table.isin(
            record_tables
        )
        counts[name + "_outside_shared_vocabulary"] = int((~shared).sum())
        counts[name] = frame.loc[shared].groupby(keys).size().rename(name)
    both = pd.concat([counts.pop("original"), counts.pop("adapted")], axis=1).fillna(0).astype(int)
    both["equal"] = both.min(axis=1)
    both = both.reset_index()
    both["variable"] = both.variable_name.where(both.source_table == CONCEPT, "(all items)")
    hours = both.groupby(["stay_id", "step_idx"])[["original", "adapted", "equal"]].sum()
    identical = (hours.original == hours.equal) & (hours.adapted == hours.equal)
    rows = both.groupby(["source_table", "variable"])[["original", "adapted", "equal"]].sum().reset_index()
    rows["equal_pct_of_original"] = (100 * rows.equal / rows.original).where(rows.original > 0)
    return {
        **counts,
        "hours_with_text": len(hours),
        "hours_with_identical_fragments": int(identical.sum()),
        "fragments": {name: int(both[name].sum()) for name in ("original", "adapted", "equal")},
        "by_variable": rows.to_dict("records"),
    }


def intervention_intervals(raw_frames, cohorts, common, *, ventilation=False):
    """Compare documented positive-rate intervals, not inferred absence of care."""
    import numpy as np
    import pandas as pd

    results = []
    variants = ("original", "adapted")
    # Concept names, plus what only the original pipeline has (milrinone, tracheostomy).
    names = [
        "dopamine_rate",
        "epinephrine_rate",
        "norepinephrine_rate",
        "phenylephrine_rate",
        "vasopressin_rate",
        "dobutamine_rate",
        "milrinone",
    ]
    if ventilation:
        names = ["invasive", "noninvasive", "Tracheostomy"]
    for drug in names:
        intervals, stats, rate_changes = {}, {}, {}
        for v in variants:
            df = raw_frames[v]["actions"]
            if ventilation:
                df = df.loc[(df.category == "ventilation") & (df.text_value == drug)].copy()
                df["numeric_value"] = 1.0
            else:
                df = df.loc[(df.category == "vasopressor") & (df.variable_name == drug)]
            valid = df.is_continuous & df.end_timestamp_hours.notna() & (df.end_timestamp_hours > df.timestamp_hours)
            stats[v] = {
                "all_events": len(df),
                "positive_interval_events": int((valid & (df.numeric_value > 0)).sum()),
                "positive_point_events": int((~valid & (df.numeric_value > 0)).sum()),
                "explicit_zero_events": int((df.numeric_value == 0).sum()),
            }
            intervals[v] = {}
            rate_changes[v] = {}
            for stay_id, group in df.loc[valid & (df.numeric_value > 0)].groupby("stay_id"):
                limit = min(
                    336.0,
                    *[
                        (
                            pd.to_datetime(cohorts[k].set_index("stay_id").loc[stay_id, "icu_outtime"])
                            - pd.to_datetime(cohorts[k].set_index("stay_id").loc[stay_id, "icu_intime"])
                        ).total_seconds()
                        / 3600
                        for k in variants
                    ],
                )
                spans = sorted(
                    (max(0.0, row.timestamp_hours), min(limit, row.end_timestamp_hours)) for row in group.itertuples()
                )
                changes = {}
                for row in group.itertuples():
                    start, end = max(0.0, row.timestamp_hours), min(limit, row.end_timestamp_hours)
                    if end > start:
                        changes[start] = changes.get(start, 0.0) + row.numeric_value
                        changes[end] = changes.get(end, 0.0) - row.numeric_value
                rate_changes[v][int(stay_id)] = changes
                merged = []
                for start, end in spans:
                    if end <= start:
                        continue
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                intervals[v][int(stay_id)] = merged
            stats[v]["positive_interval_hours"] = sum(b - a for spans in intervals[v].values() for a, b in spans)
            stats[v]["stays_with_positive_interval"] = sum(bool(spans) for spans in intervals[v].values())
        intersection = 0.0
        for stay_id in common:
            a, b = intervals["original"].get(stay_id, []), intervals["adapted"].get(stay_id, [])
            i = j = 0
            while i < len(a) and j < len(b):
                intersection += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
                if a[i][1] < b[j][1]:
                    i += 1
                else:
                    j += 1
        union = sum(stats[v]["positive_interval_hours"] for v in variants) - intersection
        weights, differences, relatives = [], [], []
        for stay_id in common:
            changes = {v: rate_changes[v].get(stay_id, {}) for v in variants}
            times = sorted(set(changes["original"]) | set(changes["adapted"]))
            rates = dict.fromkeys(variants, 0.0)
            for start, end in zip(times[:-1], times[1:]):
                for v in variants:
                    rates[v] += changes[v].get(start, 0.0)
                if min(rates.values()) > 1e-9:
                    difference = abs(rates["original"] - rates["adapted"])
                    weights.append(end - start)
                    differences.append(difference)
                    relatives.append(difference / abs(rates["original"]))
        dose = {"paired_positive_hours": sum(weights)}
        if weights:
            weights, differences, relatives = map(np.asarray, (weights, differences, relatives))
            order = np.argsort(relatives)
            cumulative = np.cumsum(weights[order]) / weights.sum()
            dose.update(
                duration_weighted_mae=float(np.average(differences, weights=weights)),
                duration_weighted_relative_mae_pct=float(100 * np.average(relatives, weights=weights)),
                median_relative_error_pct=float(100 * relatives[order][np.searchsorted(cumulative, 0.5)]),
                p95_relative_error_pct=float(100 * relatives[order][np.searchsorted(cumulative, 0.95)]),
                within_5pct_of_original_pct=float(100 * weights[relatives <= 0.05].sum() / weights.sum()),
            )
        results.append(
            {
                "status" if ventilation else "drug": drug,
                **stats,
                "intersection_hours": intersection,
                "overlap_jaccard_pct": 100 * intersection / union if union else None,
                "rates_on_shared_positive_intervals": dose,
            }
        )
    return results


def hourly_interval_rates(frame, limits):
    """Independent sweep-line reference: sum concurrent rates, then integrate.

    Coverage is the union of documented intervals, not their summed lengths.
    Point records, including stops, contribute no exposure duration.
    """
    import numpy as np
    import pandas as pd

    values, coverage = {}, {}
    valid = frame.is_continuous & np.isfinite(frame.numeric_value) & frame.end_timestamp_hours.notna()
    for stay_id, rows in frame.loc[valid].groupby("stay_id"):
        stay_id = int(stay_id)
        changes = {}
        for row in rows.itertuples():
            start, end = max(0., row.timestamp_hours), min(limits[stay_id], row.end_timestamp_hours)
            if end <= start:
                continue
            for timestamp, sign in [(start, 1), (end, -1)]:
                rate, count = changes.get(timestamp, (0., 0))
                changes[timestamp] = rate + sign * row.numeric_value, count + sign
        rate, count = 0., 0
        times = sorted(changes)
        numerator, duration = {}, {}
        for start, end in zip(times, times[1:]):
            delta, n = changes[start]
            rate, count = rate + delta, count + n
            if count <= 0:
                continue
            for hour in range(int(np.floor(start)), int(np.ceil(end))):
                hours = min(end, hour + 1.) - max(start, hour)
                numerator[hour] = numerator.get(hour, 0.) + rate * hours
                duration[hour] = duration.get(hour, 0.) + hours
        for hour, hours in duration.items():
            values[stay_id, hour] = numerator[hour] / hours
            coverage[stay_id, hour] = hours
    index = pd.MultiIndex.from_tuples(list(values), names=["stay_id", "step_idx"])
    return (pd.Series(list(values.values()), index=index, dtype=float),
            pd.Series(list(coverage.values()), index=index, dtype=float))


def unchanged_weavehr_concepts(args):
    """Compare the actual clinical event payloads, ignoring new metadata fields."""
    import numpy as np
    import pandas as pd

    if args.baseline_work_dir is None:
        return None
    old = args.baseline_work_dir / "weavehr-project/datasets/concept/data"
    new = args.work_dir / "weavehr-project/datasets/concept/data"
    core = ["subject_id", "time", "code", "numeric_value", "text_value"]
    rows = []
    for path in sorted(old.glob("*/*/*/*.parquet")):
        target = new / path.relative_to(old)
        a, b = pd.read_parquet(path, columns=core), pd.read_parquet(target, columns=core)
        hashes = [np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy()) for frame in [a, b]]
        rows.append({"concept": path.relative_to(old).parts[0], "before": len(a), "after": len(b),
                     "unchanged": bool(np.array_equal(*hashes))})
    return {"concepts": len(rows), "unchanged": sum(r["unchanged"] for r in rows), "details": rows}


def compare(args):
    import numpy as np
    import pandas as pd
    import torch

    variants = ("original", "adapted")
    naming = load_naming()
    cohorts = {v: pd.read_csv(args.work_dir / v / "cohort/cohort.csv") for v in variants}
    ids = {v: set(df.stay_id.astype(int)) for v, df in cohorts.items()}
    common = ids["original"] & ids["adapted"]
    raw = args.work_dir / "mimic-iv-clinical-database-demo-2.2"
    summary = {
        "versions": json.loads((args.work_dir / "versions.json").read_text()),
        "environments": json.loads((args.work_dir / "environments.json").read_text()),
        "raw": {
            name: len(pd.read_csv(raw / path))
            for name, path in [
                ("patients", "hosp/patients.csv.gz"),
                ("admissions", "hosp/admissions.csv.gz"),
                ("icu_stays", "icu/icustays.csv.gz"),
            ]
        },
        "cohort": {v: {"stays": len(cohorts[v]), "patients": cohorts[v].subject_id.nunique()} for v in variants},
        "common_stays": len(common),
        "original_only_stays": len(ids["original"] - common),
        "adapted_only_stays": len(ids["adapted"] - common),
    }
    details = pd.read_csv(raw / "concepts/demographics/icustay_detail.csv")
    extras = details.loc[details.stay_id.isin(ids["adapted"] - common)]
    summary["adapted_extra_stays_not_first_in_admission"] = int(
        (~extras.first_icu_stay.astype(str).str.lower().isin(["t", "true"])).sum()
    )
    indexed = {v: cohorts[v].set_index("stay_id").loc[sorted(common)] for v in variants}
    summary["cohort_field_agreement"] = {}
    for col in ["subject_id", "hadm_id", "gender", "race", "icu_intime", "icu_outtime", "hospital_expire_flag"]:
        a, b = indexed["original"][col], indexed["adapted"][col]
        known = a.notna() & b.notna()
        if col.endswith("time"):
            equal = pd.to_datetime(a[known]) == pd.to_datetime(b[known])
        else:
            equal = a[known] == b[known]
        summary["cohort_field_agreement"][col] = {"paired": int(known.sum()), "equal": int(equal.sum())}
    summary["cohort_field_agreement"]["admission_age"] = paired_stats(
        indexed["original"].admission_age, indexed["adapted"].admission_age
    )
    summary["cohort_field_agreement"]["icu_los_hours"] = paired_stats(
        indexed["original"].icu_los_hours, indexed["adapted"].icu_los_hours
    )
    septic = {v: pd.to_numeric(indexed[v].sepsis3, errors="raise") for v in variants}
    summary["sepsis"] = {
        v: {
            "positive": int((septic[v] == 1).sum()),
            "negative": int((septic[v] == 0).sum()),
            "unknown": int(septic[v].isna().sum()),
        }
        for v in variants
    }
    pos = {v: set(septic[v].index[septic[v] == 1]) for v in variants}
    summary["sepsis"]["positive_overlap"] = len(pos["original"] & pos["adapted"])
    summary["sepsis"]["original_positive_without_adapted_positive"] = len(pos["original"] - pos["adapted"])
    summary["sepsis"]["adapted_positive_without_original_positive"] = len(pos["adapted"] - pos["original"])
    onset = {
        v: (pd.to_datetime(indexed[v].sepsis_sofa_time) - pd.to_datetime(indexed[v].icu_intime)).dt.total_seconds()
        / 3600
        for v in variants
    }
    both_positive = sorted(pos["original"] & pos["adapted"])
    summary["sepsis"]["onset_hours"] = paired_stats(
        onset["original"].reindex(both_positive), onset["adapted"].reindex(both_positive)
    )

    raw_frames, binned, bins = {}, {}, {}
    summary["events"] = {}
    for v in variants:
        root = args.work_dir / v
        summary["events"][v] = {}
        raw_frames[v], binned[v], bins[v] = {}, {}, {}
        for domain in ["observations", "actions"]:
            frames = [pd.read_parquet(p) for p in sorted((root / domain).glob("*.parquet"))]
            df = pd.concat(frames, ignore_index=True)
            summary["events"][v][domain] = {"all_stays": len(df), "common_stays": int(df.stay_id.isin(common).sum())}
            raw_frames[v][domain] = canonical_events(df.loc[df.stay_id.isin(common)], v, naming)
        for path in sorted((root / "discretized").glob("*.pkl")):
            # Only deserialize the trusted, freshly generated local pipeline outputs.
            with path.open("rb") as stream:
                stay = pickle.load(stream)
            bins[v][int(stay["stay_id"])] = stay
        summary["events"][v]["discretized_stays"] = len(bins[v])
        summary["events"][v]["hours"] = sum(s["n_steps"] for s in bins[v].values())
    common_bins = common & set(bins["original"]) & set(bins["adapted"])
    summary["common_discretized_stays"] = len(common_bins)
    summary["stays_with_different_hour_count"] = sum(
        bins["original"][s]["n_steps"] != bins["adapted"][s]["n_steps"] for s in common_bins
    )
    limits = {s: min(bins[v][s]["n_steps"] for v in variants) for s in common_bins}
    summary["common_hours"] = sum(limits.values())
    for v in variants:
        for domain, field in [("observations", "obs_events"), ("actions", "act_events")]:
            frames = []
            for s in sorted(common_bins):
                df = bins[v][s][field]
                frames.append(df.loc[df.step_idx < limits[s]].assign(stay_id=s))
            binned[v][domain] = canonical_events(pd.concat(frames, ignore_index=True), v, naming)

    feature_rows, coverage_rows = [], []
    group = ["source_table", "variable_name", "unit"]
    for domain in ["observations", "actions"]:
        raw_groups = {v: {k: f for k, f in raw_frames[v][domain].groupby(group, dropna=False)} for v in variants}
        hour_groups = {v: {k: f for k, f in binned[v][domain].groupby(group, dropna=False)} for v in variants}
        for key in sorted(set(raw_groups["original"]) | set(raw_groups["adapted"])):
            row = dict(zip(group, key), domain=domain)
            for v in variants:
                df = raw_groups[v].get(key, pd.DataFrame())
                row[v + "_events"] = len(df)
                row[v + "_stays"] = df.stay_id.nunique() if len(df) else 0
            coverage_rows.append(row)
        for key in sorted(set(hour_groups["original"]) | set(hour_groups["adapted"])):
            numeric = {}
            agg = "sum" if key[:2] == (CONCEPT, "urine_output") else "mean"
            for v in variants:
                df = hour_groups[v].get(key)
                if df is None:
                    numeric[v] = pd.Series(
                        [], index=pd.MultiIndex.from_arrays([[], []], names=["stay_id", "step_idx"]), dtype=float
                    )
                else:
                    finite = df.loc[np.isfinite(pd.to_numeric(df.numeric_value, errors="coerce"))]
                    numeric[v] = finite.groupby(["stay_id", "step_idx"]).numeric_value.agg(agg)
            if len(numeric["original"]) or len(numeric["adapted"]):
                feature_rows.append(
                    dict(
                        zip(group, key),
                        domain=domain,
                        aggregation=agg,
                        **paired_stats(numeric["original"], numeric["adapted"]),
                    )
                )

    # The texts themselves: what the adapted templates write for each
    # pipeline's events, for the variables and source records both can state.
    profile = yaml.safe_load((REPO / "configs/data/mimic-iv/dataset.yaml").read_text())
    text = {
        domain: text_agreement(
            *(text_fragments(binned[v][domain], domain) for v in variants),
            set(naming["concepts"]), set(profile.get("raw_actions", {})),
        )
        for domain in ["observations", "actions"]
    }

    label_rows = []
    for level, keys in [("hourly", ["stay_id", "hour"]), ("stay", ["stay_id"])]:
        frames = {}
        for v in variants:
            path = args.work_dir / v / "downstream/labels" / f"{level}.parquet"
            df = pd.read_parquet(path)
            df = df.loc[df.stay_id.isin(common_bins)]
            if level == "hourly":
                df = df.loc[df.hour < df.stay_id.map(limits)]
            frames[v] = df.set_index(keys)
        for name in sorted(set(frames["original"].columns) | set(frames["adapted"].columns)):
            vals = {v: frames[v][name] if name in frames[v] else pd.Series(dtype=float) for v in variants}
            label_rows.append(
                {
                    "level": level,
                    "label": name,
                    **paired_stats(vals["original"], vals["adapted"]),
                    **{v + "_positive": int((vals[v] == 1).sum()) for v in variants},
                }
            )

    summary["splits"] = {}
    splits = {}
    for v in variants:
        splits[v] = pd.read_csv(args.work_dir / v / "splits/splits.csv").set_index("stay_id")
        summary["splits"][v] = splits[v].split.value_counts().to_dict()
    summary["splits"]["same_assignment_common_stays"] = int(
        (splits["original"].loc[sorted(common), "split"] == splits["adapted"].loc[sorted(common), "split"]).sum()
    )
    summary["trajectories"] = {}
    for v in variants:
        shards = sorted((args.work_dir / v / "trajectories").glob("*/*.pt"))
        windows, hours, stays = 0, 0, set()
        for path in shards:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            windows += len(shard["windows"])
            hours += len(shard["per_hour"]["stay_ids"])
            stays.update(int(x) for x in shard["per_stay"]["stay_ids"])
        summary["trajectories"][v] = {"shards": len(shards), "windows": windows, "hours": hours, "stays": len(stays)}
    intervals = intervention_intervals(raw_frames, cohorts, common)
    ventilation = intervention_intervals(raw_frames, cohorts, common, ventilation=True)
    summary["weavehr_definitions_preserved"] = unchanged_weavehr_concepts(args)
    for row in feature_rows + label_rows:
        assert 0 <= row["equal"] <= row["paired"] <= min(row["original"], row["adapted"])
        assert row["original_only"] + row["paired"] == row["original"]
        assert row["adapted_only"] + row["paired"] == row["adapted"]
    for row in intervals:
        assert np.isclose(row["intersection_hours"], row["rates_on_shared_positive_intervals"]["paired_positive_hours"])
    # Independently recompute selected headline metrics in DuckDB from the
    # native binned events. These private audit inputs remain under work-dir.
    verify_features = [f"{CONCEPT}.heart_rate", f"{CONCEPT}.creatinine", f"{CONCEPT}.sofa"]
    audit = args.work_dir / "audit"
    audit.mkdir(exist_ok=True)
    for v in variants:
        df = binned[v]["observations"].copy()
        df["feature"] = df.source_table + "." + df.variable_name
        df = df.loc[df.feature.isin(verify_features), ["feature", "stay_id", "step_idx", "numeric_value"]]
        df.to_parquet(audit / f"{v}.parquet", index=False)
    original_path, adapted_path = [str(audit / f"{v}.parquet").replace("'", "''") for v in variants]
    query = f"""
    WITH a AS (SELECT feature, stay_id, step_idx, AVG(numeric_value) AS value
               FROM read_parquet('{original_path}') WHERE isfinite(numeric_value) GROUP BY ALL),
         b AS (SELECT feature, stay_id, step_idx, AVG(numeric_value) AS value
               FROM read_parquet('{adapted_path}') WHERE isfinite(numeric_value) GROUP BY ALL)
    SELECT feature, COUNT(*) AS paired,
           SUM(CASE WHEN abs(a.value-b.value)<=1e-6+1e-5*abs(b.value) THEN 1 ELSE 0 END) AS equal,
           AVG(abs(a.value-b.value)) AS mae
    FROM a JOIN b USING (feature,stay_id,step_idx) GROUP BY feature ORDER BY feature
    """
    independent = json.loads(subprocess.check_output([args.duckdb_cli, "-json", "-c", query], text=True))
    assert len(independent) == len(verify_features)
    for row in independent:
        expected = next(x for x in feature_rows if x["source_table"] + "." + x["variable_name"] == row["feature"])
        assert row["paired"] == expected["paired"] and row["equal"] == expected["equal"]
        assert np.isclose(row["mae"], expected["mae"], atol=1e-10)
    summary["independent_duckdb_verification"] = independent
    results = {
        "summary": summary,
        "hourly_numeric": feature_rows,
        "event_coverage": coverage_rows,
        "text": text,
        "labels": label_rows,
        "pressor_intervals": intervals,
        "ventilation_intervals": ventilation,
    }
    out = args.work_dir / "comparison.json"
    out.write_text(json.dumps(clean_json(results), indent=2, allow_nan=False) + "\n")
    print(json.dumps(clean_json(summary), indent=2))
    print(f"Aggregate results: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "reference", "weavehr", "original", "adapted", "compare"])
    parser.add_argument("--work-dir", type=Path, default=REPO / "data/demo-comparison")
    parser.add_argument("--baseline-work-dir", type=Path,
                        help="Optional previous run for clinical-concept invariance audit")
    parser.add_argument("--original-repo", type=Path, default=REPO.parent / "clin-jepa-original")
    parser.add_argument("--mimic-code", type=Path, default=REPO.parent / "mimic-code")
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--weavehr-python", type=Path, default=REPO / ".venv-weavehr/bin/python")
    parser.add_argument("--duckdb-cli", type=Path)
    parser.add_argument(
        "--start-step",
        type=int,
        choices=range(1, 7),
        default=1,
        help="Resume original/adapted runs at this completed-data boundary",
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    args.duckdb_cli = args.duckdb_cli or args.work_dir / "bin/duckdb"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "reference":
        reference(args)
    elif args.stage == "weavehr":
        # WeavEHR's quickstart: two step configs, two steps. Clin-JEPA adds nothing to the project.
        config = args.work_dir / "weavehr-config"
        config.mkdir(exist_ok=True)
        dataset = {"name": "mimic-iv-demo", "version": "2.2"}
        raw = args.work_dir / "mimic-iv-clinical-database-demo-2.2"
        extraction = {"name": "Extraction", "version": "1.0.0", "config": {"data": [{**dataset, "path": str(raw)}]}}
        concept = {
            "name": "Concept",
            "version": "1.0.0",
            "config": {"extraction_step": "Extraction", "mapping_configs": [dataset]},
        }
        (config / "extraction.yml").write_text(yaml.safe_dump(extraction, sort_keys=False))
        (config / "concept.yml").write_text(yaml.safe_dump(concept, sort_keys=False))
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from weavehr import ConceptStep, ExtractionStep, WeavEHRProject\n"
            "config = Path(sys.argv[1])\n"
            "with WeavEHRProject(Path(sys.argv[2])) as project:\n"
            "    ExtractionStep.load(project, config / 'extraction.yml').run()\n"
            "    ConceptStep.load(project, config / 'concept.yml').run()\n"
        )
        command(
            args,
            [args.weavehr_python, "-c", code, config, args.work_dir / "weavehr-project"],
            name="weavehr-preparation",
        )
    elif args.stage in ("original", "adapted"):
        pipelines(args, args.stage)
    else:
        compare(args)


if __name__ == "__main__":
    main()
