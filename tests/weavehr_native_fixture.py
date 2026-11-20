"""Synthetic native input writer and WeavEHR quickstart, executed in the optional WeavEHR environment.

Use the shipped table schemas to write complete, typed source files. The test
values themselves are independent expectations: HR=70+h, creatinine=1 mg/dL,
NE=0.1 mcg/kg/min during [1h,4h), with 2 stays per patient. MIMIC-IV stays also
have a vancomycin order during [1.5h,5h) that is given at 2h and held at 3h.
"""

import gzip
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml
from weavehr.steps.extraction.registry import dataset_config_registry
from weavehr.utils.loader import ensure_configs_loaded

# Raw tables this fixture writes. WeavEHR skips every other table of the dataset, as it
# does for any raw file that is absent.
TABLES = {
    "mimic-iv": ["patients", "admissions", "icustay", "chartevents", "labevents", "outputevents", "inputevents",
                "procedureevents", "prescriptions", "emar", "microbiologyevents"],
    "aumc": ["person", "visit_occurrence", "measurement", "observation", "drug_exposure",
                       "procedure_occurrence", "device_exposure", "death"],
}


def names(dataset):
    return ("aumc", "1.5.0") if dataset == "aumc" else ("mimic-iv", "3.1")


def build_project(raw, dataset, project):
    """Build the project as WeavEHR's quickstart does: two step configs, two steps, nothing else."""
    from weavehr import ConceptStep, ExtractionStep, WeavEHRProject

    name, version = names(dataset)
    config = Path(project).parent / "weavehr-config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "extraction.yml").write_text(yaml.safe_dump({
        "name": "Extraction", "version": "1.0.0",
        "config": {"data": [{"name": name, "version": version, "path": str(raw)}]},
    }))
    (config / "concept.yml").write_text(yaml.safe_dump({
        "name": "Concept", "version": "1.0.0",
        "config": {"extraction_step": "Extraction", "mapping_configs": [{"name": name, "version": version}]},
    }))
    with WeavEHRProject(Path(project)) as weavehr_project:
        ExtractionStep.load(weavehr_project, config / "extraction.yml").run()
        ConceptStep.load(weavehr_project, config / "concept.yml").run()


def write_raw(root, dataset, n=24):
    ensure_configs_loaded()
    name, version = names(dataset)
    tables = [t for t in dataset_config_registry.filter(name, version) if t.name in TABLES[dataset]]
    schemas = {}
    for table in tables:
        for part in [table, *table.join]:
            schema = schemas.setdefault(part.path, {})
            schema.update({c.name: c.dtype for c in part.columns})
    rows = {p: [] for p in schemas}

    def add(path, **row):
        if path not in rows:
            path = path.removesuffix(".parquet") + "/*.parquet"
        rows[path].append(row)

    origin = datetime(2013, 1, 1)

    def stamp(x):
        return x.strftime("%Y-%m-%d %H:%M:%S")

    if dataset == "mimic-iv":
        for itemid, label in [(220045, "Heart Rate"), (221906, "Norepinephrine"), (225792, "Invasive Ventilation")]:
            add("icu/d_items.csv.gz", itemid=itemid, label=label)
        add("hosp/d_labitems.csv.gz", itemid=50912, label="Creatinine", fluid="Blood", category="Chemistry")
        for patient in range(1, n // 2 + 1):
            add(
                "hosp/patients.csv.gz",
                subject_id=patient,
                anchor_age=45,
                anchor_year=2013,
                anchor_year_group="2011 - 2013",
                gender="M",
                # Every fourth stay ends in death, half an hour before ICU discharge.
                dod=next(
                    (
                        (origin + timedelta(days=stay)).strftime("%Y-%m-%d")
                        for stay in (2 * patient - 1, 2 * patient)
                        if stay % 4 == 0
                    ),
                    None,
                ),
            )
    else:
        for patient in range(1, n // 2 + 1):
            add(
                "person.parquet",
                person_id=patient,
                year_of_birth=1968,
                gender_concept_id=8507,
                gender_source_value="Man",
            )
    for stay in range(1, n + 1):
        subject = (stay + 1) // 2
        t = origin + timedelta(days=stay)
        end = t + timedelta(hours=8)
        if dataset == "mimic-iv":
            add(
                "icu/icustays.csv.gz",
                subject_id=subject,
                hadm_id=stay + 100,
                stay_id=stay,
                intime=stamp(t),
                outtime=stamp(end),
                los=8 / 24,
            )
            add(
                "hosp/admissions.csv.gz",
                subject_id=subject,
                hadm_id=stay + 100,
                admittime=stamp(t),
                dischtime=stamp(end),
                deathtime=stamp(end - timedelta(minutes=30)) if stay % 4 == 0 else None,
                discharge_location="DIED" if stay % 4 == 0 else "HOME",
                race="UNKNOWN",
                hospital_expire_flag=int(stay % 4 == 0),
            )
            for h in range(8):
                add(
                    "icu/chartevents.csv.gz",
                    subject_id=subject,
                    hadm_id=stay + 100,
                    stay_id=stay,
                    itemid=220045,
                    charttime=stamp(t + timedelta(hours=h, minutes=10)),
                    storetime=stamp(t + timedelta(hours=h, minutes=10)),
                    valuenum=70 + h,
                    value=str(70 + h),
                    valueuom="bpm",
                )
            add(
                "hosp/labevents.csv.gz",
                subject_id=subject,
                hadm_id=stay + 100,
                itemid=50912,
                charttime=stamp(t + timedelta(hours=2)),
                storetime=stamp(t + timedelta(hours=2)),
                valuenum=1.0,
                value="1",
                valueuom="mg/dL",
            )
            add(
                "icu/inputevents.csv.gz",
                subject_id=subject,
                hadm_id=stay + 100,
                stay_id=stay,
                itemid=221906,
                starttime=stamp(t + timedelta(hours=1)),
                endtime=stamp(t + timedelta(hours=4)),
                storetime=stamp(t + timedelta(hours=1)),
                originalrate=0.1,
                rate=0.1,
                rateuom="mcg/kg/min",
                amount=1.44,
                amountuom="mg",
                ordercategorydescription="Continuous Med",
                statusdescription="FinishedRunning",
                linkorderid=stay,
                orderid=stay,
                patientweight=80.0,
            )
            add(
                "icu/procedureevents.csv.gz",
                subject_id=subject,
                hadm_id=stay + 100,
                stay_id=stay,
                itemid=225792,
                orderid=stay,
                linkorderid=stay,
                starttime=stamp(t + timedelta(minutes=30)),
                endtime=stamp(t + timedelta(hours=2, minutes=30)),
                storetime=stamp(t + timedelta(minutes=30)),
                value=120.0,
                valueuom="min",
                statusdescription="FinishedRunning",
            )
            add("hosp/poe.csv.gz", poe_id=f"{subject}-{stay}", poe_seq=stay, ordertime=stamp(t + timedelta(hours=1)))
            add(
                "hosp/prescriptions.csv.gz",
                subject_id=subject,
                hadm_id=stay + 100,
                pharmacy_id=stay,
                poe_id=f"{subject}-{stay}",
                poe_seq=stay,
                starttime=stamp(t + timedelta(hours=1, minutes=30)),
                stoptime=stamp(t + timedelta(hours=5)),
                drug="Vancomycin",
                drug_type="MAIN",
                prod_strength="1g Frozen Bag",
                dose_val_rx="1000",
                dose_unit_rx="mg",
                route="IV",
            )
            for hour, event in ((2, "Administered"), (3, "Not Given")):
                add(
                    "hosp/emar.csv.gz",
                    subject_id=subject,
                    hadm_id=stay + 100,
                    emar_id=f"{subject}-{stay * 10 + hour}",
                    emar_seq=stay * 10 + hour,
                    charttime=stamp(t + timedelta(hours=hour)),
                    storetime=stamp(t + timedelta(hours=hour)),
                    event_txt=event,
                    medication="Vancomycin",
                )
        else:
            add(
                "visit_occurrence.parquet",
                person_id=subject,
                visit_occurrence_id=stay,
                visit_start_datetime=stamp(t),
                visit_end_datetime=stamp(end),
                visit_concept_id=32037,
                visit_source_value="ICU",
                care_site_id=1,
            )
            for h in range(8):
                add(
                    "measurement.parquet",
                    measurement_id=100 * stay + h,
                    person_id=subject,
                    visit_occurrence_id=stay,
                    measurement_datetime=stamp(t + timedelta(hours=h, minutes=10)),
                    measurement_concept_id=3027018,
                    measurement_source_value="Hartfrequentie",
                    value_as_number=70 + h,
                    unit_source_value="bpm",
                )
            add(
                "measurement.parquet",
                measurement_id=100 * stay + 50,
                person_id=subject,
                visit_occurrence_id=stay,
                measurement_datetime=stamp(t + timedelta(hours=2)),
                measurement_concept_id=3020564,
                measurement_source_value="Kreatinine",
                value_as_number=88.4,
                unit_source_value="umol/L",
            )
            add(
                "drug_exposure.parquet",
                drug_exposure_id=stay,
                person_id=subject,
                visit_occurrence_id=stay,
                drug_concept_id=2907531,
                drug_source_value="Noradrenaline",
                drug_exposure_start_datetime=stamp(t + timedelta(hours=1)),
                drug_exposure_end_datetime=stamp(t + timedelta(hours=4)),
                sig="4 mg @ 0.1 µg/kg/min = 6 ml/uur",
            )
            add(
                "procedure_occurrence.parquet",
                person_id=subject,
                visit_occurrence_id=stay,
                procedure_occurrence_id=stay + 1000,
                procedure_concept_id=4230167,
                procedure_source_value="Beademen",
                procedure_datetime=stamp(t + timedelta(minutes=30)),
                procedure_end_datetime=stamp(t + timedelta(hours=2, minutes=30)),
            )
    for path, schema in schemas.items():
        # CSV and OMOP timestamps use the strings expected by shipped parsers.
        schema = {k: pl.String if isinstance(v, pl.Datetime) else v for k, v in schema.items()}
        data = [{k: row.get(k) for k in schema} for row in rows[path]]
        frame = pl.DataFrame(data, schema=schema)
        target = root / path.replace("*", "part-000")
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".csv.gz"):
            with gzip.open(target, "wt") as stream:
                stream.write(frame.write_csv())
        else:
            frame.write_parquet(target)


if __name__ == "__main__":
    write_raw(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    if len(sys.argv) > 4:
        build_project(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[4]))
