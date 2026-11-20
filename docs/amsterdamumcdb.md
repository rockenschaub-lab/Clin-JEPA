# AmsterdamUMCdb with WeavEHR

AmsterdamUMCdb 1.5.0 now uses the **same WeavEHR-based steps and semantic profile as
MIMIC-IV**. The retired direct-OMOP mapper and its manual feature selectors are
no longer used. Follow [the shared preparation guide](data-preparation.md).

Build the WeavEHR project for `aumc` / `1.5.0` from the AMSTEL OMOP CDM 5.4
Parquet export as WeavEHR documents. Give this dataset its own WeavEHR project
and output directory, and run the pipeline with `--dataset aumc`.

The source selector retains ICU (`care_site_id=1`) and excludes medium care.
Native visit IDs become stay IDs; hospital IDs are unknown. Outcomes without an established definition/observation remain
missing. WeavEHR handles AMSTEL concept IDs, conversions and dose-string parsing.
See the shared guide for mapping coverage, interval semantics, SOFA/Sepsis
changes and downstream tasks that still require validation.
