# Remaining clinical validation

The software path is shared and implemented in [the WeavEHR preparation
pipeline](data-preparation.md). Before reporting full-cohort transfer results:

- Validate coverage, unit distributions, stay assignment and representative
  trajectories on authorised MIMIC-IV and AMSTEL AUMC exports.
- Review specimen eligibility and ventilation/tracheostomy definitions across
  sources; compare WeavEHR SOFA, AKI and Sepsis-3 against appropriate references.
- Extend missing medication/procedure/line concepts in WeavEHR and review
  unclosed infusion or unpaired ventilation diagnostics.
- Establish observed negatives and follow-up/censoring for each binary endpoint
  before enabling the currently masked downstream tasks.
- Validate ICU mortality and hospital linkage for Amsterdam; do not equate an
  ICU/medium-care visit with a hospital admission or fill unavailable outcomes.
- Retrain and evaluate the model with the new inputs. Software smoke tests do
  not establish parity with the paper's mimic-code cohort or clinical results.
