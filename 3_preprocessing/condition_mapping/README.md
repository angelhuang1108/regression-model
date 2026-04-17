# Condition Mapping in Preprocessing

Condition mapping belongs to Stage 3 (preprocessing/feature engineering).

Current execution entrypoint:

- `python 3_preprocessing/run_condition_mapping.py`

This runner executes the condition-mapping pipeline in order:

1. `step00_exclusion_taxonomy.py`
2. `step01_normalize.py`
3. `step02_icd10_lookup.py`
4. `step03_ccsr_join.py`

Outputs are written under `3_preprocessing/condition_mapping/output/`.
