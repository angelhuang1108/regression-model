# Duration Regression Model

`sklearn.ensemble.HistGradientBoostingRegressor` predicting trial duration (days) for **COMPLETED** trials only. Preprocessing (encoding, scaling) is unchanged from the prior Ridge pipeline; the booster captures non-linear effects and interactions in the feature space.

## Filtering

Data is filtered in two stages: preprocessing ([3_preprocessing/preprocess.py](3_preprocessing/preprocess.py)) and at training time ([4_regression/train_regression.py](4_regression/train_regression.py)).

### Preprocessing filters (applied to all studies)

1. **Exclude WITHDRAWN trials** — withdrawn trials never ran, so they have no meaningful duration.
2. **Study type = INTERVENTIONAL** — observational and expanded-access studies have different duration dynamics and are out of scope.
3. **Phase in {PHASE1, PHASE2, PHASE3, PHASE1/PHASE2, PHASE2/PHASE3}** — focuses on standard clinical development phases; early-phase (PHASE0) and non-phased trials are excluded.
4. **At least one industry sponsor** — the model targets industry-sponsored trials; academic/government-only trials follow different timelines.
5. **Both `start_date` and `primary_completion_date` must be non-null** — rows missing either date cannot produce a duration target.
6. **Dates in range 1980–2027** — removes clearly erroneous entries (far-future or pre-modern dates) that would distort the target distribution.
7. **Duration ≥ 0 days** — drops the small number of records where primary completion precedes the start date, which indicates a data entry error.

### Training-time filter

8. **`overall_status = COMPLETED`** — the model predicts actual trial duration, which is only observable for completed trials. Active, terminated, or suspended trials are excluded from training and evaluation.

## Target
- `duration_days` — time from start to primary completion

## Features (ablation-tested, best-performing subset)

### Core features (always included)
- `phase` — trial phase (one-hot)
- `enrollment` — planned enrollment
- `n_sponsors` — number of sponsors
- `number_of_arms` — number of arms
- `start_year` — trial start year
- `category` — therapeutic category (one-hot, 132 levels)
- `downcase_mesh_term` — MeSH condition terms (one-hot)
- `intervention_type` — intervention types (one-hot)

### Eligibility (kept from ablation)
- `gender`, `minimum_age`, `maximum_age`, `adult`, `child`, `older_adult`

### Site footprint (kept from ablation)
- `number_of_facilities`, `number_of_countries`, `us_only`, `has_single_facility`

### Design (kept from ablation)
- `randomized`, `intervention_model`, `masking_depth_score`, `primary_purpose`, `design_complexity_composite`

### Arm/intervention (kept from ablation)
- `number_of_interventions`, `intervention_type_diversity`, `mono_therapy`, `has_placebo`, `has_active_comparator`, `n_mesh_intervention_terms`

### Design outcomes (from design_outcomes table)
- `max_planned_followup_days` — max planned follow-up parsed from time_frame
- `n_primary_outcomes`, `n_secondary_outcomes`, `n_outcomes`
- `has_survival_endpoint`, `has_safety_endpoint` — flags from measure/description
- `endpoint_complexity_score` — composite of outcome count and endpoint types

## Metrics (test set)
- RMSE ≈ 551 days
- MAE ≈ 360 days
- R² ≈ 0.3743

## Train/val/test split
60% / 20% / 20%, random_state=42
