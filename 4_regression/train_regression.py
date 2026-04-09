"""
Join studies with sponsors, select features, train/val/test split,
and run regression to predict duration_days.

Target is log1p-transformed inside TransformedTargetRegressor; predictions are
inverted to days for evaluation. Restricted to COMPLETED trials only.

Trains HistGradientBoostingRegressor models on COMPLETED trials:
  - Dedicated: PHASE1, PHASE2, PHASE3 (each fit on that phase only).
  - Early joint: PHASE1 + PHASE1/PHASE2 + PHASE2; used to score PHASE1/PHASE2 trials.
  - Late joint: PHASE2 + PHASE2/PHASE3 + PHASE3; used to score PHASE2/PHASE3 trials.
No StandardScaler; numeric NaNs are kept for HGBR. No phase one-hot inside each cohort.
"""
import re
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
_PREPROC = PROJECT_ROOT / "3_preprocessing"
if str(_PREPROC) not in sys.path:
    sys.path.insert(0, str(_PREPROC))
from eligibility_criteria_features import CRITERIA_TEXT_FEATURE_COLUMNS

CLEAN_DATA = PROJECT_ROOT / "clean_data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Features: target_duration (97% null) and number_of_groups (100% null) excluded
# category (132 unique) outperforms therapeutic_area (16 unique): R² 0.317 vs 0.285
# Phase: one-hot (R² 0.317) > phase flags (0.315) > no phase (0.280)
# downcase_mesh_term: ablation R² 0.319 vs 0.317 baseline — small gain, included
# intervention_type: ablation R² 0.320 vs 0.319 baseline — included
# eligibility: gender, minimum_age, maximum_age, adult, child, older_adult (ablation-tested)
FEATURE_COLUMNS = [
    "phase",
    "enrollment",
    "n_sponsors",
    "number_of_arms",
    "start_year",
    "category",
    "downcase_mesh_term",
    "intervention_type",
]
ELIGIBILITY_COLUMNS = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]
SITE_FOOTPRINT_FEATURES = [
    "number_of_facilities",
    "number_of_countries",
    "us_only",
    "number_of_us_states",
    "has_single_facility",
    "facility_density",
]
DESIGN_FEATURES = [
    "randomized",
    "intervention_model",
    "masking_depth_score",
    "primary_purpose",
    "design_complexity_composite",
]
ARM_INTERVENTION_FEATURES = [
    "number_of_interventions",
    "intervention_type_diversity",
    "mono_therapy",
    "has_placebo",
    "has_active_comparator",
    "n_mesh_intervention_terms",
]
TARGET_COLUMN = "duration_days"

# Best-performing columns from ablation studies (see MODEL.md)
KEPT_ELIGIBILITY = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]
KEPT_ELIGIBILITY_CRITERIA_TEXT = list(CRITERIA_TEXT_FEATURE_COLUMNS)
KEPT_SITE_FOOTPRINT = ["number_of_facilities", "number_of_countries", "us_only", "has_single_facility"]
KEPT_DESIGN = ["randomized", "intervention_model", "masking_depth_score", "primary_purpose", "design_complexity_composite"]
KEPT_ARM_INTERVENTION = [
    "number_of_interventions",
    "intervention_type_diversity",
    "mono_therapy",
    "has_placebo",
    "has_active_comparator",
    "n_mesh_intervention_terms",
]
DESIGN_OUTCOMES_FEATURES = [
    "max_planned_followup_days",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "n_outcomes",
    "has_survival_endpoint",
    "has_safety_endpoint",
    "endpoint_complexity_score",
]
KEPT_DESIGN_OUTCOMES = [
    "max_planned_followup_days",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "n_outcomes",
    "has_survival_endpoint",
    "has_safety_endpoint",
    "endpoint_complexity_score",
]

RAW_DATA = PROJECT_ROOT / "raw_data"

# Single-phase models (enough samples each); mixed labels use joint pools below
PHASE_SINGLE_MODELS = ("PHASE1", "PHASE2", "PHASE3")

EARLY_JOINT_PHASES = frozenset({"PHASE1", "PHASE1/PHASE2", "PHASE2"})
LATE_JOINT_PHASES = frozenset({"PHASE2", "PHASE2/PHASE3", "PHASE3"})

# Row order for summary table / reporting
PHASE_REPORT_ORDER = (
    "PHASE1",
    "PHASE1/PHASE2",
    "PHASE2",
    "PHASE2/PHASE3",
    "PHASE3",
)


def _parse_time_frame_days(tf: str) -> float | None:
    """Parse time_frame string to days. Returns None if unparseable."""
    if pd.isna(tf) or not isinstance(tf, str) or not tf.strip():
        return None
    tf = tf.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?", tf)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "day":
        return val
    if unit == "week":
        return val * 7
    if unit == "month":
        return val * 30.44
    if unit == "year":
        return val * 365.25
    return None


def _has_endpoint_keywords(text: str, keywords: list[str]) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def load_and_join(
    eligibility_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load clean studies, sponsors, and categorized_output; join on nct_id.
    If eligibility_columns is provided, join eligibilities table for those columns.
    """
    studies = pd.read_csv(CLEAN_DATA / "studies.csv", low_memory=False)
    sponsors = pd.read_csv(CLEAN_DATA / "sponsors.csv", low_memory=False)

    # Restrict to COMPLETED trials only (actual duration)
    studies = studies[studies["overall_status"] == "COMPLETED"].copy()

    # Aggregate sponsors: count per nct_id
    sponsor_counts = sponsors.groupby("nct_id").size().reset_index(name="n_sponsors")
    df = studies.merge(sponsor_counts, on="nct_id", how="left")
    df["n_sponsors"] = df["n_sponsors"].fillna(0).astype(int)

    # Join category from categorized_output (take highest-confidence per trial)
    categorized = pd.read_csv(RAW_DATA / "categorized_output.csv", low_memory=False)
    cat_agg = (
        categorized.sort_values("confidence", ascending=False)
        .groupby("nct_id")[["category"]]
        .first()
        .reset_index()
    )
    df = df.merge(cat_agg, on="nct_id", how="left")
    df["category"] = df["category"].fillna("Other_Unclassified")

    # Join downcase_mesh_term from browse_conditions (first per trial)
    bc_path = RAW_DATA / "browse_conditions.csv"
    if bc_path.exists():
        bc = pd.read_csv(bc_path, low_memory=False)
        mesh_col = "downcase_mesh_term" if "downcase_mesh_term" in bc.columns else "mesh_term"
        if mesh_col in bc.columns:
            mesh_agg = bc.groupby("nct_id")[mesh_col].first().reset_index()
            mesh_agg.columns = ["nct_id", "downcase_mesh_term"]
            df = df.merge(mesh_agg, on="nct_id", how="left")
            df["downcase_mesh_term"] = df["downcase_mesh_term"].fillna("unknown")

    # Join intervention_type from interventions (mode per trial)
    int_path = RAW_DATA / "interventions.csv"
    if int_path.exists():
        interventions = pd.read_csv(int_path, low_memory=False)
        if "intervention_type" in interventions.columns:
            int_agg = (
                interventions.groupby("nct_id")["intervention_type"]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                .reset_index()
            )
            df = df.merge(int_agg, on="nct_id", how="left")
            df["intervention_type"] = df["intervention_type"].fillna("UNKNOWN")

    # Join eligibilities (first row per nct_id)
    elig_path = RAW_DATA / "eligibilities.csv"
    if elig_path.exists() and eligibility_columns:
        elig = pd.read_csv(elig_path, low_memory=False)
        cols_to_join = ["nct_id"] + [c for c in eligibility_columns if c in elig.columns]
        if len(cols_to_join) > 1:
            elig_agg = elig[cols_to_join].groupby("nct_id").first().reset_index()
            df = df.merge(elig_agg, on="nct_id", how="left")

    # Join site footprint (calculated_values, facilities, countries)
    if site_footprint_columns:
        cv_path = RAW_DATA / "calculated_values.csv"
        if cv_path.exists():
            cv = pd.read_csv(cv_path, low_memory=False)
            cv_cols = ["nct_id", "number_of_facilities", "has_us_facility", "has_single_facility"]
            cv_cols = [c for c in cv_cols if c in cv.columns]
            cv_agg = cv[cv_cols].groupby("nct_id").first().reset_index()
            df = df.merge(cv_agg, on="nct_id", how="left")

        countries_path = RAW_DATA / "countries.csv"
        if countries_path.exists():
            countries = pd.read_csv(countries_path, low_memory=False)
            # Exclude removed countries (removed=True means no longer associated)
            if "removed" in countries.columns:
                countries_active = countries[~countries["removed"].fillna(False).astype(bool)]
            else:
                countries_active = countries
            n_countries = countries_active.groupby("nct_id").size().reset_index(name="number_of_countries")
            df = df.merge(n_countries, on="nct_id", how="left")
            # US-only: 1 if exactly 1 country and it's US
            if "name" in countries.columns:
                us_only = (
                    countries_active.groupby("nct_id")["name"]
                    .apply(lambda x: 1 if (len(x) == 1 and "united states" in str(x.iloc[0]).lower()) else 0)
                    .reset_index(name="us_only")
                )
                df = df.merge(us_only, on="nct_id", how="left")

        fac_path = RAW_DATA / "facilities.csv"
        if fac_path.exists() and "number_of_us_states" in site_footprint_columns:
            fac = pd.read_csv(fac_path, low_memory=False)
            us_fac = fac[fac["country"].str.upper().str.contains("UNITED STATES", na=False)]
            n_us_states = us_fac.groupby("nct_id")["state"].nunique().reset_index(name="number_of_us_states")
            df = df.merge(n_us_states, on="nct_id", how="left")

        # Derived: facility_density = number_of_facilities / enrollment
        if "facility_density" in site_footprint_columns and "number_of_facilities" in df.columns and "enrollment" in df.columns:
            enroll = pd.to_numeric(df["enrollment"], errors="coerce").fillna(1)
            df["facility_density"] = df["number_of_facilities"].fillna(0) / enroll.replace(0, 1)

    # Join designs (one row per nct_id)
    if design_columns:
        designs_path = RAW_DATA / "designs.csv"
        if designs_path.exists():
            designs = pd.read_csv(designs_path, low_memory=False)
            design_cols = ["nct_id", "allocation", "intervention_model", "primary_purpose", "masking",
                          "subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]
            design_cols = [c for c in design_cols if c in designs.columns]
            design_agg = designs[design_cols].groupby("nct_id").first().reset_index()
            df = df.merge(design_agg, on="nct_id", how="left")

            # Derived: randomized (1 if RANDOMIZED)
            if "randomized" in design_columns and "allocation" in df.columns:
                df["randomized"] = (df["allocation"].str.upper() == "RANDOMIZED").astype(int)

            # Derived: masking_depth_score (NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3, QUADRUPLE=4)
            if "masking_depth_score" in design_columns and "masking" in df.columns:
                mask_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
                df["masking_depth_score"] = df["masking"].str.upper().map(mask_map).fillna(0)
                # Add role flags: +0.25 per masked role (max 1 extra)
                for role in ["subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]:
                    if role in df.columns:
                        df["masking_depth_score"] += df[role].apply(
                            lambda x: 0.25 if x in (True, "true", "True", 1) else 0
                        )

            # Derived: design_complexity_composite (randomized + multi-arm + normalized masking)
            if "design_complexity_composite" in design_columns:
                r = (df["allocation"].str.upper() == "RANDOMIZED").astype(int) if "allocation" in df.columns else 0
                m = df["masking_depth_score"].fillna(0) if "masking_depth_score" in df.columns else 0
                if "masking_depth_score" not in df.columns and "masking" in df.columns:
                    mask_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
                    m = df["masking"].str.upper().map(mask_map).fillna(0)
                arms = pd.to_numeric(df["number_of_arms"], errors="coerce").fillna(1)
                multi = (arms > 1).astype(int)
                df["design_complexity_composite"] = r + multi + (m / 5)

    # Join arm/intervention complexity (interventions, design_groups, browse_interventions)
    if arm_intervention_columns:
        int_path = RAW_DATA / "interventions.csv"
        if int_path.exists():
            interventions = pd.read_csv(int_path, low_memory=False)
            if "intervention_type" in interventions.columns:
                n_int = interventions.groupby("nct_id").size().reset_index(name="number_of_interventions")
                df = df.merge(n_int, on="nct_id", how="left")
                n_types = interventions.groupby("nct_id")["intervention_type"].nunique().reset_index(name="intervention_type_diversity")
                df = df.merge(n_types, on="nct_id", how="left")
                df["mono_therapy"] = (df["intervention_type_diversity"].fillna(0) == 1).astype(int)

        dg_path = RAW_DATA / "design_groups.csv"
        if dg_path.exists():
            dg = pd.read_csv(dg_path, low_memory=False)
            if "group_type" in dg.columns:
                dg["_gt"] = dg["group_type"].fillna("").astype(str).str.upper()
                dg["_title"] = dg.get("title", pd.Series([""] * len(dg))).fillna("").astype(str).str.upper()
                dg["_combined"] = dg["_gt"] + " " + dg["_title"]
                has_placebo = dg.groupby("nct_id")["_combined"].apply(lambda x: 1 if x.str.contains("PLACEBO", na=False).any() else 0).reset_index(name="has_placebo")
                df = df.merge(has_placebo, on="nct_id", how="left")
                has_ac = dg.groupby("nct_id")["_combined"].apply(lambda x: 1 if x.str.contains("ACTIVE.COMPARATOR|ACTIVE_COMPARATOR|COMPARATOR", na=False, regex=True).any() else 0).reset_index(name="has_active_comparator")
                df = df.merge(has_ac, on="nct_id", how="left")

        bi_path = RAW_DATA / "browse_interventions.csv"
        if bi_path.exists():
            bi = pd.read_csv(bi_path, low_memory=False)
            mesh_col = "downcase_mesh_term" if "downcase_mesh_term" in bi.columns else "mesh_term"
            if mesh_col in bi.columns:
                n_mesh = bi.groupby("nct_id")[mesh_col].nunique().reset_index(name="n_mesh_intervention_terms")
                df = df.merge(n_mesh, on="nct_id", how="left")

    # Join design_outcomes (per-trial aggregates)
    if design_outcomes_columns:
        do_path = RAW_DATA / "design_outcomes.csv"
        if do_path.exists():
            nct_ids = set(df["nct_id"].unique())
            usecols = ["nct_id", "outcome_type", "measure", "time_frame"]
            if "description" in pd.read_csv(do_path, nrows=0).columns:
                usecols.append("description")
            chunks = []
            for chunk in pd.read_csv(do_path, chunksize=200_000, low_memory=False, usecols=usecols):
                chunk = chunk[chunk["nct_id"].isin(nct_ids)]
                if len(chunk) > 0:
                    chunks.append(chunk)
            do = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols + ["_tf_days"])
            if len(do) == 0:
                do = None
            else:
                do["_tf_days"] = do["time_frame"].apply(_parse_time_frame_days)
                SURVIVAL_KW = ["survival", "os", "pfs", "dfs", "overall survival", "progression-free survival"]
                SAFETY_KW = ["safety", "adverse", "ae", "sae", "toxicity", "tolerability"]
                meas = do["measure"] if "measure" in do.columns else pd.Series([""] * len(do))
                desc = do["description"] if "description" in do.columns else pd.Series([""] * len(do))
                do["_has_survival"] = meas.apply(lambda x: _has_endpoint_keywords(x, SURVIVAL_KW)) | desc.apply(lambda x: _has_endpoint_keywords(x, SURVIVAL_KW))
                do["_has_safety"] = meas.apply(lambda x: _has_endpoint_keywords(x, SAFETY_KW)) | desc.apply(lambda x: _has_endpoint_keywords(x, SAFETY_KW))
                n_outcomes = do.groupby("nct_id").size().reset_index(name="n_outcomes")
                max_tf = do.groupby("nct_id")["_tf_days"].max().reset_index(name="max_planned_followup_days")
                agg = n_outcomes.merge(max_tf, on="nct_id", how="left")
                if "outcome_type" in do.columns:
                    n_prim = do.groupby("nct_id")["outcome_type"].apply(lambda x: (x.fillna("").str.upper() == "PRIMARY").sum()).reset_index(name="n_primary_outcomes")
                    n_sec = do.groupby("nct_id")["outcome_type"].apply(lambda x: (x.fillna("").str.upper() == "SECONDARY").sum()).reset_index(name="n_secondary_outcomes")
                    agg = agg.merge(n_prim, on="nct_id", how="left").merge(n_sec, on="nct_id", how="left")
                else:
                    agg["n_primary_outcomes"] = 0
                    agg["n_secondary_outcomes"] = 0
                has_surv = do.groupby("nct_id")["_has_survival"].max().reset_index(name="has_survival_endpoint")
                has_safe = do.groupby("nct_id")["_has_safety"].max().reset_index(name="has_safety_endpoint")
                agg = agg.merge(has_surv, on="nct_id", how="left").merge(has_safe, on="nct_id", how="left")
                agg["endpoint_complexity_score"] = (
                    agg["n_outcomes"].fillna(0) * 0.5
                    + agg["n_primary_outcomes"].fillna(0) * 0.3
                    + agg["n_secondary_outcomes"].fillna(0) * 0.2
                    + agg["has_survival_endpoint"].fillna(0) * 2
                    + agg["has_safety_endpoint"].fillna(0) * 1
                )
                df = df.merge(agg, on="nct_id", how="left")

    return df


def prepare_features(
    df: pd.DataFrame,
    eligibility_columns: list[str] | None = None,
    eligibility_criteria_text_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
    *,
    encode_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build X (float matrix with NaN allowed for HGBR), y, phase labels, and encoders.
    When encode_phase is False (per-phase models), phase one-hot is omitted — phase is constant.
    No median imputation for enrollment / number_of_arms / start_year; no scaling.
    """
    df = df.copy()

    # start_year as float; NaN if start_date missing
    if "start_date" in df.columns:
        df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year
    else:
        df["start_year"] = np.nan
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")

    # Core numerics: keep NaN (missingness is a signal for trees)
    for col in ["enrollment", "number_of_arms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values.astype(np.float64)

    # Optional phase one-hot (unused for dedicated single-phase models)
    if encode_phase:
        phase_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        phase_encoded = phase_encoder.fit_transform(df[["phase"]])
        phase_blocks: list[np.ndarray] = [phase_encoded]
    else:
        phase_encoder = None
        phase_blocks = []

    # Encode category (one-hot)
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = cat_encoder.fit_transform(df[["category"]])

    # Encode downcase_mesh_term (one-hot, top 50 to limit features)
    mesh_parts = []
    mesh_encoder = None
    if "downcase_mesh_term" in df.columns:
        top_mesh = df["downcase_mesh_term"].value_counts().head(50).index.tolist()
        df["mesh_trimmed"] = df["downcase_mesh_term"].where(
            df["downcase_mesh_term"].isin(top_mesh), "other"
        )
        mesh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        mesh_parts = [mesh_encoder.fit_transform(df[["mesh_trimmed"]])]

    # Encode intervention_type (one-hot, top 15 by count — same pattern as mesh_term)
    int_parts = []
    int_encoder = None
    if "intervention_type" in df.columns:
        top_int = df["intervention_type"].value_counts().head(15).index.tolist()
        df["intervention_trimmed"] = df["intervention_type"].where(
            df["intervention_type"].isin(top_int), "other"
        )
        int_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        int_parts = [int_encoder.fit_transform(df[["intervention_trimmed"]])]

    # Eligibility features
    elig_parts = []
    elig_encoders = {}
    elig_feature_names = []
    if eligibility_columns:
        for col in eligibility_columns:
            if col not in df.columns:
                continue
            if col == "gender":
                df["gender_fill"] = df["gender"].fillna("ALL").astype(str)
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                elig_parts.append(enc.fit_transform(df[["gender_fill"]]))
                elig_encoders["gender"] = enc
                elig_feature_names.extend(enc.get_feature_names_out(["gender_fill"]))
            elif col in ("minimum_age", "maximum_age"):
                raw = df[col].astype(str).str.extract(r"(\d+)", expand=False)
                vals = pd.to_numeric(raw, errors="coerce")
                elig_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                elig_feature_names.append(col)
            elif col in ("adult", "child", "older_adult"):
                s = df[col]

                def _tri_state(v: object) -> float:
                    if pd.isna(v):
                        return np.nan
                    return 1.0 if v in (True, "true", "True", "YES", "Yes", 1) else 0.0

                elig_parts.append(np.column_stack([s.map(_tri_state).astype(np.float64).values]))
                elig_feature_names.append(col)

    # Eligibility criteria text (numeric; from clean_data/studies via preprocess)
    criteria_parts: list[np.ndarray] = []
    criteria_text_feature_names: list[str] = []
    if eligibility_criteria_text_columns:
        for col in eligibility_criteria_text_columns:
            if col not in df.columns:
                vals = np.zeros(len(df), dtype=float)
            else:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            criteria_parts.append(np.column_stack([vals]))
            criteria_text_feature_names.append(col)

    # Site footprint features
    site_parts = []
    site_feature_names = []
    if site_footprint_columns:
        for col in site_footprint_columns:
            if col not in df.columns:
                continue
            if col == "has_single_facility":

                def _sf(x: object) -> float:
                    if pd.isna(x):
                        return np.nan
                    return 1.0 if x in (True, "true", "True", "YES", "Yes", 1) else 0.0

                vals = df[col].map(_sf).astype(np.float64)
                site_parts.append(np.column_stack([vals.values]))
                site_feature_names.append(col)
            elif col in ("number_of_facilities", "number_of_countries", "number_of_us_states"):
                vals = pd.to_numeric(df[col], errors="coerce")
                site_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                site_feature_names.append(col)
            elif col == "us_only":
                vals = pd.to_numeric(df[col], errors="coerce")
                site_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                site_feature_names.append(col)
            elif col == "facility_density":
                vals = pd.to_numeric(df[col], errors="coerce")
                site_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                site_feature_names.append(col)

    # Design features
    design_parts = []
    design_feature_names = []
    if design_columns:
        for col in design_columns:
            if col not in df.columns:
                continue
            if col == "randomized":
                vals = pd.to_numeric(df[col], errors="coerce")
                design_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                design_feature_names.append(col)
            elif col == "intervention_model":
                df["intervention_model_fill"] = df["intervention_model"].fillna("UNKNOWN").astype(str)
                top_mod = df["intervention_model_fill"].value_counts().head(6).index.tolist()
                df["intervention_model_trimmed"] = df["intervention_model_fill"].where(
                    df["intervention_model_fill"].isin(top_mod), "other"
                )
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                design_parts.append(enc.fit_transform(df[["intervention_model_trimmed"]]))
                design_feature_names.extend(enc.get_feature_names_out(["intervention_model_trimmed"]))
            elif col == "primary_purpose":
                df["primary_purpose_fill"] = df["primary_purpose"].fillna("OTHER").astype(str)
                top_pp = df["primary_purpose_fill"].value_counts().head(6).index.tolist()
                df["primary_purpose_trimmed"] = df["primary_purpose_fill"].where(
                    df["primary_purpose_fill"].isin(top_pp), "other"
                )
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                design_parts.append(enc.fit_transform(df[["primary_purpose_trimmed"]]))
                design_feature_names.extend(enc.get_feature_names_out(["primary_purpose_trimmed"]))
            elif col in ("masking_depth_score", "design_complexity_composite"):
                vals = pd.to_numeric(df[col], errors="coerce")
                design_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                design_feature_names.append(col)

    # Design outcomes features
    do_parts = []
    do_feature_names = []
    if design_outcomes_columns:
        for col in design_outcomes_columns:
            if col not in df.columns:
                continue
            if col in ("has_survival_endpoint", "has_safety_endpoint"):

                def _ep(x: object) -> float:
                    if pd.isna(x):
                        return np.nan
                    return 1.0 if x in (True, "true", "True", 1) else 0.0

                vals = df[col].map(_ep).astype(np.float64)
                do_parts.append(np.column_stack([vals.values]))
                do_feature_names.append(col)
            else:
                vals = pd.to_numeric(df[col], errors="coerce")
                do_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
                do_feature_names.append(col)

    # Arm/intervention features
    arm_parts = []
    arm_feature_names = []
    if arm_intervention_columns:
        for col in arm_intervention_columns:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            arm_parts.append(np.column_stack([vals.to_numpy(dtype=np.float64, na_value=np.nan)]))
            arm_feature_names.append(col)

    # Numeric features (n_sponsors: count, 0 = none after merge — keep as-is)
    enroll = pd.to_numeric(df["enrollment"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    n_spon = pd.to_numeric(df["n_sponsors"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    n_arms = pd.to_numeric(df["number_of_arms"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    sy = pd.to_numeric(df["start_year"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    X_numeric = np.column_stack([enroll, n_spon, n_arms, sy])

    X = np.hstack(
        phase_blocks
        + [cat_encoded]
        + mesh_parts
        + int_parts
        + elig_parts
        + criteria_parts
        + site_parts
        + design_parts
        + do_parts
        + arm_parts
        + [X_numeric]
    )
    X = np.asarray(X, dtype=np.float64)

    phases = df["phase"].astype(str).values

    artifacts = {
        "phase_encoder": phase_encoder,
        "cat_encoder": cat_encoder,
        "mesh_encoder": mesh_encoder,
        "int_encoder": int_encoder,
        "elig_encoders": elig_encoders,
        "elig_feature_names": elig_feature_names,
        "criteria_text_feature_names": criteria_text_feature_names,
        "site_feature_names": site_feature_names,
        "design_feature_names": design_feature_names,
        "do_feature_names": do_feature_names,
        "arm_feature_names": arm_feature_names,
    }
    return X, y, phases, artifacts


def _eval_split(name: str, model: TransformedTargetRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(X)
    return {
        "set": name,
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


def _joint_test_metrics_by_phase(
    model: TransformedTargetRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ph_test: np.ndarray,
    phase_labels: tuple[str, ...],
) -> dict[str, dict]:
    """Test RMSE/MAE/R² per phase label on the joint model's test fold (need n>=2 for R²)."""
    out: dict[str, dict] = {}
    for ph in phase_labels:
        mask = ph_test == ph
        y_s = y_test[mask]
        X_s = X_test[mask]
        if len(y_s) < 2:
            continue
        ev = _eval_split("test", model, X_s, y_s)
        out[ph] = {
            "n": int(len(y_s)),
            "r2": ev["r2"],
            "rmse": ev["rmse"],
            "mae": ev["mae"],
        }
    return out


def _new_regressor() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(max_iter=200, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def _train_val_test_split(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _train_val_test_split_with_phase(
    X: np.ndarray,
    y: np.ndarray,
    phases: np.ndarray,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_train, X_temp, y_train, y_temp, ph_train, ph_temp = train_test_split(
        X, y, phases, test_size=0.4, random_state=random_state
    )
    X_val, X_test, y_val, y_test, ph_val, ph_test = train_test_split(
        X_temp, y_temp, ph_temp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, ph_train, ph_val, ph_test


def main() -> None:
    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )

    completed = df[df["overall_status"] == "COMPLETED"].copy()
    completed["phase"] = completed["phase"].astype(str)
    phase_counts = completed["phase"].value_counts()

    lines: list[str] = []
    lines.append(
        "HistGradientBoostingRegressor: dedicated models for PHASE1, PHASE2, PHASE3; "
        "early joint (PHASE1+PHASE1/PHASE2+PHASE2) for PHASE1/PHASE2 rows; "
        "late joint (PHASE2+PHASE2/PHASE3+PHASE3) for PHASE2/PHASE3 rows."
    )
    lines.append("No StandardScaler; numeric NaNs preserved for HGBR. No phase one-hot inside each cohort.")
    lines.append("")
    lines.append("Trial counts in loaded cohort (COMPLETED, by phase label):")
    for ph in PHASE_REPORT_ORDER:
        n = int(phase_counts.get(ph, 0))
        lines.append(f"  {ph}: n={n:,}")
    lines.append("")

    prep_kw = dict(
        eligibility_columns=KEPT_ELIGIBILITY,
        eligibility_criteria_text_columns=KEPT_ELIGIBILITY_CRITERIA_TEXT,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
        encode_phase=False,
    )

    # phase_label -> (test_n, test_r2, model_description)
    summary: dict[str, tuple[int, float, str]] = {}
    early_by_phase: dict[str, dict] = {}
    late_by_phase: dict[str, dict] = {}

    # --- Dedicated single-phase models (PHASE1, PHASE2, PHASE3) ---
    for phase in PHASE_SINGLE_MODELS:
        df_p = completed[completed["phase"] == phase].copy()
        n_phase = len(df_p)
        lines.append("=" * 50)
        lines.append(f"MODEL dedicated {phase}  (n={n_phase:,} rows before dropna on target)")
        lines.append("=" * 50)

        if n_phase < 30:
            lines.append("  Skipped: not enough rows for a stable train/val/test split.")
            lines.append("")
            continue

        X, y, _, _ = prepare_features(df_p, **prep_kw)
        n_xy = len(y)
        lines.append(f"  After target present: n={n_xy:,}")

        if n_xy < 30:
            lines.append("  Skipped: too few rows with duration_days after preprocessing.")
            lines.append("")
            continue

        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(X, y)
        lines.append(
            f"  Split: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}"
        )

        model = _new_regressor()
        model.fit(X_train, y_train)

        for split_name, X_s, y_s in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            m = _eval_split(split_name, model, X_s, y_s)
            lines.append(
                f"  {m['set']:5}: RMSE={m['rmse']:,.0f} days  MAE={m['mae']:,.0f} days  R²={m['r2']:.4f}"
            )

        test_m = _eval_split("test", model, X_test, y_test)
        summary[phase] = (len(y_test), test_m["r2"], f"dedicated {phase}-only")
        lines.append("")

    # --- Early joint: train on PHASE1 + PHASE1/PHASE2 + PHASE2; report subset PHASE1/PHASE2 on test ---
    df_early = completed[completed["phase"].isin(EARLY_JOINT_PHASES)].copy()
    lines.append("=" * 50)
    lines.append(
        f"MODEL early joint  (PHASE1 + PHASE1/PHASE2 + PHASE2; n={len(df_early):,} before dropna)"
    )
    lines.append("=" * 50)
    if len(df_early) < 30:
        lines.append("  Skipped: cohort too small.")
        lines.append("")
    else:
        X_e, y_e, ph_e, _ = prepare_features(df_early, **prep_kw)
        lines.append(f"  After target present: n={len(y_e):,}")
        if len(y_e) < 30:
            lines.append("  Skipped: too few rows after preprocessing.")
            lines.append("")
        else:
            X_tr, X_va, X_te_e, y_tr, y_va, y_te_e, _, _, ph_te_e = _train_val_test_split_with_phase(
                X_e, y_e, ph_e
            )
            lines.append(
                f"  Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te_e):,}  (all early-pool phases)"
            )
            m_early = _new_regressor()
            m_early.fit(X_tr, y_tr)
            for split_name, X_s, y_s in (
                ("train", X_tr, y_tr),
                ("val", X_va, y_va),
                ("test", X_te_e, y_te_e),
            ):
                m = _eval_split(split_name, m_early, X_s, y_s)
                lines.append(
                    f"  {m['set']:5}: RMSE={m['rmse']:,.0f} days  MAE={m['mae']:,.0f} days  R²={m['r2']:.4f}"
                )
            early_by_phase = _joint_test_metrics_by_phase(
                m_early,
                X_te_e,
                y_te_e,
                ph_te_e,
                ("PHASE1", "PHASE1/PHASE2", "PHASE2"),
            )
            for ph in ("PHASE1", "PHASE1/PHASE2", "PHASE2"):
                if ph not in early_by_phase:
                    lines.append(f"  test ({ph} subset): too few rows for R².")
                    continue
                d = early_by_phase[ph]
                lines.append(
                    f"  test ({ph} subset): n={d['n']:,}  "
                    f"RMSE={d['rmse']:,.0f}  MAE={d['mae']:,.0f}  R²={d['r2']:.4f}"
                )
            if "PHASE1/PHASE2" in early_by_phase:
                d12 = early_by_phase["PHASE1/PHASE2"]
                summary["PHASE1/PHASE2"] = (
                    d12["n"],
                    d12["r2"],
                    "early joint (PHASE1+PHASE1/PHASE2+PHASE2)",
                )
            lines.append("")

    # --- Late joint: train on PHASE2 + PHASE2/PHASE3 + PHASE3; report subset PHASE2/PHASE3 on test ---
    df_late = completed[completed["phase"].isin(LATE_JOINT_PHASES)].copy()
    lines.append("=" * 50)
    lines.append(
        f"MODEL late joint  (PHASE2 + PHASE2/PHASE3 + PHASE3; n={len(df_late):,} before dropna)"
    )
    lines.append("=" * 50)
    if len(df_late) < 30:
        lines.append("  Skipped: cohort too small.")
        lines.append("")
    else:
        X_l, y_l, ph_l, _ = prepare_features(df_late, **prep_kw)
        lines.append(f"  After target present: n={len(y_l):,}")
        if len(y_l) < 30:
            lines.append("  Skipped: too few rows after preprocessing.")
            lines.append("")
        else:
            X_tr, X_va, X_te_l, y_tr, y_va, y_te_l, _, _, ph_te_l = _train_val_test_split_with_phase(
                X_l, y_l, ph_l
            )
            lines.append(
                f"  Split: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te_l):,}  (all late-pool phases)"
            )
            m_late = _new_regressor()
            m_late.fit(X_tr, y_tr)
            for split_name, X_s, y_s in (
                ("train", X_tr, y_tr),
                ("val", X_va, y_va),
                ("test", X_te_l, y_te_l),
            ):
                m = _eval_split(split_name, m_late, X_s, y_s)
                lines.append(
                    f"  {m['set']:5}: RMSE={m['rmse']:,.0f} days  MAE={m['mae']:,.0f} days  R²={m['r2']:.4f}"
                )
            late_by_phase = _joint_test_metrics_by_phase(
                m_late,
                X_te_l,
                y_te_l,
                ph_te_l,
                ("PHASE2", "PHASE2/PHASE3", "PHASE3"),
            )
            for ph in ("PHASE2", "PHASE2/PHASE3", "PHASE3"):
                if ph not in late_by_phase:
                    lines.append(f"  test ({ph} subset): too few rows for R².")
                    continue
                d = late_by_phase[ph]
                lines.append(
                    f"  test ({ph} subset): n={d['n']:,}  "
                    f"RMSE={d['rmse']:,.0f}  MAE={d['mae']:,.0f}  R²={d['r2']:.4f}"
                )
            if "PHASE2/PHASE3" in late_by_phase:
                d23 = late_by_phase["PHASE2/PHASE3"]
                summary["PHASE2/PHASE3"] = (
                    d23["n"],
                    d23["r2"],
                    "late joint (PHASE2+PHASE2/PHASE3+PHASE3)",
                )
            lines.append("")

    # --- Baseline: old approach (dedicated model trained only on mixed-phase rows) ---
    lines.append("=" * 50)
    lines.append("BASELINE — dedicated model on mixed-phase rows only (previous approach)")
    lines.append("=" * 50)
    for mix_phase in ("PHASE1/PHASE2", "PHASE2/PHASE3"):
        df_m = completed[completed["phase"] == mix_phase].copy()
        n_m = len(df_m)
        lines.append(f"  {mix_phase} cohort: n={n_m:,} before dropna")
        if n_m < 30:
            lines.append(f"    Skipped: too few rows.")
            continue
        X_m, y_m, _, _ = prepare_features(df_m, **prep_kw)
        if len(y_m) < 30:
            lines.append(f"    Skipped after dropna: n={len(y_m):,}")
            continue
        X_tr, X_va, X_te, y_tr, y_va, y_te = _train_val_test_split(X_m, y_m)
        base = _new_regressor()
        base.fit(X_tr, y_tr)
        te = _eval_split("test", base, X_te, y_te)
        lines.append(
            f"    test: n={len(y_te):,}  RMSE={te['rmse']:,.0f}  MAE={te['mae']:,.0f}  R²={te['r2']:.4f}"
        )
    lines.append("")
    lines.append(
        "Interpretation: baseline R² uses a test split drawn only from the mixed-phase cohort; "
        "joint-model R² for that label uses only mixed-phase rows in the joint pool's test fold "
        "(different test composition). Compare qualitatively."
    )
    lines.append("")

    lines.append("=" * 50)
    lines.append("TABLE — JOINT MODELS, ALL PHASE SUBSETS (that phase's rows in the joint test fold)")
    lines.append("Phase\tWhich model\tTest n\tR²\tRMSE (days)\tMAE (days)")
    joint_comparison_order: list[tuple[str, str]] = [
        ("PHASE1", "JOINT_EARLY"),
        ("PHASE1/PHASE2", "JOINT_EARLY"),
        ("PHASE2", "JOINT_EARLY"),
        ("PHASE2", "JOINT_LATE"),
        ("PHASE2/PHASE3", "JOINT_LATE"),
        ("PHASE3", "JOINT_LATE"),
    ]
    for ph, tag in joint_comparison_order:
        src = early_by_phase if tag == "JOINT_EARLY" else late_by_phase
        if ph not in src:
            continue
        d = src[ph]
        lines.append(
            f"{ph}\t{tag}\t{d['n']:,}\t{d['r2']:.4f}\t{round(d['rmse'])}\t{round(d['mae'])}"
        )
    lines.append("")

    lines.append("=" * 50)
    lines.append("TABLE — BEST JOINT PER LABEL (highest test R² when both joints cover the label)")
    lines.append("Phase\tWhich model\tTest n\tR²\tRMSE (days)\tMAE (days)")
    for ph in PHASE_REPORT_ORDER:
        cands: list[tuple[str, dict]] = []
        if ph in early_by_phase:
            cands.append(("JOINT_EARLY", early_by_phase[ph]))
        if ph in late_by_phase:
            cands.append(("JOINT_LATE", late_by_phase[ph]))
        if not cands:
            continue
        tag, d = max(cands, key=lambda x: x[1]["r2"])
        lines.append(
            f"{ph}\t{tag}\t{d['n']:,}\t{d['r2']:.4f}\t{round(d['rmse'])}\t{round(d['mae'])}"
        )
    lines.append("")

    lines.append("=" * 50)
    lines.append("SUMMARY — TEST R² BY ROW PHASE (model used for that label)")
    lines.append("=" * 50)
    for ph in PHASE_REPORT_ORDER:
        if ph in summary:
            n_t, r2, desc = summary[ph]
            lines.append(f"  {ph}: n={n_t:,}  R²={r2:.4f}  — {desc}")
        else:
            lines.append(f"  {ph}: n/a  (skipped or insufficient test rows)")
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "regression_report.txt").write_text(report)
    logger.info("Wrote %s", RESULTS_DIR / "regression_report.txt")


if __name__ == "__main__":
    main()
