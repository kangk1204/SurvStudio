from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def make_example_dataset(seed: int = 42, n_patients: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(61.5, 10.2, size=n_patients), 29, 87)
    sex = rng.choice(["Female", "Male"], p=[0.44, 0.56], size=n_patients)
    stage = rng.choice(["I", "II", "III", "IV"], p=[0.16, 0.30, 0.34, 0.20], size=n_patients)
    treatment = rng.choice(["Standard", "Combination"], p=[0.52, 0.48], size=n_patients)
    smoking = rng.choice(["Never", "Former", "Current"], p=[0.43, 0.33, 0.24], size=n_patients)
    biomarker_score = rng.normal(0, 1, size=n_patients)
    immune_index = rng.normal(0.4, 0.9, size=n_patients)

    stage_effect = pd.Series(stage).map({"I": 0.0, "II": 0.30, "III": 0.78, "IV": 1.20}).to_numpy()
    treatment_effect = pd.Series(treatment).map({"Standard": 0.0, "Combination": -0.36}).to_numpy()
    sex_effect = pd.Series(sex).map({"Female": 0.0, "Male": 0.12}).to_numpy()
    smoking_effect = pd.Series(smoking).map({"Never": 0.0, "Former": 0.16, "Current": 0.34}).to_numpy()

    os_linear_predictor = (
        0.028 * (age - 60)
        + 0.48 * biomarker_score
        - 0.24 * immune_index
        + stage_effect
        + treatment_effect
        + sex_effect
        + smoking_effect
    )
    pfs_linear_predictor = os_linear_predictor + 0.18 * biomarker_score + 0.12 * (stage == "IV")

    os_shape = 1.32
    pfs_shape = 1.18
    os_scale = 39.0
    pfs_scale = 24.0

    os_u = rng.random(n_patients)
    pfs_u = rng.random(n_patients)
    censor = rng.uniform(7.0, 72.0, size=n_patients)

    os_event_time = os_scale * ((-np.log(os_u)) / np.exp(os_linear_predictor)) ** (1 / os_shape)
    pfs_event_time = pfs_scale * ((-np.log(pfs_u)) / np.exp(pfs_linear_predictor)) ** (1 / pfs_shape)

    os_time = np.minimum(os_event_time, censor)
    os_event = (os_event_time <= censor).astype(int)
    pfs_time = np.minimum(pfs_event_time, censor)
    pfs_event = (pfs_event_time <= censor).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"PT-{idx:04d}" for idx in range(1, n_patients + 1)],
            "os_months": np.round(os_time, 2),
            "os_event": os_event,
            "pfs_months": np.round(pfs_time, 2),
            "pfs_event": pfs_event,
            "age": np.round(age, 1),
            "sex": sex,
            "stage": stage,
            "treatment": treatment,
            "smoking_status": smoking,
            "biomarker_score": np.round(biomarker_score, 3),
            "immune_index": np.round(immune_index, 3),
        }
    )

    # Inject missingness in a size-safe way so small cohorts do not crash.
    immune_missing_n = min(n_patients, int(round(n_patients * 0.05)), 18)
    smoking_missing_n = min(n_patients, int(round(n_patients * (12 / 360))), 12)
    if immune_missing_n > 0:
        df.loc[rng.choice(df.index, size=immune_missing_n, replace=False), "immune_index"] = np.nan
    if smoking_missing_n > 0:
        df.loc[rng.choice(df.index, size=smoking_missing_n, replace=False), "smoking_status"] = np.nan

    return df


def load_tcga_luad_example_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "tcga_luad_xena_example.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing bundled TCGA example dataset: {dataset_path}")
    return pd.read_csv(dataset_path)


def load_tcga_luad_upload_ready_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "tcga_luad_upload_ready.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing bundled upload-ready TCGA dataset: {dataset_path}")
    return pd.read_csv(dataset_path)


def load_gbsg2_upload_ready_dataset() -> pd.DataFrame:
    dataset_path = DATA_DIR / "gbsg2_upload_ready.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing bundled upload-ready GBSG2 dataset: {dataset_path}")
    return pd.read_csv(dataset_path)
