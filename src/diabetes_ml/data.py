import hashlib
import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ROOT, TARGET, FEATURES, write_json

FILENAME = "diabetes_012_health_indicators_BRFSS2015.csv"
URL = "https://www.kaggle.com/api/v1/datasets/download/alexteboul/diabetes-health-indicators-dataset"


def download():
    directory = ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FILENAME
    if target.exists():
        return target
    archive = directory / "kaggle_dataset.zip"
    if not archive.exists():
        with urlopen(Request(URL, headers={"User-Agent": "DiabetesML/0.1"}), timeout=90) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.split("/")[-1] == FILENAME)
        target.write_bytes(bundle.read(member))
    write_json(directory / "source.json", {
        "source_url": "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
        "download_url": URL, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "filename": FILENAME, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    })
    return target


def prepare(seed=42):
    path = download()
    df = pd.read_csv(path)
    assert list(df.columns) == [TARGET] + FEATURES, "Unexpected data schema"
    assert len(df) == 253680, "Unexpected source row count"
    assert df.notna().all().all(), "Missing values must be investigated"
    assert np.isfinite(df.to_numpy()).all()
    assert set(df[TARGET].unique()) == {0, 1, 2}
    assert (df.to_numpy() == np.floor(df.to_numpy())).all(), "Expected integer survey codes"
    binary = [c for c in FEATURES if c not in ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]]
    assert all(set(df[c].unique()).issubset({0, 1}) for c in binary)
    ranges = {"BMI": (12, 98), "GenHlth": (1, 5), "MentHlth": (0, 30),
              "PhysHlth": (0, 30), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)}
    assert all(df[c].between(*limits).all() for c, limits in ranges.items())
    source = json.loads((ROOT / "data/raw/source.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["sha256"] == digest, "Raw data changed since download"
    y = df[TARGET].to_numpy(dtype=np.int64)
    train, hold = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
    validation, test = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=seed)
    tuning, selection = train_test_split(validation, test_size=0.5, stratify=y[validation], random_state=seed + 1)
    splits = dict(train=train, tuning=tuning, selection=selection, test=test)
    # IDs are original row positions, never used as predictors.
    np.savez_compressed(ROOT / "data/processed/splits.npz", **splits)
    split_rows = []
    for name, idx in splits.items():
        for c in range(3):
            split_rows.append({"split": name, "class": c, "count": int((y[idx] == c).sum()), "total": len(idx)})
    pd.DataFrame(split_rows).to_csv(ROOT / "results/split_counts.csv", index=False)
    feature_hash = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    train_hashes = np.unique(feature_hash[train])
    profile = {
        "rows": len(df), "feature_count": len(FEATURES), "source_sha256": digest,
        "class_counts": {str(c): int((y == c).sum()) for c in range(3)},
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_full_rows_after_first": int(df.duplicated().sum()),
        "duplicate_feature_profiles_after_first": int(df[FEATURES].duplicated().sum()),
        "test_profiles_seen_in_training": int(np.isin(feature_hash[test], train_hashes).sum()),
        "duplicate_policy": "Keep original rows. Coarse survey answers can coincide for different respondents. Report unseen-profile test subset separately; profile hashes are not person identifiers.",
        "split_seed": seed, "split_ratios": {"train": .8, "tuning": .05, "selection": .05, "test": .1},
        "feature_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in FEATURES},
    }
    write_json(ROOT / "results/data_audit.json", profile)
    df.describe().T.to_csv(ROOT / "results/descriptive_statistics.csv")
    return df, splits, feature_hash
