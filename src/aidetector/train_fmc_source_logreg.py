from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from aidetector.data import load_manifest
from aidetector.train_baseline import extract_features


def split_multiclass(
    df,
    target_col: str,
    group_col: str | None,
    val_size: float,
    test_size: float,
    seed: int,
):
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("__missing_group__").astype(str)
        splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(splitter_test.split(df, groups=groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        rel_val = val_size / (1.0 - test_size)
        tv_groups = train_val_df[group_col].fillna("__missing_group__").astype(str)
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        train_idx, val_idx = next(splitter_val.split(train_val_df, groups=tv_groups))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        return train_df, val_df, test_df

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )
    rel_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=rel_val,
        random_state=seed,
        stratify=train_df[target_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_xy(df, target_col: str):
    x = []
    y = []
    dropped = 0
    for row in tqdm(df.to_dict("records"), desc="fmc source features"):
        try:
            x.append(extract_features(str(row["audio_path"])))
            y.append(str(row[target_col]))
        except Exception:
            dropped += 1
    if not x:
        raise RuntimeError("No valid audio samples for FMC source classifier")
    return np.stack(x), np.array(y), dropped


def evaluate(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, labels: list[str]):
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    macro_f1 = float(f1_score(y_true_idx, y_pred_idx, average="macro"))
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FMC generator/source using logistic regression")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_col", type=str, default="source")
    parser.add_argument("--group_col", type=str, default="group_id")
    parser.add_argument("--include_human", action="store_true", help="Include MusicCaps (human) as a class")
    parser.add_argument("--min_class_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    if "source" not in df.columns:
        raise ValueError("Manifest needs `source` column for FMC source classification")

    df = df[df["source"] != "fma"].copy()
    if not args.include_human:
        df = df[df["label"] == 1].copy()

    counts = df[args.target_col].value_counts()
    keep = counts[counts >= args.min_class_samples].index
    df = df[df[args.target_col].isin(keep)].reset_index(drop=True)

    if df[args.target_col].nunique() < 2:
        raise ValueError("Need at least 2 classes for FMC source classification")

    train_df, val_df, test_df = split_multiclass(
        df,
        target_col=args.target_col,
        group_col=args.group_col if args.group_col else None,
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
    )

    x_train, y_train, d_train = build_xy(train_df, args.target_col)
    x_val, y_val, d_val = build_xy(val_df, args.target_col)
    x_test, y_test, d_test = build_xy(test_df, args.target_col)

    le = LabelEncoder()
    y_train_idx = le.fit_transform(y_train)

    val_mask = np.isin(y_val, le.classes_)
    test_mask = np.isin(y_test, le.classes_)
    dropped_unseen = {
        "val": int((~val_mask).sum()),
        "test": int((~test_mask).sum()),
    }
    x_val = x_val[val_mask]
    y_val = y_val[val_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]
    if len(y_val) == 0 or len(y_test) == 0:
        raise RuntimeError("After filtering unseen classes, val/test became empty. Increase data per class.")

    y_val_idx = le.transform(y_val)
    y_test_idx = le.transform(y_test)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train_idx)

    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)

    labels = [str(x) for x in le.classes_]
    results = {
        "classes": labels,
        "class_counts": {str(k): int(v) for k, v in counts.loc[keep].items()},
        "dropped_corrupt": {"train": int(d_train), "val": int(d_val), "test": int(d_test)},
        "dropped_unseen_class": dropped_unseen,
        "val": evaluate(y_val_idx, val_pred, labels),
        "test": evaluate(y_test_idx, test_pred, labels),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le, "classes": labels}, args.output_dir / "fmc_source_logreg.joblib")
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
