from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from aidetector.audio_io import load_audio
from aidetector.data import load_manifest, split_df, split_df_grouped, split_df_holdout_ai_source
from aidetector.metrics import binary_metrics


def extract_features(audio_path: str) -> np.ndarray:
    wav, sr = load_audio(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    duration_sec = float(wav.numel()) / float(sr)
    rms = torch.sqrt(torch.mean(wav.pow(2)) + 1e-12)
    loudness_db = float(20.0 * torch.log10(rms + 1e-12))

    if wav.numel() < 2:
        zcr = 0.0
    else:
        signs = torch.sign(wav)
        zc = (signs[1:] * signs[:-1] < 0).float().mean()
        zcr = float(zc.item())

    return np.array([duration_sec, loudness_db, zcr], dtype=np.float32)


def build_xy(df):
    x = []
    y = []
    dropped = 0
    for row in tqdm(df.to_dict("records"), desc="baseline features"):
        try:
            x.append(extract_features(str(row["audio_path"])))
            y.append(int(row["label"]))
        except Exception:
            dropped += 1
    if not x:
        raise RuntimeError("No valid audio samples for baseline feature extraction.")
    return np.stack(x), np.array(y, dtype=np.int32), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sanity baseline (duration + loudness + ZCR)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--group_col", type=str, default=None, help="Optional leakage-control group column for splitting")
    parser.add_argument("--holdout_ai_source", type=str, default=None, help="Optional strict test: unseen AI source")
    parser.add_argument("--human_test_ratio", type=float, default=1.0, help="Human:AI ratio in holdout test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    if args.holdout_ai_source:
        splits = split_df_holdout_ai_source(
            df,
            holdout_ai_source=args.holdout_ai_source,
            group_col=args.group_col,
            val_size=0.15,
            human_test_ratio=args.human_test_ratio,
            seed=args.seed,
        )
        print(
            f"Using holdout source split: holdout_ai_source={args.holdout_ai_source}, "
            f"test_size={len(splits.test)}"
        )
    elif args.group_col:
        splits = split_df_grouped(df, group_col=args.group_col, val_size=0.15, test_size=0.15, seed=args.seed)
    else:
        splits = split_df(df, val_size=0.15, test_size=0.15, seed=args.seed)

    x_train, y_train, drop_train = build_xy(splits.train)
    x_val, y_val, drop_val = build_xy(splits.val)
    x_test, y_test, drop_test = build_xy(splits.test)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)

    val_prob = clf.predict_proba(x_val)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]

    results = {
        "features": ["duration_sec", "loudness_db", "zero_crossing_rate"],
        "dropped_corrupt": {"train": int(drop_train), "val": int(drop_val), "test": int(drop_test)},
        "val": binary_metrics(y_val, val_prob),
        "test": binary_metrics(y_test, test_prob),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.output_dir / "baseline_logreg.joblib")
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
