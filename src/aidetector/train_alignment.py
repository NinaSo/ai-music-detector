from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from aidetector.clap_audio import CLAPAudioConfig, CLAPAudioEmbedder
from aidetector.data import load_manifest, split_df, split_df_grouped, split_df_holdout_ai_source
from aidetector.metrics import binary_metrics


def build_xy(rows: list[dict], embedder: CLAPAudioEmbedder, batch_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    x_batches: list[np.ndarray] = []
    y_all: list[int] = []
    dropped = 0

    for i in tqdm(range(0, len(rows), batch_size), desc="CLAP-audio embedding"):
        batch = rows[i : i + batch_size]
        audio_arrays: list[np.ndarray] = []
        labels: list[int] = []
        for r in batch:
            try:
                audio_arrays.append(embedder.load_audio_array(str(r["audio_path"])))
                labels.append(int(r["label"]))
            except Exception:
                dropped += 1
        if not audio_arrays:
            continue
        x_batches.append(embedder.embed_audio_arrays(audio_arrays))
        y_all.extend(labels)

    if not x_batches:
        raise RuntimeError("No valid audio samples for CLAP-audio feature extraction.")
    x = np.concatenate(x_batches, axis=0).astype(np.float32)
    y = np.array(y_all, dtype=np.int32)
    return x, y, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLAP-audio detector (no captions required)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--group_col", type=str, default=None, help="Optional leakage-control group column for splitting")
    parser.add_argument("--holdout_ai_source", type=str, default=None, help="Optional strict test: unseen AI source")
    parser.add_argument("--human_test_ratio", type=float, default=1.0, help="Human:AI ratio in holdout test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    if len(df) < 20:
        raise ValueError("Need more rows for CLAP-audio training")

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

    embedder = CLAPAudioEmbedder(
        CLAPAudioConfig(
            model_name=args.clap_model,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
        )
    )

    x_train, y_train, d_train = build_xy(splits.train.to_dict("records"), embedder, args.batch_size)
    x_val, y_val, d_val = build_xy(splits.val.to_dict("records"), embedder, args.batch_size)
    x_test, y_test, d_test = build_xy(splits.test.to_dict("records"), embedder, args.batch_size)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    clf.fit(x_train, y_train)

    val_prob = clf.predict_proba(x_val)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]

    results = {
        "feature_type": "clap_audio_embedding",
        "embedding_dim": int(x_train.shape[1]),
        "dropped_corrupt": {"train": int(d_train), "val": int(d_val), "test": int(d_test)},
        "val": binary_metrics(y_val, val_prob),
        "test": binary_metrics(y_test, test_prob),
        "notes": "CLAP audio embeddings + logistic regression (no captions).",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": clf,
            "clap_model": args.clap_model,
            "sample_rate": int(args.sample_rate),
            "feature_type": "clap_audio_embedding",
        },
        args.output_dir / "alignment_logreg.joblib",
    )
    np.save(args.output_dir / "val_scores.npy", val_prob)
    np.save(args.output_dir / "test_scores.npy", test_prob)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
