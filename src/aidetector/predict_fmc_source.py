from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from aidetector.train_baseline import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FMC generator/source for a single audio file")
    parser.add_argument("--audio", type=Path, required=True, help="Path to input audio file")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to fmc_source_logreg.joblib",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top classes to return")
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    blob = joblib.load(args.model)
    if not isinstance(blob, dict) or "model" not in blob or "label_encoder" not in blob:
        raise ValueError("Model file must contain dict with keys: model, label_encoder")

    clf = blob["model"]
    le = blob["label_encoder"]

    x = extract_features(str(args.audio)).reshape(1, -1)
    probs = clf.predict_proba(x)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(le.inverse_transform([pred_idx])[0])

    order = np.argsort(probs)[::-1]
    top_k = max(1, min(args.top_k, len(order)))
    top_preds = [
        {
            "label": str(le.inverse_transform([int(i)])[0]),
            "prob": float(probs[int(i)]),
        }
        for i in order[:top_k]
    ]

    out = {
        "audio": str(args.audio),
        "pred_label": pred_label,
        "pred_prob": float(probs[pred_idx]),
        "top_k": top_preds,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
