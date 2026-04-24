from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from aidetector.clap_audio import CLAPAudioConfig, CLAPAudioEmbedder


def predict_clap_audio_prob(audio_path: str, model_path: Path) -> float:
    blob = joblib.load(model_path)
    if not isinstance(blob, dict) or "classifier" not in blob:
        raise ValueError("Invalid CLAP model file. Expected dict with key `classifier`.")

    clf = blob["classifier"]
    clap_model = str(blob.get("clap_model", "laion/clap-htsat-unfused"))
    sample_rate = int(blob.get("sample_rate", 48000))

    embedder = CLAPAudioEmbedder(
        CLAPAudioConfig(model_name=clap_model, sample_rate=sample_rate, batch_size=1)
    )
    arr = embedder.load_audio_array(audio_path)
    x = embedder.embed_audio_arrays([arr])
    prob = clf.predict_proba(x)[0, 1]
    return float(prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AI probability with CLAP-audio branch (no captions)")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to alignment_logreg.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    prob = predict_clap_audio_prob(str(args.audio), args.model)
    pred = int(prob >= float(args.threshold))
    out = {
        "audio": str(args.audio),
        "clap_prob": prob,
        "threshold": float(args.threshold),
        "pred_label": pred,
        "pred_label_name": "ai_generated" if pred == 1 else "human",
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
