from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio

from aidetector.audio_io import load_audio
from aidetector.cnn_detector import CNNArtifactDetector
from aidetector.device import get_best_device
from aidetector.train_baseline import extract_features


def to_mono_resampled(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def fix_clip_length(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    curr = waveform.shape[-1]
    if curr == num_samples:
        return waveform
    if curr > num_samples:
        start = (curr - num_samples) // 2
        return waveform[:, start : start + num_samples]

    padded = torch.zeros((1, num_samples), dtype=waveform.dtype)
    padded[:, :curr] = waveform
    return padded


def _build_eval_clips(waveform: torch.Tensor, num_samples: int, num_eval_clips: int) -> list[torch.Tensor]:
    curr = waveform.shape[-1]
    if curr <= num_samples:
        return [fix_clip_length(waveform, num_samples)]

    if num_eval_clips <= 1:
        return [fix_clip_length(waveform, num_samples)]

    starts = np.linspace(0, curr - num_samples, num=num_eval_clips, dtype=int).tolist()
    return [waveform[:, s : s + num_samples] for s in starts]


def predict_cnn_with_details(
    audio_path: str,
    model_path: Path,
    sample_rate: int,
    clip_seconds: float,
    num_eval_clips: int = 1,
    agg: str = "mean",
) -> dict:
    device = get_best_device()
    model = CNNArtifactDetector(sample_rate=sample_rate).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    waveform, sr = load_audio(audio_path)
    waveform = to_mono_resampled(waveform, sr, sample_rate)
    num_samples = int(sample_rate * clip_seconds)
    clips = _build_eval_clips(waveform, num_samples=num_samples, num_eval_clips=num_eval_clips)

    x = torch.stack(clips, dim=0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

    if agg == "max":
        prob = float(np.max(probs))
    elif agg == "median":
        prob = float(np.median(probs))
    else:
        prob = float(np.mean(probs))

    return {
        "cnn_prob": prob,
        "cnn_clip_probs": probs.astype(float).tolist(),
        "cnn_eval_clips": int(len(clips)),
        "cnn_agg": agg,
    }


def predict_cnn(
    audio_path: str,
    model_path: Path,
    sample_rate: int,
    clip_seconds: float,
    num_eval_clips: int = 1,
    agg: str = "mean",
) -> float:
    out = predict_cnn_with_details(
        audio_path=audio_path,
        model_path=model_path,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        num_eval_clips=num_eval_clips,
        agg=agg,
    )
    return float(out["cnn_prob"])


def predict_baseline(audio_path: str, model_path: Path) -> float:
    clf = joblib.load(model_path)
    x = extract_features(audio_path).reshape(1, -1)
    prob = clf.predict_proba(x)[0, 1]
    return float(prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AI probability for a single audio file")
    parser.add_argument("--audio", type=Path, required=True, help="Path to input audio file")
    parser.add_argument("--cnn_model", type=Path, default=None, help="Path to cnn_best.pt")
    parser.add_argument("--baseline_model", type=Path, default=None, help="Path to baseline_logreg.joblib")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--cnn_eval_clips", type=int, default=1, help="Number of evenly spaced clips for CNN eval")
    parser.add_argument(
        "--cnn_agg",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="Aggregation over multi-clip CNN probabilities",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold_json",
        type=Path,
        default=None,
        help="Optional path to threshold.json produced by train_cnn; overrides --threshold",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    scores: dict[str, float] = {}
    details: dict = {}

    if args.cnn_model is not None:
        cnn_out = predict_cnn_with_details(
            audio_path=str(args.audio),
            model_path=args.cnn_model,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            num_eval_clips=max(1, int(args.cnn_eval_clips)),
            agg=args.cnn_agg,
        )
        scores["cnn_prob"] = float(cnn_out["cnn_prob"])
        details.update(cnn_out)

    if args.baseline_model is not None:
        scores["baseline_prob"] = predict_baseline(
            audio_path=str(args.audio),
            model_path=args.baseline_model,
        )

    if not scores:
        raise ValueError("Provide at least one model path: --cnn_model and/or --baseline_model")

    threshold = float(args.threshold)
    threshold_source = "arg"
    if args.threshold_json is not None:
        with open(args.threshold_json, "r", encoding="utf-8") as f:
            t = json.load(f)
        if "threshold" not in t:
            raise ValueError(f"threshold json missing `threshold`: {args.threshold_json}")
        threshold = float(t["threshold"])
        threshold_source = str(args.threshold_json)

    fused_prob = float(np.mean(list(scores.values())))
    pred_label = int(fused_prob >= threshold)

    out = {
        "audio": str(args.audio),
        **scores,
        **details,
        "fused_prob": fused_prob,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "pred_label": pred_label,
        "pred_label_name": "ai_generated" if pred_label == 1 else "human",
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
