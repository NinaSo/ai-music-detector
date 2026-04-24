from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_branch(entries: list[dict], branch: str) -> dict:
    metrics = ["accuracy", "auroc", "auprc"]
    out: dict[str, float] = {}
    vals = {m: [] for m in metrics}

    for e in entries:
        b = e.get(branch, {})
        t = b.get("test", {})
        for m in metrics:
            v = t.get(m)
            if isinstance(v, (int, float)):
                vals[m].append(float(v))

    for m in metrics:
        arr = np.array(vals[m], dtype=np.float64)
        if arr.size == 0:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
        else:
            out[f"{m}_mean"] = float(arr.mean())
            out[f"{m}_std"] = float(arr.std(ddof=0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple holdout-source evaluations and aggregate mean/std")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--holdout_sources",
        type=str,
        default="musicldm,audioldm2,mustango,stable_audio_open",
        help="Comma-separated AI sources to hold out one-by-one",
    )
    parser.add_argument("--group_col", type=str, default="group_id")
    parser.add_argument("--human_test_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn_epochs", type=int, default=8)
    parser.add_argument("--cnn_batch_size", type=int, default=16)
    parser.add_argument("--cnn_num_workers", type=int, default=0)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--calibration_metric", type=str, default="f1", choices=["f1", "balanced_accuracy", "youden_j"])
    parser.add_argument("--run_alignment", action="store_true", help="Enable alignment branch (off by default)")
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    sources = [s.strip() for s in args.holdout_sources.split(",") if s.strip()]
    if not sources:
        raise ValueError("No holdout sources provided")

    args.output_root.mkdir(parents=True, exist_ok=True)

    per_source: list[dict] = []
    for source in sources:
        out_dir = args.output_root / f"holdout_{source}"
        cmd = [
            sys.executable,
            "-m",
            "aidetector.train_all",
            "--manifest",
            str(args.manifest),
            "--output_dir",
            str(out_dir),
            "--group_col",
            args.group_col,
            "--holdout_ai_source",
            source,
            "--human_test_ratio",
            str(args.human_test_ratio),
            "--seed",
            str(args.seed),
            "--sample_rate",
            str(args.sample_rate),
            "--clip_seconds",
            str(args.clip_seconds),
            "--cnn_batch_size",
            str(args.cnn_batch_size),
            "--cnn_epochs",
            str(args.cnn_epochs),
            "--cnn_lr",
            str(args.cnn_lr),
            "--cnn_num_workers",
            str(args.cnn_num_workers),
            "--calibration_metric",
            args.calibration_metric,
        ]

        if not args.run_alignment:
            cmd.append("--skip_alignment")
        if args.skip_cnn:
            cmd.append("--skip_cnn")
        if args.skip_baseline:
            cmd.append("--skip_baseline")

        run_command(cmd)

        metrics_path = out_dir / "combined_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")

        row = load_json(metrics_path)
        row["holdout_source"] = source
        per_source.append(row)

    summary = {
        "manifest": str(args.manifest),
        "holdout_sources": sources,
        "per_source": per_source,
        "cnn_summary": summarize_branch(per_source, "cnn"),
        "baseline_summary": summarize_branch(per_source, "baseline"),
    }

    out_path = args.output_root / "holdout_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
