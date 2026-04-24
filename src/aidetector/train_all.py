from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train prototype branches (alignment + CNN + baseline)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    # Shared.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_col", type=str, default=None, help="Optional leakage-control group column")
    parser.add_argument("--holdout_ai_source", type=str, default=None, help="Optional strict test: unseen AI source")
    parser.add_argument("--human_test_ratio", type=float, default=1.0, help="Human:AI ratio in holdout test set")

    # Alignment branch params.
    parser.add_argument("--skip_alignment", action="store_true")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--alignment_batch_size", type=int, default=8)

    # CNN branch params.
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--cnn_batch_size", type=int, default=16)
    parser.add_argument("--cnn_epochs", type=int, default=8)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--cnn_num_workers", type=int, default=0)
    parser.add_argument(
        "--cnn_init_model",
        type=Path,
        default=None,
        help="Optional path to existing cnn_best.pt to continue training",
    )
    parser.add_argument(
        "--calibration_metric",
        type=str,
        default="f1",
        choices=["f1", "balanced_accuracy", "youden_j"],
    )

    # Baseline branch params.
    parser.add_argument("--skip_baseline", action="store_true")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir = args.output_dir / "alignment"
    cnn_dir = args.output_dir / "cnn"
    baseline_dir = args.output_dir / "baseline"

    if not args.skip_alignment:
        cmd = [
            sys.executable,
            "-m",
            "aidetector.train_alignment",
            "--manifest",
            str(args.manifest),
            "--output_dir",
            str(alignment_dir),
            "--clap_model",
            args.clap_model,
            "--batch_size",
            str(args.alignment_batch_size),
            "--seed",
            str(args.seed),
        ]
        if args.group_col:
            cmd.extend(["--group_col", args.group_col])
        if args.holdout_ai_source:
            cmd.extend(["--holdout_ai_source", args.holdout_ai_source])
            cmd.extend(["--human_test_ratio", str(args.human_test_ratio)])
        if args.cnn_init_model:
            cmd.extend(["--init_model", str(args.cnn_init_model)])
        run_command(cmd)

    if not args.skip_cnn:
        cmd = [
            sys.executable,
            "-m",
            "aidetector.train_cnn",
            "--manifest",
            str(args.manifest),
            "--output_dir",
            str(cnn_dir),
            "--sample_rate",
            str(args.sample_rate),
            "--clip_seconds",
            str(args.clip_seconds),
            "--batch_size",
            str(args.cnn_batch_size),
            "--epochs",
            str(args.cnn_epochs),
            "--lr",
            str(args.cnn_lr),
            "--num_workers",
            str(args.cnn_num_workers),
            "--calibration_metric",
            args.calibration_metric,
            "--seed",
            str(args.seed),
        ]
        if args.group_col:
            cmd.extend(["--group_col", args.group_col])
        if args.holdout_ai_source:
            cmd.extend(["--holdout_ai_source", args.holdout_ai_source])
            cmd.extend(["--human_test_ratio", str(args.human_test_ratio)])
        run_command(cmd)

    if not args.skip_baseline:
        cmd = [
            sys.executable,
            "-m",
            "aidetector.train_baseline",
            "--manifest",
            str(args.manifest),
            "--output_dir",
            str(baseline_dir),
            "--seed",
            str(args.seed),
        ]
        if args.group_col:
            cmd.extend(["--group_col", args.group_col])
        if args.holdout_ai_source:
            cmd.extend(["--holdout_ai_source", args.holdout_ai_source])
            cmd.extend(["--human_test_ratio", str(args.human_test_ratio)])
        run_command(cmd)

    combined = {
        "manifest": str(args.manifest),
        "alignment": load_json_if_exists(alignment_dir / "metrics.json"),
        "cnn": load_json_if_exists(cnn_dir / "metrics.json"),
        "baseline": load_json_if_exists(baseline_dir / "metrics.json"),
    }

    combined_path = args.output_dir / "combined_metrics.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Wrote combined summary to: {combined_path}")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
