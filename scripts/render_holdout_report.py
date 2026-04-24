#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_metric(v: float | int | None, digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{digits}f}"


def metric_block(row: dict, key: str) -> dict:
    block = row.get(key, {})
    if key == "cnn":
        return block.get("test", {}) or block.get("test_calibrated", {}) or {}
    return block.get("test", {}) or {}


def build_report(summary: dict) -> str:
    per_source = summary.get("per_source", [])
    lines: list[str] = []

    lines.append("# Holdout Generalization Report")
    lines.append("")
    lines.append(f"- Manifest: `{summary.get('manifest', '-')}`")
    lines.append(f"- Holdout sources: `{', '.join(summary.get('holdout_sources', []))}`")
    lines.append("")
    lines.append("## Per-Source Results")
    lines.append("")
    lines.append("| Source | CNN Thr | CNN Acc (0.5) | CNN Acc (cal) | CNN AUROC | CNN AUPRC | Baseline Acc | Baseline AUROC | Baseline AUPRC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in per_source:
        source = row.get("holdout_source", "-")
        cnn = row.get("cnn", {})
        cnn_test_default = cnn.get("test_default", {})
        cnn_test_cal = cnn.get("test_calibrated", cnn.get("test", {}))
        baseline_test = row.get("baseline", {}).get("test", {})
        threshold = cnn_test_cal.get("threshold", cnn.get("calibration", {}).get("threshold"))

        lines.append(
            "| "
            + " | ".join(
                [
                    str(source),
                    fmt_metric(threshold),
                    fmt_metric(cnn_test_default.get("accuracy")),
                    fmt_metric(cnn_test_cal.get("accuracy")),
                    fmt_metric(cnn_test_cal.get("auroc")),
                    fmt_metric(cnn_test_cal.get("auprc")),
                    fmt_metric(baseline_test.get("accuracy")),
                    fmt_metric(baseline_test.get("auroc")),
                    fmt_metric(baseline_test.get("auprc")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Aggregate Summary (Mean +/- Std Across Holdouts)")
    lines.append("")
    lines.append("| Model | Accuracy | AUROC | AUPRC |")
    lines.append("|---|---:|---:|---:|")

    cnn_s = summary.get("cnn_summary", {})
    base_s = summary.get("baseline_summary", {})
    lines.append(
        f"| CNN | {fmt_metric(cnn_s.get('accuracy_mean'))} +/- {fmt_metric(cnn_s.get('accuracy_std'))} "
        f"| {fmt_metric(cnn_s.get('auroc_mean'))} +/- {fmt_metric(cnn_s.get('auroc_std'))} "
        f"| {fmt_metric(cnn_s.get('auprc_mean'))} +/- {fmt_metric(cnn_s.get('auprc_std'))} |"
    )
    lines.append(
        f"| Baseline | {fmt_metric(base_s.get('accuracy_mean'))} +/- {fmt_metric(base_s.get('accuracy_std'))} "
        f"| {fmt_metric(base_s.get('auroc_mean'))} +/- {fmt_metric(base_s.get('auroc_std'))} "
        f"| {fmt_metric(base_s.get('auprc_mean'))} +/- {fmt_metric(base_s.get('auprc_std'))} |"
    )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `CNN Acc (cal)` uses the validation-selected threshold per holdout (`calibration_metric`).")
    lines.append("- AUROC/AUPRC are threshold-free and should be compared first for robustness.")
    lines.append("- Baseline uses fixed threshold `0.5` unless separately calibrated.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one-page Markdown report from holdout_summary.json")
    parser.add_argument("--summary", required=True, type=Path, help="Path to holdout_summary.json")
    parser.add_argument("--output", type=Path, default=None, help="Output Markdown path")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    report = build_report(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote report: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
