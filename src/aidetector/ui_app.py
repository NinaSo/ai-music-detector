from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from aidetector.predict import predict_baseline, predict_cnn_with_details
from aidetector.predict_clap_audio import predict_clap_audio_prob
from aidetector.train_baseline import extract_features

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_default_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "combined_metrics": run_dir / "combined_metrics.json",
        "cnn_model": run_dir / "cnn" / "cnn_best.pt",
        "threshold_json": run_dir / "cnn" / "threshold.json",
        "baseline_model": run_dir / "baseline" / "baseline_logreg.joblib",
    }


def parse_model_test_metrics(combined: dict) -> dict[str, float | None]:
    cnn_test = combined.get("cnn", {}).get("test", {})
    baseline_test = combined.get("baseline", {}).get("test", {})
    alignment_test = combined.get("alignment", {}).get("test", {})
    return {
        "cnn_accuracy": cnn_test.get("accuracy"),
        "cnn_auroc": cnn_test.get("auroc"),
        "cnn_auprc": cnn_test.get("auprc"),
        "clap_accuracy": alignment_test.get("accuracy"),
        "clap_auroc": alignment_test.get("auroc"),
        "clap_auprc": alignment_test.get("auprc"),
        "baseline_accuracy": baseline_test.get("accuracy"),
        "baseline_auroc": baseline_test.get("auroc"),
        "baseline_auprc": baseline_test.get("auprc"),
    }


def cnn_metrics_for_comparison(combined: dict) -> dict[str, float | None]:
    cnn = combined.get("cnn", {})
    return {
        "acc_default": cnn.get("test_default", {}).get("accuracy"),
        "acc_calibrated": cnn.get("test_calibrated", {}).get("accuracy", cnn.get("test", {}).get("accuracy")),
        "auroc": cnn.get("test", {}).get("auroc"),
        "auprc": cnn.get("test", {}).get("auprc"),
    }


def _get_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _load_font(size: int = 24):
    """Load a font cross-platform, falling back to PIL default if none found."""
    from PIL import ImageFont
    import sys

    candidates = []
    if sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    elif sys.platform.startswith("win"):
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _get_pil_modules():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None, None, None
    return Image, ImageDraw, ImageFont


def _draw_grouped_bar_png(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, np.ndarray]],
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    Image, ImageDraw, ImageFont = _get_pil_modules()
    if Image is None:
        raise RuntimeError("No plotting backend available. Install matplotlib or pillow.")

    w, h = 1600, 900
    margin_l, margin_r, margin_t, margin_b = 120, 60, 110, 180
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    img = Image.new("RGB", (w, h), (250, 251, 253))
    d = ImageDraw.Draw(img)
    f_title = _load_font(size=40)
    f_text = _load_font(size=24)
    f_small = _load_font(size=20)

    d.text((margin_l, 28), title, fill=(25, 35, 52), font=f_title)

    x0, y0 = margin_l, margin_t
    x1, y1 = margin_l + plot_w, margin_t + plot_h
    d.line((x0, y1, x1, y1), fill=(40, 70, 120), width=3)
    d.line((x0, y0, x0, y1), fill=(40, 70, 120), width=3)

    # y ticks
    for t in np.linspace(y_min, y_max, 6):
        yy = y1 - (t - y_min) / max(1e-9, (y_max - y_min)) * plot_h
        d.line((x0 - 8, yy, x1, yy), fill=(224, 230, 240), width=1)
        d.text((30, yy - 10), f"{t:.2f}", fill=(60, 74, 98), font=f_small)

    n_cat = max(1, len(categories))
    n_series = max(1, len(series))
    group_w = plot_w / n_cat
    bar_w = (group_w * 0.72) / n_series
    start_offset = (group_w - n_series * bar_w) / 2
    palette = [(108, 176, 236), (31, 119, 180), (255, 170, 75), (70, 195, 143)]

    for i, cat in enumerate(categories):
        gx = x0 + i * group_w
        for j, (_, vals) in enumerate(series):
            val = float(vals[i]) if i < len(vals) else 0.0
            val = max(y_min, min(y_max, val))
            bx0 = gx + start_offset + j * bar_w
            bx1 = bx0 + bar_w * 0.92
            by1 = y1
            by0 = y1 - (val - y_min) / max(1e-9, (y_max - y_min)) * plot_h
            d.rectangle((bx0, by0, bx1, by1), fill=palette[j % len(palette)], outline=(50, 70, 100))

        # category label
        d.text((gx + group_w * 0.1, y1 + 18), cat, fill=(40, 54, 76), font=f_small)

    # legend
    lx, ly = margin_l, h - 78
    for j, (name, _) in enumerate(series):
        c = palette[j % len(palette)]
        d.rectangle((lx, ly, lx + 30, ly + 20), fill=c, outline=(50, 70, 100))
        d.text((lx + 40, ly - 1), name, fill=(40, 54, 76), font=f_text)
        lx += 270

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _draw_heatmap_png(
    path: Path,
    title: str,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    vmin: float,
    vmax: float,
) -> None:
    Image, ImageDraw, ImageFont = _get_pil_modules()
    if Image is None:
        raise RuntimeError("No plotting backend available. Install matplotlib or pillow.")

    n_r, n_c = matrix.shape
    w, h = 1600, 1100
    margin_l, margin_r, margin_t, margin_b = 300, 90, 120, 210
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    cw = plot_w / max(1, n_c)
    ch = plot_h / max(1, n_r)

    img = Image.new("RGB", (w, h), (250, 251, 253))
    d = ImageDraw.Draw(img)
    f_title = _load_font(size=38)
    f_text = _load_font(size=22)
    f_small = _load_font(size=18)

    d.text((margin_l, 34), title, fill=(25, 35, 52), font=f_title)

    def color(v):
        x = (float(v) - vmin) / max(1e-9, (vmax - vmin))
        x = max(0.0, min(1.0, x))
        # light -> dark blue
        r = int(235 - 175 * x)
        g = int(242 - 175 * x)
        b = int(252 - 125 * x)
        return (r, g, b)

    x0, y0 = margin_l, margin_t
    for i in range(n_r):
        for j in range(n_c):
            v = matrix[i, j]
            rx0 = x0 + j * cw
            ry0 = y0 + i * ch
            rx1 = rx0 + cw
            ry1 = ry0 + ch
            d.rectangle((rx0, ry0, rx1, ry1), fill=color(v), outline=(180, 190, 210))
            label = f"{v:.2f}" if vmax <= 1.0 else f"{int(v)}"
            d.text((rx0 + 8, ry0 + 6), label, fill=(26, 33, 48), font=f_small)

    # Row labels
    for i, rlab in enumerate(row_labels):
        ry = y0 + i * ch + ch * 0.35
        d.text((30, ry), str(rlab), fill=(40, 54, 76), font=f_text)

    # Col labels
    for j, clab in enumerate(col_labels):
        cx = x0 + j * cw + 4
        d.text((cx, y0 + plot_h + 14), str(clab), fill=(40, 54, 76), font=f_text)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build_linked_manifest(dataset_type: str, dataset_root: Path, fmc_human_dirs: list[str]) -> pd.DataFrame:
    files = [p for p in dataset_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise ValueError(f"No audio files found under: {dataset_root}")

    rows: list[dict] = []
    if dataset_type == "fma":
        for p in files:
            stem = p.stem
            rows.append(
                {
                    "audio_path": str(p.resolve()),
                    "caption": "",
                    "label": 0,
                    "source": "fma_linked",
                    "group_id": f"fma_track_{stem}",
                }
            )
    elif dataset_type == "fmc":
        human_set = {x.strip() for x in fmc_human_dirs if x.strip()}
        for p in files:
            rel = p.relative_to(dataset_root)
            source = rel.parts[0] if len(rel.parts) > 1 else p.parent.name
            label = 0 if source in human_set else 1
            rows.append(
                {
                    "audio_path": str(p.resolve()),
                    "caption": "",
                    "label": int(label),
                    "source": source,
                    "group_id": f"fmc_track_{p.stem}",
                }
            )
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return pd.DataFrame(rows)


def merge_manifests(old_manifest: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    old_df = pd.read_csv(old_manifest)
    for col, default in [("caption", ""), ("source", "unknown"), ("group_id", "unknown_group")]:
        if col not in old_df.columns:
            old_df[col] = default
    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
    return merged


def run_training(cmd: list[str], log_placeholder) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    all_lines: list[str] = []
    if proc.stdout is None:
        raise RuntimeError("Failed to open subprocess stdout pipe")
    for line in proc.stdout:
        cleaned = line.rstrip()
        all_lines.append(cleaned)
        lines.append(cleaned)
        lines = lines[-300:]
        log_placeholder.code("\n".join(lines))
    code = proc.wait()
    return code, "\n".join(all_lines)


def render_metrics_comparison(old_combined: dict, new_combined: dict) -> None:
    old_m = parse_model_test_metrics(old_combined)
    new_m = parse_model_test_metrics(new_combined)
    rows = []
    for key, label in [
        ("cnn_accuracy", "CNN Accuracy"),
        ("cnn_auroc", "CNN AUROC"),
        ("cnn_auprc", "CNN AUPRC"),
        ("clap_accuracy", "CLAP Accuracy"),
        ("clap_auroc", "CLAP AUROC"),
        ("clap_auprc", "CLAP AUPRC"),
        ("baseline_accuracy", "Baseline Accuracy"),
        ("baseline_auroc", "Baseline AUROC"),
        ("baseline_auprc", "Baseline AUPRC"),
    ]:
        rows.append({"metric": label, "old": old_m.get(key), "new": new_m.get(key)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def holdout_summary_to_df(holdout_summary: dict) -> pd.DataFrame:
    per_source = holdout_summary.get("per_source", [])
    rows = []
    for row in per_source:
        src = row.get("holdout_source", "unknown")
        cnn = row.get("cnn", {})
        baseline = row.get("baseline", {})
        rows.append(
            {
                "source": src,
                "cnn_acc_default": cnn.get("test_default", {}).get("accuracy"),
                "cnn_acc_calibrated": cnn.get("test_calibrated", {}).get("accuracy", cnn.get("test", {}).get("accuracy")),
                "cnn_auroc": cnn.get("test", {}).get("auroc"),
                "cnn_auprc": cnn.get("test", {}).get("auprc"),
                "baseline_acc": baseline.get("test", {}).get("accuracy"),
                "baseline_auroc": baseline.get("test", {}).get("auroc"),
                "baseline_auprc": baseline.get("test", {}).get("auprc"),
                "threshold": cnn.get("test", {}).get("threshold", cnn.get("calibration", {}).get("threshold")),
            }
        )
    return pd.DataFrame(rows)


def predict_fmc_source(audio_path: str, model_path: Path, top_k: int = 3) -> dict:
    blob = joblib.load(model_path)
    if not isinstance(blob, dict) or "model" not in blob or "label_encoder" not in blob:
        raise ValueError("FMC source model file must contain dict keys: model, label_encoder")

    clf = blob["model"]
    le = blob["label_encoder"]
    x = extract_features(audio_path).reshape(1, -1)
    probs = clf.predict_proba(x)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(le.inverse_transform([pred_idx])[0])

    order = np.argsort(probs)[::-1]
    top_k = max(1, min(int(top_k), len(order)))
    top_preds = [
        {"label": str(le.inverse_transform([int(i)])[0]), "prob": float(probs[int(i)])}
        for i in order[:top_k]
    ]
    return {
        "pred_label": pred_label,
        "pred_prob": float(probs[pred_idx]),
        "top_k": top_preds,
    }


def render_holdout_dashboard(holdout_summary: dict) -> None:
    if not holdout_summary.get("per_source"):
        st.info("No per-source holdout results found in summary.")
        return

    df = holdout_summary_to_df(holdout_summary)
    if df.empty:
        st.info("No valid holdout rows found.")
        return
    st.dataframe(df, use_container_width=True)

    st.markdown("**Baseline vs CNN (calibrated) Accuracy by Holdout Source**")
    acc_plot = df[["source", "baseline_acc", "cnn_acc_calibrated"]].set_index("source")
    st.bar_chart(acc_plot)

    st.markdown("**Threshold Calibration Effect (CNN Accuracy)**")
    cal_plot = df[["source", "cnn_acc_default", "cnn_acc_calibrated"]].set_index("source")
    st.bar_chart(cal_plot)

    st.markdown("**Calibrated Threshold by Holdout Source**")
    thr_plot = df[["source", "threshold"]].set_index("source")
    st.bar_chart(thr_plot)


def render_fmc_confusion_dashboard(fmc_metrics: dict) -> None:
    classes = fmc_metrics.get("classes", [])
    cm = fmc_metrics.get("test", {}).get("confusion_matrix")
    if not classes or cm is None:
        st.info("No FMC confusion matrix found in metrics.")
        return

    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    st.markdown("**FMC Source Logistic Regression Test Confusion Matrix**")
    st.dataframe(cm_df, use_container_width=True)

    # Normalized by true class (row) for readability.
    row_sums = cm_df.sum(axis=1).replace(0, 1)
    cm_norm = cm_df.div(row_sums, axis=0)
    st.markdown("**Normalized Confusion Matrix (row-wise)**")
    st.dataframe(cm_norm.round(3), use_container_width=True)


def render_clap_dashboard(clap_metrics: dict) -> None:
    if not clap_metrics:
        st.info("No CLAP metrics found.")
        return

    val = clap_metrics.get("val", {})
    test = clap_metrics.get("test", {})
    if not val and not test:
        st.info("CLAP metrics JSON does not contain val/test blocks.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("CLAP Test AUROC", f"{float(test.get('auroc', float('nan'))):.4f}" if "auroc" in test else "n/a")
    c2.metric("CLAP Test AUPRC", f"{float(test.get('auprc', float('nan'))):.4f}" if "auprc" in test else "n/a")
    c3.metric("CLAP Test Accuracy", f"{float(test.get('accuracy', float('nan'))):.4f}" if "accuracy" in test else "n/a")

    rows = [
        {"split": "val", "accuracy": val.get("accuracy"), "auroc": val.get("auroc"), "auprc": val.get("auprc")},
        {"split": "test", "accuracy": test.get("accuracy"), "auroc": test.get("auroc"), "auprc": test.get("auprc")},
    ]
    df = pd.DataFrame(rows).set_index("split")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df)

    dropped = clap_metrics.get("dropped_corrupt", {})
    if dropped:
        st.caption(
            "Dropped corrupt in CLAP embedding: "
            f"train={dropped.get('train', 0)}, val={dropped.get('val', 0)}, test={dropped.get('test', 0)}"
        )


def export_training_dashboard_pngs(
    holdout_summary: dict,
    fmc_metrics: dict,
    clap_metrics: dict,
    out_dir: Path,
    before_combined: dict | None = None,
    after_combined: dict | None = None,
) -> list[Path]:
    plt = _get_matplotlib_pyplot()
    use_pil_fallback = plt is None

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if holdout_summary and holdout_summary.get("per_source"):
        df = holdout_summary_to_df(holdout_summary)
        if not df.empty:
            sources = df["source"].astype(str).tolist()
            x = np.arange(len(sources))
            width = 0.35

            # Baseline vs CNN calibrated
            p = out_dir / "holdout_baseline_vs_cnn.png"
            if use_pil_fallback:
                _draw_grouped_bar_png(
                    p,
                    "Baseline vs CNN (calibrated) Accuracy by Holdout Source",
                    sources,
                    [
                        ("baseline_acc", df["baseline_acc"].astype(float).to_numpy()),
                        ("cnn_acc_calibrated", df["cnn_acc_calibrated"].astype(float).to_numpy()),
                    ],
                    y_min=0.0,
                    y_max=1.0,
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(x - width / 2, df["baseline_acc"].astype(float).to_numpy(), width, label="baseline_acc")
                ax.bar(x + width / 2, df["cnn_acc_calibrated"].astype(float).to_numpy(), width, label="cnn_acc_calibrated")
                ax.set_title("Baseline vs CNN (calibrated) Accuracy by Holdout Source")
                ax.set_xticks(x)
                ax.set_xticklabels(sources, rotation=45, ha="right")
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0.0, 1.0)
                ax.legend()
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

            # Threshold calibration effect
            p = out_dir / "threshold_calibration_effect.png"
            if use_pil_fallback:
                _draw_grouped_bar_png(
                    p,
                    "Threshold Calibration Effect (CNN Accuracy)",
                    sources,
                    [
                        ("cnn_acc_default", df["cnn_acc_default"].astype(float).to_numpy()),
                        ("cnn_acc_calibrated", df["cnn_acc_calibrated"].astype(float).to_numpy()),
                    ],
                    y_min=0.0,
                    y_max=1.0,
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(x - width / 2, df["cnn_acc_default"].astype(float).to_numpy(), width, label="cnn_acc_default")
                ax.bar(x + width / 2, df["cnn_acc_calibrated"].astype(float).to_numpy(), width, label="cnn_acc_calibrated")
                ax.set_title("Threshold Calibration Effect (CNN Accuracy)")
                ax.set_xticks(x)
                ax.set_xticklabels(sources, rotation=45, ha="right")
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0.0, 1.0)
                ax.legend()
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

            # Threshold per source
            p = out_dir / "holdout_thresholds.png"
            if use_pil_fallback:
                _draw_grouped_bar_png(
                    p,
                    "Calibrated Threshold by Holdout Source",
                    sources,
                    [("threshold", df["threshold"].astype(float).to_numpy())],
                    y_min=0.0,
                    y_max=1.0,
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(x, df["threshold"].astype(float).to_numpy(), width=0.55)
                ax.set_title("Calibrated Threshold by Holdout Source")
                ax.set_xticks(x)
                ax.set_xticklabels(sources, rotation=45, ha="right")
                ax.set_ylabel("Threshold")
                ax.set_ylim(0.0, 1.0)
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

    if fmc_metrics:
        classes = fmc_metrics.get("classes", [])
        cm = fmc_metrics.get("test", {}).get("confusion_matrix")
        if classes and cm is not None:
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)

            p = out_dir / "fmc_confusion_matrix.png"
            if use_pil_fallback:
                _draw_heatmap_png(
                    p,
                    "FMC Source Test Confusion Matrix",
                    cm_df.values.astype(float),
                    classes,
                    classes,
                    vmin=0.0,
                    vmax=float(np.max(cm_df.values) if np.max(cm_df.values) > 0 else 1.0),
                )
            else:
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(cm_df.values, aspect="auto")
                ax.set_title("FMC Source Test Confusion Matrix")
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha="right")
                ax.set_yticks(range(len(classes)))
                ax.set_yticklabels(classes)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

            cm_norm = cm_df.div(cm_df.sum(axis=1).replace(0, 1), axis=0)
            p = out_dir / "fmc_confusion_matrix_normalized.png"
            if use_pil_fallback:
                _draw_heatmap_png(
                    p,
                    "FMC Source Normalized Confusion Matrix (Row-wise)",
                    cm_norm.values.astype(float),
                    classes,
                    classes,
                    vmin=0.0,
                    vmax=1.0,
                )
            else:
                fig, ax = plt.subplots(figsize=(8, 7))
                im = ax.imshow(cm_norm.values, aspect="auto", vmin=0.0, vmax=1.0)
                ax.set_title("FMC Source Normalized Confusion Matrix (Row-wise)")
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha="right")
                ax.set_yticks(range(len(classes)))
                ax.set_yticklabels(classes)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

    if clap_metrics:
        val = clap_metrics.get("val", {})
        test = clap_metrics.get("test", {})
        if val or test:
            rows = [
                {"split": "val", "accuracy": val.get("accuracy"), "auroc": val.get("auroc"), "auprc": val.get("auprc")},
                {"split": "test", "accuracy": test.get("accuracy"), "auroc": test.get("auroc"), "auprc": test.get("auprc")},
            ]
            df = pd.DataFrame(rows).set_index("split").fillna(0.0)
            p = out_dir / "clap_val_test_metrics.png"
            if use_pil_fallback:
                _draw_grouped_bar_png(
                    p,
                    "CLAP Metrics: Validation vs Test",
                    df.columns.tolist(),
                    [
                        ("val", df.loc["val"].to_numpy(dtype=float)),
                        ("test", df.loc["test"].to_numpy(dtype=float)),
                    ],
                    y_min=0.0,
                    y_max=1.0,
                )
            else:
                x = np.arange(len(df.columns))
                width = 0.35
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.bar(x - width / 2, df.loc["val"].to_numpy(dtype=float), width, label="val")
                ax.bar(x + width / 2, df.loc["test"].to_numpy(dtype=float), width, label="test")
                ax.set_xticks(x)
                ax.set_xticklabels(df.columns.tolist())
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel("Score")
                ax.set_title("CLAP Metrics: Validation vs Test")
                ax.legend()
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

    if before_combined and after_combined:
        b = cnn_metrics_for_comparison(before_combined)
        a = cnn_metrics_for_comparison(after_combined)
        metric_order = [
            ("acc_default", "acc_default"),
            ("acc_calibrated", "acc_calibrated"),
            ("auroc", "auroc"),
            ("auprc", "auprc"),
        ]
        cats: list[str] = []
        bvals: list[float] = []
        avals: list[float] = []
        for key, label in metric_order:
            bv = b.get(key)
            av = a.get(key)
            if bv is None or av is None:
                continue
            cats.append(label)
            bvals.append(float(bv))
            avals.append(float(av))

        if cats:
            p = out_dir / "cnn_before_after_adaptation.png"
            ymax = max(1.0, max(bvals + avals) + 0.05)
            if use_pil_fallback:
                _draw_grouped_bar_png(
                    p,
                    "CNN Performance: Before vs After Adaptation",
                    cats,
                    [("before", np.array(bvals)), ("after", np.array(avals))],
                    y_min=0.0,
                    y_max=float(ymax),
                )
            else:
                x = np.arange(len(cats))
                width = 0.35
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.bar(x - width / 2, np.array(bvals), width, label="before")
                ax.bar(x + width / 2, np.array(avals), width, label="after")
                ax.set_xticks(x)
                ax.set_xticklabels(cats)
                ax.set_ylabel("Score")
                ax.set_ylim(0.0, float(ymax))
                ax.set_title("CNN Performance: Before vs After Adaptation")
                ax.legend()
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
            saved.append(p)

    return saved


def page_predict() -> None:
    st.subheader("Predict AI Probability")
    st.caption("Upload one WAV/MP3 file and get AI-vs-human probability from saved models.")

    # Predict-tab defaults/presets in session state.
    defaults_state = {
        "pred_fusion_mode": "cnn_only",
        "pred_cnn_weight": 0.8,
        "pred_cnn_eval_clips": 7,
        "pred_cnn_agg": "max",
        "pred_fallback_threshold": 0.5,
        "pred_use_threshold_json": True,
    }
    for k, v in defaults_state.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.button("Enable High Recall Mode"):
        st.session_state["pred_fusion_mode"] = "cnn_only"
        st.session_state["pred_cnn_weight"] = 0.9
        st.session_state["pred_cnn_eval_clips"] = 15
        st.session_state["pred_cnn_agg"] = "max"
        st.session_state["pred_fallback_threshold"] = 0.4
        st.session_state["pred_use_threshold_json"] = False
        st.success("High Recall Mode enabled: CNN-only, 15 clips, max-agg, threshold 0.40, threshold.json disabled.")

    run_dir_text = st.text_input(
        "Run Directory",
        value="artifacts/all",
        help="Path to a training output directory containing cnn/, baseline/, combined_metrics.json",
    )
    run_dir = Path(run_dir_text)
    defaults = find_default_paths(run_dir)

    cnn_model = Path(st.text_input("CNN Model Path", value=str(defaults["cnn_model"])).strip())
    baseline_model = Path(st.text_input("Baseline Model Path", value=str(defaults["baseline_model"])).strip())
    clap_model = Path(
        st.text_input(
            "CLAP Audio Model Path (optional, no captions required)",
            value=str(run_dir / "alignment" / "alignment_logreg.joblib"),
        ).strip()
    )
    threshold_json = Path(st.text_input("Threshold JSON Path", value=str(defaults["threshold_json"])).strip())
    use_threshold_json = st.checkbox(
        "Use threshold.json (if exists)",
        value=bool(st.session_state["pred_use_threshold_json"]),
        key="pred_use_threshold_json",
    )
    fmc_source_model = Path(
        st.text_input(
            "FMC Source Model Path (optional, used when prediction is AI)",
            value="",
            help="Path to fmc_source_logreg.joblib; leave blank to skip source attribution",
        ).strip()
    )
    st.caption(
        "Model status: "
        f"CNN={'OK' if cnn_model.is_file() else 'MISSING'} | "
        f"CLAP={'OK' if clap_model.is_file() else 'MISSING'} | "
        f"Baseline={'OK' if baseline_model.is_file() else 'MISSING'}"
    )
    fmc_top_k = int(st.number_input("FMC Source Top-K", value=3, min_value=1, max_value=10, step=1))
    fusion_options = ["cnn_only", "clap_only", "weighted_cnn_clap"]
    if st.session_state["pred_fusion_mode"] not in fusion_options:
        st.session_state["pred_fusion_mode"] = "cnn_only"
    fusion_mode = st.selectbox(
        "Fusion Mode",
        options=fusion_options,
        index=fusion_options.index(st.session_state["pred_fusion_mode"]),
        help="Choose inference branch strategy for final probability",
        key="pred_fusion_mode",
    )
    cnn_weight = st.slider(
        "CNN Weight (weighted_cnn_clap)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["pred_cnn_weight"]),
        step=0.05,
        key="pred_cnn_weight",
    )
    sample_rate = st.number_input("Sample Rate", value=32000, step=1000)
    clip_seconds = st.number_input("Clip Seconds", value=10.0, step=1.0)
    cnn_eval_clips = int(
        st.number_input(
            "CNN Eval Clips (full-song windows)",
            value=int(st.session_state["pred_cnn_eval_clips"]),
            min_value=1,
            max_value=40,
            step=1,
            key="pred_cnn_eval_clips",
        )
    )
    cnn_agg = st.selectbox(
        "CNN Clip Aggregation",
        options=["max", "mean", "median"],
        index=["max", "mean", "median"].index(st.session_state["pred_cnn_agg"]),
        key="pred_cnn_agg",
    )
    fallback_threshold = st.slider(
        "Fallback Threshold (if no threshold.json)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["pred_fallback_threshold"]),
        key="pred_fallback_threshold",
    )

    uploaded = st.file_uploader("Audio File (wav/mp3/flac/ogg/m4a)", type=[x.lstrip(".") for x in AUDIO_EXTS])
    if st.button("Run Prediction", type="primary"):
        if uploaded is None:
            st.error("Please upload an audio file.")
            return

        suffix = Path(uploaded.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            temp_path = Path(tmp.name)

        try:
            scores: dict[str, float] = {}
            cnn_details: dict = {}
            clap_error: str | None = None
            if cnn_model.is_file():
                cnn_out = predict_cnn_with_details(
                    audio_path=str(temp_path),
                    model_path=cnn_model,
                    sample_rate=int(sample_rate),
                    clip_seconds=float(clip_seconds),
                    num_eval_clips=max(1, int(cnn_eval_clips)),
                    agg=str(cnn_agg),
                )
                scores["cnn_prob"] = float(cnn_out["cnn_prob"])
                cnn_details = cnn_out
            if baseline_model.is_file():
                scores["baseline_prob"] = predict_baseline(
                    audio_path=str(temp_path),
                    model_path=baseline_model,
                )
            if clap_model.is_file():
                try:
                    scores["clap_prob"] = predict_clap_audio_prob(
                        audio_path=str(temp_path),
                        model_path=clap_model,
                    )
                except Exception as e:
                    clap_error = str(e)
                    st.warning(f"CLAP prediction failed: {e}")
            if not scores:
                st.error("No valid model found. Set a CNN and/or baseline model path.")
                return

            threshold = float(fallback_threshold)
            if use_threshold_json and threshold_json.is_file():
                data = load_json_if_exists(threshold_json)
                threshold = float(data.get("threshold", threshold))

            cnn_prob = scores.get("cnn_prob")
            clap_prob = scores.get("clap_prob")

            if fusion_mode == "cnn_only":
                if cnn_prob is None:
                    st.error("`cnn_only` selected but CNN model/path is missing.")
                    return
                final_prob = float(cnn_prob)
            elif fusion_mode == "clap_only":
                if clap_prob is None:
                    st.error("`clap_only` selected but CLAP model/path is missing.")
                    return
                final_prob = float(clap_prob)
            elif fusion_mode == "weighted_cnn_clap":
                if cnn_prob is None or clap_prob is None:
                    if clap_error is not None:
                        st.error(f"`weighted_cnn_clap` unavailable because CLAP failed: {clap_error}")
                    else:
                        st.error("`weighted_cnn_clap` requires both CNN and CLAP models.")
                    return
                final_prob = float(cnn_weight * cnn_prob + (1.0 - cnn_weight) * clap_prob)
            else:
                final_prob = float(np.mean(list(scores.values())))

            pred_label = int(final_prob >= threshold)
            pred_name = "ai_generated" if pred_label == 1 else "human"

            st.success(f"Prediction: {pred_name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Probability (AI)", f"{final_prob:.4f}")
            c2.metric("Threshold", f"{threshold:.4f}")
            c3.metric("Label", pred_name)
            c4.metric("Fusion", fusion_mode)
            st.json(scores)
            if "cnn_prob" in scores and "clap_prob" in scores:
                st.markdown("#### CNN vs CLAP (AI probability)")
                cmp_df = pd.DataFrame(
                    [
                        {"branch": "cnn", "prob_ai": float(scores["cnn_prob"])},
                        {"branch": "clap_audio", "prob_ai": float(scores["clap_prob"])},
                    ]
                )
                st.dataframe(cmp_df, use_container_width=True)
            if cnn_details.get("cnn_clip_probs"):
                clip_df = pd.DataFrame(
                    {
                        "clip_idx": list(range(len(cnn_details["cnn_clip_probs"]))),
                        "cnn_prob": cnn_details["cnn_clip_probs"],
                    }
                )
                st.markdown("#### CNN Per-Clip Scores")
                st.line_chart(clip_df.set_index("clip_idx"))

            if cnn_prob is not None and clap_prob is not None:
                cnn_label = int(cnn_prob >= threshold)
                clap_label = int(clap_prob >= threshold)
                if cnn_label != clap_label:
                    st.warning(
                        "Branch disagreement: CNN and CLAP predict different labels at current threshold."
                    )

            should_run_source = (
                pred_label == 1
                or (cnn_prob is not None and cnn_prob >= threshold)
                or (clap_prob is not None and clap_prob >= threshold)
            )
            if should_run_source:
                st.markdown("### FMC Source Prediction")
                if fmc_source_model.is_file():
                    try:
                        fmc_pred = predict_fmc_source(
                            audio_path=str(temp_path),
                            model_path=fmc_source_model,
                            top_k=fmc_top_k,
                        )
                        st.metric("Predicted FMC Source", fmc_pred["pred_label"])
                        topk_df = pd.DataFrame(fmc_pred["top_k"])
                        st.dataframe(topk_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not run FMC source prediction: {e}")
                else:
                    st.info("FMC source model not found; set a valid model path to enable source prediction.")
        finally:
            temp_path.unlink(missing_ok=True)


def page_further_training() -> None:
    st.subheader("Further Training")
    st.caption("Continue training from an existing run and compare old vs new metrics.")

    old_run_dir_text = st.text_input(
        "Old Run Directory",
        value="artifacts/all",
        help="Path to the existing training run you want to continue from",
    )
    old_run_dir = Path(old_run_dir_text)
    old_paths = find_default_paths(old_run_dir)
    old_combined = load_json_if_exists(old_paths["combined_metrics"])
    old_manifest_str = old_combined.get("manifest") if old_combined else None
    old_manifest = Path(old_manifest_str) if old_manifest_str else None

    if old_combined:
        st.write("Current (old) metrics:")
        st.json(parse_model_test_metrics(old_combined))
    else:
        st.warning(f"No old combined metrics found at: {old_paths['combined_metrics']}")

    st.markdown("### Dashboard")
    holdout_summary_path = Path(
        st.text_input(
            "Holdout Summary JSON",
            value="",
            help="Path to holdout_suite/holdout_summary.json",
        )
    )
    fmc_metrics_path = Path(
        st.text_input(
            "FMC Source Metrics JSON",
            value="",
            help="Path to fmc_source_logreg/metrics.json",
        )
    )
    clap_metrics_path = Path(
        st.text_input(
            "CLAP Metrics JSON",
            value="",
            help="Path to clap_audio_only/metrics.json",
        )
    )
    before_combined_path = Path(
        st.text_input(
            "CNN Before (combined_metrics.json)",
            value=str(old_paths["combined_metrics"]),
        )
    )
    after_combined_path = Path(
        st.text_input(
            "CNN After (combined_metrics.json)",
            value="",
            help="Path to a newer run's combined_metrics.json for comparison",
        )
    )

    c_dash1, c_dash2, c_dash3 = st.columns(3)
    with c_dash1:
        st.markdown("#### Holdout Dashboard")
        holdout_summary = load_json_if_exists(holdout_summary_path)
        if holdout_summary:
            render_holdout_dashboard(holdout_summary)
        else:
            st.info(f"No holdout summary found at: {holdout_summary_path}")
    with c_dash2:
        st.markdown("#### FMC Source Dashboard")
        fmc_metrics = load_json_if_exists(fmc_metrics_path)
        if fmc_metrics:
            render_fmc_confusion_dashboard(fmc_metrics)
        else:
            st.info(f"No FMC source metrics found at: {fmc_metrics_path}")
    with c_dash3:
        st.markdown("#### CLAP Dashboard")
        clap_metrics = load_json_if_exists(clap_metrics_path)
        if clap_metrics:
            render_clap_dashboard(clap_metrics)
        else:
            st.info(f"No CLAP metrics found at: {clap_metrics_path}")

    st.markdown("#### CNN Before vs After (Adaptation)")
    before_combined = load_json_if_exists(before_combined_path)
    after_combined = load_json_if_exists(after_combined_path)
    if before_combined and after_combined:
        b = cnn_metrics_for_comparison(before_combined)
        a = cnn_metrics_for_comparison(after_combined)
        rows = []
        for key in ["acc_default", "acc_calibrated", "auroc", "auprc"]:
            rows.append({"metric": key, "before": b.get(key), "after": a.get(key)})
        cmp_df = pd.DataFrame(rows).set_index("metric")
        st.dataframe(cmp_df, use_container_width=True)
        st.bar_chart(cmp_df)
    else:
        st.info("Provide valid before/after combined_metrics.json files to render adaptation comparison.")

    st.markdown("#### Export Dashboard Plots (PNG)")
    export_dir = Path(
        st.text_input(
            "PNG Export Directory",
            value="report/figures",
            help="Directory where dashboard PNG plots will be saved",
        ).strip()
    )
    if st.button("Export Dashboard Plots (PNG)"):
        try:
            saved = export_training_dashboard_pngs(
                holdout_summary=holdout_summary,
                fmc_metrics=fmc_metrics,
                clap_metrics=clap_metrics,
                out_dir=export_dir,
                before_combined=before_combined,
                after_combined=after_combined,
            )
        except Exception as e:
            st.error(f"Export failed: {e}")
        else:
            if not saved:
                st.warning("No plots were exported. Check that the JSON metric files are present and valid.")
            else:
                st.success(f"Exported {len(saved)} plot(s) to: {export_dir}")
                for p in saved:
                    st.code(str(p))

    st.markdown("---")
    data_mode = st.radio("Data Source", options=["existing_manifest", "link_dataset_folder"], horizontal=True)

    manifest_path: Path | None = None
    linked_df: pd.DataFrame | None = None

    if data_mode == "existing_manifest":
        manifest_text = st.text_input(
            "Training Manifest Path",
            value=str(old_manifest) if old_manifest else "",
        )
        if manifest_text.strip():
            manifest_path = Path(manifest_text.strip())
    else:
        dataset_type = st.selectbox("Linked Dataset Type", options=["fmc", "fma"], index=0)
        dataset_root_text = st.text_input("Dataset Root Path")
        fmc_humans_text = st.text_input("FMC Human Folders (comma-separated)", value="MusicCaps")
        if dataset_root_text.strip():
            dataset_root = Path(dataset_root_text.strip())
            if dataset_root.exists():
                try:
                    linked_df = build_linked_manifest(
                        dataset_type=dataset_type,
                        dataset_root=dataset_root,
                        fmc_human_dirs=[x.strip() for x in fmc_humans_text.split(",")],
                    )
                    st.info(f"Linked rows discovered: {len(linked_df)}")
                except Exception as e:
                    st.error(f"Failed to link dataset: {e}")
            else:
                st.error(f"Dataset root does not exist: {dataset_root}")

    merge_with_old = st.checkbox("Merge linked/new data with old manifest", value=True)
    continue_from_old_cnn = st.checkbox("Continue from old CNN checkpoint", value=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_text = st.text_input(
        "New Output Directory",
        value=f"artifacts/ui_runs/run_{now}",
        help="Where to write the new training run's outputs",
    )
    output_dir = Path(output_dir_text)

    c1, c2, c3 = st.columns(3)
    group_col = c1.text_input("Group Column", value="group_id")
    seed = int(c2.number_input("Seed", value=42, step=1))
    skip_alignment = c3.checkbox("Skip Alignment", value=True)

    c4, c5, c6 = st.columns(3)
    cnn_epochs = int(c4.number_input("CNN Epochs", value=4, step=1))
    cnn_batch_size = int(c5.number_input("CNN Batch Size", value=16, step=1))
    cnn_lr = float(c6.number_input("CNN LR", value=0.001, step=0.0005, format="%.4f"))

    if st.button("Start Further Training", type="primary"):
        output_dir.mkdir(parents=True, exist_ok=True)

        effective_manifest: Path | None = None
        if data_mode == "existing_manifest":
            if manifest_path is None:
                st.error("Provide a training manifest path.")
                return
            if not manifest_path.exists():
                st.error(f"Training manifest not found: {manifest_path}")
                return
            effective_manifest = manifest_path
        else:
            if linked_df is None:
                st.error("Link a valid dataset folder first.")
                return
            if merge_with_old and old_manifest and old_manifest.exists():
                merged = merge_manifests(old_manifest=old_manifest, new_df=linked_df)
            else:
                merged = linked_df
            effective_manifest = output_dir / "training_manifest.csv"
            merged.to_csv(effective_manifest, index=False)

        cmd = [
            sys.executable,
            "-m",
            "aidetector.train_all",
            "--manifest",
            str(effective_manifest),
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--cnn_epochs",
            str(cnn_epochs),
            "--cnn_batch_size",
            str(cnn_batch_size),
            "--cnn_lr",
            str(cnn_lr),
            "--cnn_num_workers",
            "0",
            "--calibration_metric",
            "f1",
        ]
        if group_col.strip():
            cmd.extend(["--group_col", group_col.strip()])
        if skip_alignment:
            cmd.append("--skip_alignment")
        if continue_from_old_cnn and old_paths["cnn_model"].exists():
            cmd.extend(["--cnn_init_model", str(old_paths["cnn_model"])])

        st.write("Running command:")
        st.code(" ".join(cmd))
        log_placeholder = st.empty()
        code, log_text = run_training(cmd=cmd, log_placeholder=log_placeholder)
        (output_dir / "train.log").write_text(log_text, encoding="utf-8")

        if code != 0:
            st.error(f"Training failed with exit code {code}. See {output_dir / 'train.log'}")
            return

        st.success("Training finished.")
        new_combined = load_json_if_exists(output_dir / "combined_metrics.json")
        if not new_combined:
            st.error(f"No combined_metrics.json found in: {output_dir}")
            return

        st.markdown("### Old vs New Metrics")
        render_metrics_comparison(old_combined=old_combined, new_combined=new_combined)
        st.markdown("### New Metrics JSON")
        st.json(new_combined)


def main() -> None:
    st.set_page_config(page_title="AI Music Detector UI", layout="wide")
    st.title("AI Music Detector")
    tab_predict, tab_train = st.tabs(["Predict", "Further Training"])
    with tab_predict:
        page_predict()
    with tab_train:
        page_further_training()


if __name__ == "__main__":
    main()
