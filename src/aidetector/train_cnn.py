from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aidetector.cnn_detector import CNNArtifactDetector
from aidetector.data import AudioClipDataset, load_manifest, split_df, split_df_grouped, split_df_holdout_ai_source
from aidetector.device import get_best_device
from aidetector.metrics import binary_metrics, optimize_threshold


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    skipped = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["waveform"].to(device)
        y = batch["label"].to(device)
        bad = batch["bad"].to(device).bool()
        valid = ~bad
        if valid.sum().item() == 0:
            skipped += int(bad.numel())
            continue

        optimizer.zero_grad(set_to_none=True)
        logits = model(x[valid])
        loss = criterion(logits, y[valid])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if skipped > 0:
        print(f"Skipped {skipped} corrupt samples during training epoch")
    if not losses:
        raise RuntimeError("No valid samples in this epoch. Check manifest/audio quality.")
    return float(np.mean(losses))


@torch.inference_mode()
def collect_scores(model, loader, device, desc: str = "eval"):
    model.eval()
    all_scores = []
    all_labels = []
    skipped = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["waveform"].to(device)
        y = batch["label"].cpu().numpy()
        bad = batch["bad"].cpu().numpy().astype(bool)
        valid = ~bad
        if not valid.any():
            skipped += int(bad.sum())
            continue

        logits = model(x[torch.from_numpy(valid).to(device)])
        probs = torch.sigmoid(logits).cpu().numpy()

        all_scores.append(probs)
        all_labels.append(y[valid])

    if not all_scores:
        return None, None, skipped

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels).astype(int)
    return y_true, y_score, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN artifact detector")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--init_model",
        type=Path,
        default=None,
        help="Optional path to existing cnn_best.pt for continued training",
    )
    parser.add_argument("--group_col", type=str, default=None, help="Optional leakage-control group column for splitting")
    parser.add_argument("--holdout_ai_source", type=str, default=None, help="Optional strict test: unseen AI source")
    parser.add_argument("--human_test_ratio", type=float, default=1.0, help="Human:AI ratio in holdout test set")
    parser.add_argument(
        "--calibration_metric",
        type=str,
        default="f1",
        choices=["f1", "balanced_accuracy", "youden_j"],
        help="Metric used to calibrate decision threshold on validation set",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    train_ds = AudioClipDataset(splits.train, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds, train=True)
    val_ds = AudioClipDataset(splits.val, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds, train=False)
    test_ds = AudioClipDataset(splits.test, sample_rate=args.sample_rate, clip_seconds=args.clip_seconds, train=False)

    device = get_best_device()
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = CNNArtifactDetector(sample_rate=args.sample_rate).to(device)
    if args.init_model is not None:
        if not args.init_model.exists():
            raise FileNotFoundError(f"init model not found: {args.init_model}")
        state = torch.load(args.init_model, map_location=device)
        model.load_state_dict(state)
        print(f"Initialized CNN weights from: {args.init_model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    y_train = splits.train["label"].to_numpy()
    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        y_val, val_score, val_skipped = collect_scores(model, val_loader, device, desc="val")
        if val_skipped > 0:
            print(f"Skipped {val_skipped} corrupt samples during validation")
        if y_val is None or val_score is None:
            val_metrics = {"accuracy": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
        else:
            val_metrics = binary_metrics(y_val, val_score, threshold=0.5)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_auroc={val_metrics['auroc']:.4f} val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            torch.save(model.state_dict(), args.output_dir / "cnn_best.pt")

    model.load_state_dict(torch.load(args.output_dir / "cnn_best.pt", map_location=device))
    y_val, val_score, val_skipped = collect_scores(model, val_loader, device, desc="val-best")
    y_test, test_score, test_skipped = collect_scores(model, test_loader, device, desc="test")
    if val_skipped > 0:
        print(f"Skipped {val_skipped} corrupt samples during validation(best)")
    if test_skipped > 0:
        print(f"Skipped {test_skipped} corrupt samples during testing")

    if y_val is None or val_score is None or y_test is None or test_score is None:
        raise RuntimeError("No valid validation/test samples available for final evaluation.")

    calibration = optimize_threshold(y_val, val_score, metric=args.calibration_metric)
    calibrated_threshold = float(calibration["threshold"])

    val_metrics_default = binary_metrics(y_val, val_score, threshold=0.5)
    val_metrics_calibrated = binary_metrics(y_val, val_score, threshold=calibrated_threshold)
    test_metrics_default = binary_metrics(y_test, test_score, threshold=0.5)
    test_metrics_calibrated = binary_metrics(y_test, test_score, threshold=calibrated_threshold)

    results = {
        "calibration": calibration,
        "val_default": val_metrics_default,
        "val_calibrated": val_metrics_calibrated,
        "test_default": test_metrics_default,
        "test_calibrated": test_metrics_calibrated,
        # Backward-compatible key expected by existing comparison snippets.
        "test": test_metrics_calibrated,
    }

    with open(args.output_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    np.save(args.output_dir / "val_scores.npy", val_score)
    np.save(args.output_dir / "test_scores.npy", test_score)
    np.save(args.output_dir / "val_labels.npy", y_val)
    np.save(args.output_dir / "test_labels.npy", y_test)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
