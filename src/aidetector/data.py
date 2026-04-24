from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from aidetector.audio_io import load_audio


REQUIRED_COLUMNS = {"audio_path", "label"}
_MISSING_GROUP = "__missing_group__"


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    if "caption" not in df.columns:
        df["caption"] = ""

    df["audio_path"] = df["audio_path"].astype(str)
    df["label"] = df["label"].astype(int)
    df["caption"] = df["caption"].fillna("").astype(str)

    exists_mask = df["audio_path"].map(lambda p: Path(p).exists())
    dropped = int((~exists_mask).sum())
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing audio files")
    return df[exists_mask].reset_index(drop=True)


def split_df(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> SplitResult:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    rel_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=rel_val,
        random_state=seed,
        stratify=train_df["label"],
    )
    return SplitResult(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def split_df_grouped(
    df: pd.DataFrame,
    group_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1")
    if group_col not in df.columns:
        raise ValueError(f"group_col `{group_col}` not found in manifest")

    groups = df[group_col].fillna(_MISSING_GROUP).astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3:
        raise ValueError(
            f"Need at least 3 unique groups for train/val/test, found {unique_groups} in `{group_col}`"
        )

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter_test.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    rel_val = val_size / (1.0 - test_size)
    train_val_groups = train_val_df[group_col].fillna(_MISSING_GROUP).astype(str)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_groups))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if part["label"].nunique() < 2:
            raise ValueError(
                f"Grouped split produced single-class {name} set. "
                "Use more diverse groups or adjust split ratios."
            )

    return SplitResult(train=train_df, val=val_df, test=test_df)


def _sample_groups_to_target_rows(
    df: pd.DataFrame,
    group_col: str,
    target_rows: int,
    seed: int,
) -> pd.DataFrame:
    groups = df[group_col].fillna(_MISSING_GROUP).astype(str)
    group_sizes = df.assign(_g=groups).groupby("_g").size().reset_index(name="n")
    group_sizes = group_sizes.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    chosen: list[str] = []
    total = 0
    for _, row in group_sizes.iterrows():
        chosen.append(str(row["_g"]))
        total += int(row["n"])
        if total >= target_rows:
            break
    return df[groups.isin(chosen)]


def split_df_holdout_ai_source(
    df: pd.DataFrame,
    holdout_ai_source: str,
    group_col: str | None = None,
    val_size: float = 0.15,
    human_test_ratio: float = 1.0,
    seed: int = 42,
) -> SplitResult:
    if "source" not in df.columns:
        raise ValueError("Manifest must contain `source` column for holdout-ai-source splitting")
    if val_size <= 0 or val_size >= 0.5:
        raise ValueError("val_size must be in (0, 0.5)")
    if human_test_ratio <= 0:
        raise ValueError("human_test_ratio must be > 0")

    ai_test = df[(df["label"] == 1) & (df["source"] == holdout_ai_source)].copy()
    if ai_test.empty:
        raise ValueError(f"No AI rows found for holdout source `{holdout_ai_source}`")

    human_pool = df[df["label"] == 0].copy()
    if human_pool.empty:
        raise ValueError("No human rows found to build balanced holdout test set")

    target_human_rows = max(1, int(round(len(ai_test) * human_test_ratio)))
    if group_col and group_col in human_pool.columns:
        human_test = _sample_groups_to_target_rows(human_pool, group_col, target_human_rows, seed)
    else:
        n = min(target_human_rows, len(human_pool))
        human_test = human_pool.sample(n=n, random_state=seed)

    test_idx = ai_test.index.union(human_test.index)
    test_df = df.loc[test_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    remain_df = df.drop(index=test_idx).reset_index(drop=True)

    if remain_df["label"].nunique() < 2:
        raise ValueError("Remaining train/val pool has a single class after holdout split")

    if group_col and group_col in remain_df.columns:
        remain_groups = remain_df[group_col].fillna(_MISSING_GROUP).astype(str)
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter_val.split(remain_df, groups=remain_groups))
        train_df = remain_df.iloc[train_idx].reset_index(drop=True)
        val_df = remain_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(
            remain_df,
            test_size=val_size,
            random_state=seed,
            stratify=remain_df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if part["label"].nunique() < 2:
            raise ValueError(
                f"Holdout split produced single-class {name} set. "
                "Adjust holdout source/ratio or dataset composition."
            )

    return SplitResult(train=train_df, val=val_df, test=test_df)


def _to_mono_resampled(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform [channels, samples], got {waveform.shape}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def _fix_clip_length(waveform: torch.Tensor, num_samples: int, train: bool) -> torch.Tensor:
    curr = waveform.shape[-1]
    if curr == num_samples:
        return waveform

    if curr > num_samples:
        if train:
            start = torch.randint(0, curr - num_samples + 1, (1,)).item()
        else:
            start = (curr - num_samples) // 2
        return waveform[:, start : start + num_samples]

    padded = torch.zeros((1, num_samples), dtype=waveform.dtype)
    padded[:, :curr] = waveform
    return padded


class AudioClipDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 32000,
        clip_seconds: float = 10.0,
        train: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * clip_seconds)
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        path = row["audio_path"]
        bad = False
        try:
            waveform, sr = load_audio(path)
            waveform = _to_mono_resampled(waveform, sr, self.sample_rate)
            waveform = _fix_clip_length(waveform, self.num_samples, self.train)
        except Exception:
            # Keep dataloader/training resilient to a few corrupt audio files.
            waveform = torch.zeros((1, self.num_samples), dtype=torch.float32)
            bad = True

        return {
            "waveform": waveform,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "caption": str(row.get("caption", "")),
            "path": path,
            "bad": torch.tensor(bad, dtype=torch.bool),
        }
