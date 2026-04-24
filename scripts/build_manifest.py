#!/usr/bin/env python3
import argparse
from pathlib import Path
import random

import pandas as pd

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def resolve_audio_path(path_value: str, root: Path | None) -> str:
    p = Path(path_value)
    if p.is_absolute() or root is None:
        return str(p)
    return str((root / p).resolve())


def build_fmc_rows(fake_csv: Path, fake_root: Path | None) -> pd.DataFrame:
    df = pd.read_csv(fake_csv)

    if "audio_path" not in df.columns:
        candidates = ["path", "file", "filename", "audio"]
        match = [c for c in candidates if c in df.columns]
        if not match:
            raise ValueError("FakeMusicCaps CSV must contain `audio_path` or one of path/file/filename/audio")
        df = df.rename(columns={match[0]: "audio_path"})

    if "caption" not in df.columns:
        caption_candidates = ["text", "prompt", "description"]
        match = [c for c in caption_candidates if c in df.columns]
        if match:
            df = df.rename(columns={match[0]: "caption"})
        else:
            df["caption"] = ""

    if "label" not in df.columns:
        df["label"] = 1

    if "group_id" not in df.columns:
        group_candidates = ["artist_id", "creator", "generator", "model", "source", "id"]
        match = [c for c in group_candidates if c in df.columns]
        if match:
            df["group_id"] = "fmc_" + df[match[0]].astype(str)
        else:
            # Fallback: unique-ish group by filename stem.
            df["group_id"] = df["audio_path"].astype(str).map(lambda p: f"fmc_track_{Path(p).stem}")

    df = df[["audio_path", "caption", "label", "group_id"]].copy()
    df["audio_path"] = df["audio_path"].astype(str).map(lambda x: resolve_audio_path(x, fake_root))
    df["source"] = "fmc"
    return df


def build_fmc_rows_from_dirs(fake_root: Path, human_dirs: list[str]) -> pd.DataFrame:
    files = [p for p in fake_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise ValueError(f"No audio files found under {fake_root}")

    human_set = {h.strip() for h in human_dirs if h.strip()}
    rows = []
    for p in files:
        rel = p.relative_to(fake_root)
        source = rel.parts[0] if len(rel.parts) > 1 else p.parent.name
        label = 0 if source in human_set else 1
        stem = p.stem
        rows.append(
            {
                "audio_path": str(p.resolve()),
                "caption": "",
                "label": label,
                "group_id": f"fmc_track_{stem}",
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def build_fma_rows(
    fma_root: Path,
    fma_limit: int | None,
    seed: int,
    fma_tracks_csv: Path | None,
    fma_track_col: str,
    fma_artist_col: str,
) -> pd.DataFrame:
    files = [p for p in fma_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise ValueError(f"No audio files found under {fma_root}")

    rng = random.Random(seed)
    rng.shuffle(files)
    if fma_limit is not None:
        files = files[:fma_limit]

    out = pd.DataFrame(
        {
            "audio_path": [str(p.resolve()) for p in files],
            "caption": [""] * len(files),
            "label": [0] * len(files),
            "source": ["fma"] * len(files),
        }
    )
    out["track_id"] = out["audio_path"].map(lambda p: Path(p).stem)

    artist_map = {}
    if fma_tracks_csv is not None:
        meta = pd.read_csv(fma_tracks_csv)
        if fma_track_col in meta.columns and fma_artist_col in meta.columns:
            artist_map = dict(zip(meta[fma_track_col].astype(str), meta[fma_artist_col].astype(str)))
        else:
            print(
                f"Warning: {fma_tracks_csv} missing columns "
                f"`{fma_track_col}` and/or `{fma_artist_col}`. "
                "Falling back to per-track group_id."
            )

    if artist_map:
        out["group_id"] = out["track_id"].map(lambda tid: f"fma_artist_{artist_map.get(str(tid), 'unknown')}")
    else:
        # Fallback group by track id when artist metadata is unavailable.
        out["group_id"] = out["track_id"].map(lambda tid: f"fma_track_{tid}")

    return out.drop(columns=["track_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined manifest for prototype training")
    parser.add_argument("--fake_csv", type=Path, default=None, help="Optional CSV from FakeMusicCaps")
    parser.add_argument("--fake_root", type=Path, default=None, help="Base directory for fake audio if paths are relative")
    parser.add_argument(
        "--fake_human_dirs",
        type=str,
        default="MusicCaps",
        help="Comma-separated folder names inside fake_root treated as human (label=0) in no-CSV mode",
    )
    parser.add_argument("--fma_root", type=Path, required=True, help="Directory containing FMA audio files")
    parser.add_argument("--fma_limit", type=int, default=None, help="Optional cap on number of FMA files")
    parser.add_argument("--fma_tracks_csv", type=Path, default=None, help="Optional FMA metadata CSV with track->artist mapping")
    parser.add_argument("--fma_track_col", type=str, default="track_id")
    parser.add_argument("--fma_artist_col", type=str, default="artist_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.fake_csv is not None:
        fake_df = build_fmc_rows(args.fake_csv, args.fake_root)
    else:
        if args.fake_root is None:
            raise ValueError("Provide either --fake_csv or --fake_root")
        human_dirs = [x.strip() for x in args.fake_human_dirs.split(",")]
        fake_df = build_fmc_rows_from_dirs(args.fake_root, human_dirs)

    fma_df = build_fma_rows(
        args.fma_root,
        args.fma_limit,
        args.seed,
        args.fma_tracks_csv,
        args.fma_track_col,
        args.fma_artist_col,
    )

    full_df = pd.concat([fake_df, fma_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(args.output, index=False)

    print(f"Wrote manifest: {args.output}")
    print(full_df["label"].value_counts(dropna=False).to_string())
    print(full_df["source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
