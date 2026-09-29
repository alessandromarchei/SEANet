#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize flat AV-HuBERT embeddings:\n"
            "  idXXXXX_VIDEOID_UTT.npy\n"
            "into:\n"
            "  idXXXXX/VIDEOID/UTT.npy"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing the flat .npy files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show operations without modifying files.",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move.",
    )

    return parser.parse_args()


def parse_filename(path: Path):
    """
    Example:

        id00413_Q6JbpwUtDqE_00227.npy

    becomes:

        speaker = id00413
        video   = Q6JbpwUtDqE
        utt     = 00227

    We split the first and last '_' so underscores
    inside the video ID are preserved.
    """

    stem = path.stem

    try:
        speaker, remainder = stem.split("_", 1)
        video, utterance = remainder.rsplit("_", 1)

    except ValueError:
        raise ValueError(
            f"Cannot parse filename: {path.name}"
        )

    if not speaker:
        raise ValueError(
            f"Missing speaker in {path.name}"
        )

    if not video:
        raise ValueError(
            f"Missing video ID in {path.name}"
        )

    if not utterance:
        raise ValueError(
            f"Missing utterance ID in {path.name}"
        )

    return speaker, video, utterance


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()

    if not root.is_dir():
        raise RuntimeError(
            f"Directory does not exist: {root}"
        )

    # Only files directly under root.
    # Already reorganized files are therefore ignored.
    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix == ".npy"
    )

    print()
    print("=" * 72)
    print("AV-HUBERT EMBEDDING REORGANIZER")
    print("=" * 72)
    print(f"Root       : {root}")
    print(f"Flat files : {len(files):,}")
    print(f"Operation  : {'COPY' if args.copy else 'MOVE'}")
    print(f"Dry run    : {args.dry_run}")
    print()

    processed = 0
    skipped = 0
    invalid = 0

    for src in tqdm(
        files,
        desc="Reorganizing",
        unit="file",
    ):
        try:
            speaker, video, utterance = parse_filename(src)

        except ValueError as exc:
            tqdm.write(f"[INVALID] {exc}")
            invalid += 1
            continue

        # --------------------------------------------------
        # Desired structure:
        #
        # root/
        #   id00413/
        #     Q6JbpwUtDqE/
        #       00227.npy
        # --------------------------------------------------

        dst_dir = (
            root
            / speaker
            / video
        )

        dst = (
            dst_dir
            / f"{utterance}.npy"
        )

        if dst.exists():
            tqdm.write(
                f"[SKIP] Already exists: {dst}"
            )
            skipped += 1
            continue

        if args.dry_run:
            tqdm.write(
                f"{src.name} -> "
                f"{speaker}/{video}/{utterance}.npy"
            )
            processed += 1
            continue

        dst_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

        processed += 1

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Processed : {processed:,}")
    print(f"Skipped   : {skipped:,}")
    print(f"Invalid   : {invalid:,}")

    remaining = sum(
        1
        for p in root.iterdir()
        if p.is_file() and p.suffix == ".npy"
    )

    print(f"Flat files remaining: {remaining:,}")


if __name__ == "__main__":
    main()