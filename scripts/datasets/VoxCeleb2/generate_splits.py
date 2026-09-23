#!/usr/bin/env python3

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a speaker-independent train/validation split "
            "from an existing SEANet data_list.csv."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Original data_list.csv",
    )

    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output CSV, e.g. configs/data_list_2.csv",
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--val-speakers",
        type=int,
        help=(
            "Number of speakers reserved exclusively "
            "for validation."
        ),
    )

    group.add_argument(
        "--val-speaker-fraction",
        type=float,
        help=(
            "Fraction of speakers reserved exclusively "
            "for validation, e.g. 0.10."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--train-mixtures",
        type=int,
        default=None,
        help=(
            "Maximum number of resulting training mixtures. "
            "Example: 20000. Default: keep all eligible."
        ),
    )

    parser.add_argument(
        "--val-mixtures",
        type=int,
        default=None,
        help=(
            "Maximum number of resulting validation mixtures. "
            "Example: 5000. Default: keep all eligible."
        ),
    )

    parser.add_argument(
        "--pool-splits",
        nargs="+",
        default=["train", "val"],
        help=(
            "Input splits used as the train/val candidate pool. "
            "Default: train val"
        ),
    )

    parser.add_argument(
        "--test-split",
        default="test",
        help=(
            "Input split preserved unchanged as test "
            "(default: test)."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting output CSV.",
    )

    return parser.parse_args()


def load_rows(path):
    rows = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, 1):

            if not row:
                continue

            if len(row) < 10:
                raise ValueError(
                    f"{path}:{line_no}: malformed row "
                    f"with {len(row)} columns."
                )

            rows.append(row)

    return rows


def speakers_in_row(row):
    return {
        row[2].strip(),
        row[6].strip(),
    }


def utterances_in_row(row):
    return {
        (
            row[1].strip(),
            row[2].strip(),
            row[3].strip(),
        ),
        (
            row[5].strip(),
            row[6].strip(),
            row[7].strip(),
        ),
    }


def main():
    args = parse_args()

    if not args.input_csv.is_file():
        raise FileNotFoundError(
            args.input_csv
        )

    if args.output_csv.exists() and not args.force:
        raise FileExistsError(
            f"{args.output_csv} already exists. "
            f"Use --force to overwrite it."
        )

    if (
        args.val_speaker_fraction is not None
        and not (
            0.0
            < args.val_speaker_fraction
            < 1.0
        )
    ):
        raise ValueError(
            "--val-speaker-fraction must be between 0 and 1."
        )

    rng = random.Random(args.seed)

    rows = load_rows(args.input_csv)

    pool_split_set = set(
        args.pool_splits
    )

    pool_rows = [
        row
        for row in rows
        if row[0] in pool_split_set
    ]

    test_rows = [
        row
        for row in rows
        if row[0] == args.test_split
    ]

    other_rows = [
        row
        for row in rows
        if (
            row[0] not in pool_split_set
            and row[0] != args.test_split
        )
    ]

    if other_rows:
        raise RuntimeError(
            "Input contains splits not accounted for by "
            "--pool-splits or --test-split:\n"
            f"{sorted(set(r[0] for r in other_rows))}"
        )

    # ========================================================
    # Candidate speakers
    # ========================================================

    pool_speakers = set()

    for row in pool_rows:
        pool_speakers.update(
            speakers_in_row(row)
        )

    test_speakers = set()

    for row in test_rows:
        test_speakers.update(
            speakers_in_row(row)
        )

    # The source CSV itself should already satisfy this.
    preexisting_test_overlap = (
        pool_speakers
        & test_speakers
    )

    if preexisting_test_overlap:
        raise RuntimeError(
            "The candidate train/val speaker pool already "
            "overlaps with test speakers.\n"
            f"Overlap: {len(preexisting_test_overlap)} speakers."
        )

    speakers = sorted(
        pool_speakers
    )

    if args.val_speakers is not None:

        n_val_speakers = args.val_speakers

    else:

        n_val_speakers = round(
            len(speakers)
            * args.val_speaker_fraction
        )

    if not (
        1
        <= n_val_speakers
        < len(speakers)
    ):
        raise ValueError(
            f"Invalid validation speaker count: "
            f"{n_val_speakers}. "
            f"Available speakers: {len(speakers)}."
        )

    shuffled = speakers.copy()
    rng.shuffle(shuffled)

    val_speakers = set(
        shuffled[:n_val_speakers]
    )

    train_speakers = set(
        shuffled[n_val_speakers:]
    )

    assert not (
        train_speakers
        & val_speakers
    )

    # ========================================================
    # Assign mixtures
    # ========================================================

    train_candidates = []
    val_candidates = []
    cross_group = []

    for row in pool_rows:

        row_speakers = speakers_in_row(row)

        if row_speakers <= train_speakers:

            new_row = row.copy()
            new_row[0] = "train"

            train_candidates.append(
                new_row
            )

        elif row_speakers <= val_speakers:

            new_row = row.copy()
            new_row[0] = "val"

            val_candidates.append(
                new_row
            )

        else:

            cross_group.append(
                row
            )

    # ========================================================
    # Shuffle deterministically before optional subsampling
    # ========================================================

    rng.shuffle(train_candidates)
    rng.shuffle(val_candidates)

    if args.train_mixtures is not None:

        if args.train_mixtures > len(
            train_candidates
        ):
            raise RuntimeError(
                f"Requested {args.train_mixtures:,} "
                f"training mixtures, but only "
                f"{len(train_candidates):,} are eligible "
                f"with this speaker split."
            )

        train_candidates = (
            train_candidates[
                :args.train_mixtures
            ]
        )

    if args.val_mixtures is not None:

        if args.val_mixtures > len(
            val_candidates
        ):
            raise RuntimeError(
                f"Requested {args.val_mixtures:,} "
                f"validation mixtures, but only "
                f"{len(val_candidates):,} are eligible "
                f"with this speaker split."
            )

        val_candidates = (
            val_candidates[
                :args.val_mixtures
            ]
        )

    # ========================================================
    # Final checks
    # ========================================================

    final_train_speakers = set()
    final_val_speakers = set()
    final_test_speakers = set()

    final_train_utts = set()
    final_val_utts = set()
    final_test_utts = set()

    for row in train_candidates:
        final_train_speakers.update(
            speakers_in_row(row)
        )
        final_train_utts.update(
            utterances_in_row(row)
        )

    for row in val_candidates:
        final_val_speakers.update(
            speakers_in_row(row)
        )
        final_val_utts.update(
            utterances_in_row(row)
        )

    for row in test_rows:
        final_test_speakers.update(
            speakers_in_row(row)
        )
        final_test_utts.update(
            utterances_in_row(row)
        )

    checks = {
        "train/val speaker overlap":
            final_train_speakers
            & final_val_speakers,

        "train/test speaker overlap":
            final_train_speakers
            & final_test_speakers,

        "val/test speaker overlap":
            final_val_speakers
            & final_test_speakers,

        "train/val utterance overlap":
            final_train_utts
            & final_val_utts,

        "train/test utterance overlap":
            final_train_utts
            & final_test_utts,

        "val/test utterance overlap":
            final_val_utts
            & final_test_utts,
    }

    for name, overlap in checks.items():

        if overlap:
            raise RuntimeError(
                f"Safety check failed: {name} = "
                f"{len(overlap):,}"
            )

    # ========================================================
    # Output
    # ========================================================

    output_rows = (
        train_candidates
        + val_candidates
        + test_rows
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            lineterminator="\n",
        )

        writer.writerows(
            output_rows
        )

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 72)
    print("SPEAKER-INDEPENDENT SPLIT CREATED")
    print("=" * 72)

    print()
    print(f"Input:   {args.input_csv}")
    print(f"Output:  {args.output_csv}")
    print(f"Seed:    {args.seed}")

    print()
    print("Original candidate pool")
    print("-----------------------")
    print(
        f"Mixtures:       {len(pool_rows):,}"
    )
    print(
        f"Speakers:       {len(pool_speakers):,}"
    )

    print()
    print("Speaker assignment")
    print("------------------")
    print(
        f"Train speakers: {len(train_speakers):,}"
    )
    print(
        f"Val speakers:   {len(val_speakers):,}"
    )
    print(
        f"Test speakers:  {len(test_speakers):,}"
    )

    print()
    print("Eligible mixtures before subsampling")
    print("------------------------------------")
    print(
        f"Train eligible: {len(train_candidates):,}"
        if args.train_mixtures is None
        else "Train final:    "
             f"{len(train_candidates):,}"
    )

    print(
        f"Val eligible:   {len(val_candidates):,}"
        if args.val_mixtures is None
        else "Val final:      "
             f"{len(val_candidates):,}"
    )

    print(
        f"Cross-group discarded: "
        f"{len(cross_group):,}"
    )

    print()
    print("Final CSV")
    print("---------")
    print(
        f"Train mixtures: {len(train_candidates):,}"
    )
    print(
        f"Val mixtures:   {len(val_candidates):,}"
    )
    print(
        f"Test mixtures:  {len(test_rows):,}"
    )
    print(
        f"Total:          {len(output_rows):,}"
    )

    print()
    print("Observed speakers in final mixtures")
    print("-----------------------------------")
    print(
        f"Train: {len(final_train_speakers):,}"
    )
    print(
        f"Val:   {len(final_val_speakers):,}"
    )
    print(
        f"Test:  {len(final_test_speakers):,}"
    )

    print()
    print("Independence checks")
    print("-------------------")

    for name, overlap in checks.items():
        print(
            f"{name:<32}: "
            f"{len(overlap):>6,}  "
            f"{'OK' if not overlap else 'FAIL'}"
        )

    print()


if __name__ == "__main__":
    main()