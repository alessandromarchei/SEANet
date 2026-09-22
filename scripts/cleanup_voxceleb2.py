#!/usr/bin/env python3

"""
Safely remove unused VoxCeleb2 MP4/WAV files based on a SEANet data_list.csv.

Expected CSV format (no header):

    0  set_type_target
    1  split_target
    2  speaker_target
    3  utterance_target
    4  ...
    5  split_interferer
    6  speaker_interferer
    7  utterance_interferer
    8  snr
    9  length

Example:

train,train,id08108,RCODGWdfzdY/00197,0,train,id07077,oi72_HHDWY4/00288,0.89,4.096

This corresponds to keeping:

    id08108/RCODGWdfzdY/00197.mp4
    id07077/oi72_HHDWY4/00288.mp4

and equivalently for WAV.

IMPORTANT:
    Dry-run is the default.
    Files are deleted ONLY when --delete is explicitly passed.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path


# ============================================================
# Formatting
# ============================================================

def human_size(num_bytes):
    value = float(num_bytes)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(value) < 1024.0:
            return f"{value:,.2f} {unit}"
        value /= 1024.0

    return f"{value:,.2f} EiB"


def separator(char="=", width=80):
    print(char * width)


def section(title):
    print()
    separator("=")
    print(title)
    separator("=")


# ============================================================
# CSV parsing
# ============================================================
def normalize_relative_path(split, speaker, utterance):
    """
    split     = train
    speaker   = id08108
    utterance = RCODGWdfzdY/00197

    returns:
        train/id08108/RCODGWdfzdY/00197
    """

    split = split.strip().strip("/")
    speaker = speaker.strip().strip("/")
    utterance = utterance.strip().strip("/")

    if not split:
        raise ValueError("Empty split")

    if not speaker:
        raise ValueError("Empty speaker ID")

    if not utterance:
        raise ValueError("Empty utterance")

    path = Path(split) / speaker / utterance

    if ".." in path.parts:
        raise ValueError(f"Unsafe path: {path}")

    return path

def load_used_entries(csv_path):
    """
    Return the set of utterances referenced anywhere in the CSV.

    Both target and interferer are included.
    """

    used = set()

    row_count = 0
    malformed = []

    split_counter = Counter()
    speaker_counter = Counter()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):

            # Ignore completely empty rows
            if not row or all(not x.strip() for x in row):
                continue

            row_count += 1

            if len(row) < 8:
                malformed.append(
                    (line_number, f"Expected >= 8 columns, found {len(row)}", row)
                )
                continue

            try:
                target_split = row[1].strip()
                target_speaker = row[2].strip()
                target_utterance = row[3].strip()

                interferer_split = row[5].strip()
                interferer_speaker = row[6].strip()
                interferer_utterance = row[7].strip()

                target = normalize_relative_path(
                    target_split,
                    target_speaker,
                    target_utterance,
                )

                interferer = normalize_relative_path(
                    interferer_split,
                    interferer_speaker,
                    interferer_utterance,
                )

                used.add(target)
                used.add(interferer)

                split_counter[target_split] += 1
                split_counter[interferer_split] += 1

                speaker_counter[target_speaker] += 1
                speaker_counter[interferer_speaker] += 1

            except Exception as exc:
                malformed.append(
                    (line_number, str(exc), row)
                )

    return {
        "used": used,
        "rows": row_count,
        "malformed": malformed,
        "splits": split_counter,
        "speakers": speaker_counter,
    }


# ============================================================
# File scanning
# ============================================================

def scan_files(root, extension):
    """
    Recursively scan root for files with given extension.

    Returns:
        [
            {
                "path": absolute Path,
                "relative": path relative to root WITHOUT extension,
                "size": bytes
            }
        ]
    """

    root = root.resolve()
    extension = extension.lower()

    files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() != extension:
            continue

        try:
            size = path.stat().st_size
        except OSError as exc:
            print(f"[WARNING] Cannot stat {path}: {exc}")
            continue

        relative = path.relative_to(root).with_suffix("")

        files.append({
            "path": path,
            "relative": relative,
            "size": size,
        })

    return files


def classify_files(files, used_entries):

    keep = []
    delete = []

    for info in files:
        if info["relative"] in used_entries:
            keep.append(info)
        else:
            delete.append(info)

    return keep, delete


# ============================================================
# Statistics
# ============================================================

def total_size(files):
    return sum(x["size"] for x in files)


def print_stats(label, all_files, keep_files, delete_files):

    size_all = total_size(all_files)
    size_keep = total_size(keep_files)
    size_delete = total_size(delete_files)

    print()
    print(f"{label}")
    print("-" * 80)

    print(
        f"{'Category':<24}"
        f"{'Files':>15}"
        f"{'Space':>20}"
        f"{'% files':>12}"
        f"{'% space':>12}"
    )

    print("-" * 80)

    def row(name, files, size):

        file_pct = (
            len(files) / len(all_files) * 100
            if all_files else 0
        )

        space_pct = (
            size / size_all * 100
            if size_all else 0
        )

        print(
            f"{name:<24}"
            f"{len(files):>15,}"
            f"{human_size(size):>20}"
            f"{file_pct:>11.2f}%"
            f"{space_pct:>11.2f}%"
        )

    row("TOTAL", all_files, size_all)
    row("KEEP", keep_files, size_keep)
    row("DELETE", delete_files, size_delete)

    print("-" * 80)

    print(
        f"Space before cleanup : {human_size(size_all)}"
    )

    print(
        f"Space after cleanup  : {human_size(size_keep)}"
    )

    print(
        f"Space recovered      : {human_size(size_delete)}"
    )


# ============================================================
# Missing references
# ============================================================

def find_missing(used_entries, files):

    existing = {
        x["relative"]
        for x in files
    }

    return sorted(
        used_entries - existing,
        key=str,
    )


# ============================================================
# Deletion
# ============================================================

def delete_files(files):

    deleted_count = 0
    deleted_bytes = 0
    errors = []

    for i, info in enumerate(files, start=1):

        path = info["path"]

        try:
            path.unlink()

            deleted_count += 1
            deleted_bytes += info["size"]

            if i % 1000 == 0:
                print(
                    f"Deleted {i:,}/{len(files):,} files..."
                )

        except Exception as exc:
            errors.append(
                (path, str(exc))
            )

    return deleted_count, deleted_bytes, errors


def remove_empty_directories(root):

    removed = 0

    directories = sorted(
        [x for x in root.rglob("*") if x.is_dir()],
        key=lambda x: len(x.parts),
        reverse=True,
    )

    for directory in directories:

        try:
            directory.rmdir()
            removed += 1
        except OSError:
            # Directory not empty
            pass

    return removed


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Remove VoxCeleb2 MP4/WAV files not referenced "
            "by SEANet data_list.csv."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="SEANet data_list.csv",
    )

    parser.add_argument(
        "--mp4-root",
        type=Path,
        default=None,
        help="Root containing VoxCeleb2 MP4 files",
    )

    parser.add_argument(
        "--wav-root",
        type=Path,
        default=None,
        help="Root containing VoxCeleb2 WAV files",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete unused files. Default is DRY RUN.",
    )

    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Remove empty directories after deletion.",
    )

    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of files to show per category.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if not args.csv.is_file():
        print(f"ERROR: CSV does not exist: {args.csv}")
        sys.exit(1)

    if args.mp4_root is None and args.wav_root is None:
        print(
            "ERROR: specify at least one of "
            "--mp4-root or --wav-root"
        )
        sys.exit(1)

    for root in [args.mp4_root, args.wav_root]:
        if root is not None and not root.is_dir():
            print(f"ERROR: directory does not exist: {root}")
            sys.exit(1)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    section("SEANet VoxCeleb2 Dataset Cleanup")

    print(f"CSV       : {args.csv.resolve()}")
    print(f"MP4 root  : {args.mp4_root}")
    print(f"WAV root  : {args.wav_root}")

    print()
    print(
        "MODE      : "
        + ("DELETE" if args.delete else "DRY RUN")
    )

    section("Reading CSV")

    result = load_used_entries(args.csv)

    used = result["used"]
    malformed = result["malformed"]

    print(f"Mixtures              : {result['rows']:,}")
    print(f"Unique utterances     : {len(used):,}")
    print(f"Unique speakers       : {len(result['speakers']):,}")

    print()
    print("CSV references by split:")

    for split, count in result["splits"].most_common():
        print(f"  {split:<15} {count:>10,}")

    # --------------------------------------------------------
    # Fail safe
    # --------------------------------------------------------

    if malformed:

        section("ERROR: malformed CSV entries")

        print(
            f"Found {len(malformed):,} malformed rows."
        )

        for line, error, row in malformed[:20]:
            print()
            print(f"Line {line}: {error}")
            print(row)

        print()
        print(
            "Cleanup aborted because the CSV could not "
            "be parsed with 100% confidence."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    datasets = []

    if args.mp4_root:

        section("Scanning MP4 files")

        mp4_files = scan_files(
            args.mp4_root,
            ".mp4",
        )

        mp4_keep, mp4_delete = classify_files(
            mp4_files,
            used,
        )

        mp4_missing = find_missing(
            used,
            mp4_files,
        )

        datasets.append({
            "name": "MP4",
            "root": args.mp4_root,
            "all": mp4_files,
            "keep": mp4_keep,
            "delete": mp4_delete,
            "missing": mp4_missing,
        })

    if args.wav_root:

        section("Scanning WAV files")

        wav_files = scan_files(
            args.wav_root,
            ".wav",
        )

        wav_keep, wav_delete = classify_files(
            wav_files,
            used,
        )

        wav_missing = find_missing(
            used,
            wav_files,
        )

        datasets.append({
            "name": "WAV",
            "root": args.wav_root,
            "all": wav_files,
            "keep": wav_keep,
            "delete": wav_delete,
            "missing": wav_missing,
        })

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    section("Cleanup Statistics")

    for dataset in datasets:

        print_stats(
            dataset["name"],
            dataset["all"],
            dataset["keep"],
            dataset["delete"],
        )

    # --------------------------------------------------------
    # Global
    # --------------------------------------------------------

    total_before = sum(
        total_size(d["all"])
        for d in datasets
    )

    total_keep = sum(
        total_size(d["keep"])
        for d in datasets
    )

    total_delete = sum(
        total_size(d["delete"])
        for d in datasets
    )

    section("GLOBAL SUMMARY")

    print(
        f"{'':25}"
        f"{'Files':>15}"
        f"{'Space':>20}"
    )

    print("-" * 60)

    print(
        f"{'Before cleanup':25}"
        f"{sum(len(d['all']) for d in datasets):>15,}"
        f"{human_size(total_before):>20}"
    )

    print(
        f"{'Keep':25}"
        f"{sum(len(d['keep']) for d in datasets):>15,}"
        f"{human_size(total_keep):>20}"
    )

    print(
        f"{'Delete':25}"
        f"{sum(len(d['delete']) for d in datasets):>15,}"
        f"{human_size(total_delete):>20}"
    )

    print()
    print(
        f"Disk usage: "
        f"{human_size(total_before)} "
        f"-> {human_size(total_keep)}"
    )

    if total_before:
        percentage = total_delete / total_before * 100
        print(
            f"Recovered : {human_size(total_delete)} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Missing references
    # --------------------------------------------------------

    section("Missing Dataset References")

    any_missing = False

    for dataset in datasets:

        missing = dataset["missing"]

        print(
            f"{dataset['name']}: "
            f"{len(missing):,} / {len(used):,} "
            f"CSV utterances not found"
        )

        if missing:

            any_missing = True

            for path in missing[:args.show]:
                print(f"  MISSING: {path}")

            if len(missing) > args.show:
                print(
                    f"  ... and "
                    f"{len(missing) - args.show:,} more"
                )

    # --------------------------------------------------------
    # Files scheduled for deletion
    # --------------------------------------------------------

    section("Files Scheduled For Deletion")

    for dataset in datasets:

        print()
        print(
            f"{dataset['name']}: "
            f"{len(dataset['delete']):,} files"
        )

        for info in dataset["delete"][:args.show]:
            print(
                f"  {human_size(info['size']):>12}  "
                f"{info['path']}"
            )

        remaining = (
            len(dataset["delete"]) - args.show
        )

        if remaining > 0:
            print(
                f"  ... and {remaining:,} more"
            )

    # --------------------------------------------------------
    # Dry run ends here
    # --------------------------------------------------------

    if not args.delete:

        section("DRY RUN COMPLETE")

        print("NO FILES WERE DELETED.")
        print()
        print(
            f"Would delete : "
            f"{sum(len(d['delete']) for d in datasets):,} files"
        )

        print(
            f"Would recover: "
            f"{human_size(total_delete)}"
        )

        print()
        print(
            "Inspect the statistics and missing references carefully."
        )

        print()
        print(
            "To actually delete the unused files, "
            "run the same command with:"
        )

        print()
        print("    --delete")

        return

    # --------------------------------------------------------
    # Extra safety
    # --------------------------------------------------------

    if any_missing:

        section("DELETION ABORTED")

        print(
            "Some CSV-referenced utterances were not found."
        )

        print(
            "For safety, no files have been deleted."
        )

        print(
            "Fix the dataset paths/root directories first."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------

    section("DELETING UNUSED FILES")

    print(
        f"Deleting "
        f"{sum(len(d['delete']) for d in datasets):,} files..."
    )

    print(
        f"Expected recovery: {human_size(total_delete)}"
    )

    total_deleted_count = 0
    total_deleted_bytes = 0
    all_errors = []

    for dataset in datasets:

        print()
        print(f"Deleting {dataset['name']}...")

        count, size, errors = delete_files(
            dataset["delete"]
        )

        total_deleted_count += count
        total_deleted_bytes += size

        all_errors.extend(errors)

        if args.remove_empty_dirs:

            removed = remove_empty_directories(
                dataset["root"]
            )

            print(
                f"Removed {removed:,} empty directories."
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    section("CLEANUP COMPLETE")

    print(
        f"Files deleted : {total_deleted_count:,}"
    )

    print(
        f"Space freed   : {human_size(total_deleted_bytes)}"
    )

    print(
        f"Errors        : {len(all_errors):,}"
    )

    if all_errors:

        print()
        print("Deletion errors:")

        for path, error in all_errors[:50]:
            print(f"  {path}: {error}")


if __name__ == "__main__":
    main()