#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
from collections import Counter


# ============================================================
# Utilities
# ============================================================

def canonical_utt(speaker, utt):
    """
    Convert:
        speaker = id08108
        utt     = RCODGWdfzdY/00197

    into:
        id08108_RCODGWdfzdY_00197
    """
    utt = utt.replace("\\", "/").strip("/")
    utt = utt.replace("/", "_")

    return f"{speaker}_{utt}"


def extract_utt_from_npz(path):
    """
    Example:
        /scratch/.../mouths/id00015_AD2Mmf774IU_00077.npz

    ->
        id00015_AD2Mmf774IU_00077
    """
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    return name


def extract_speakers(utterances):
    """
    id00015_AD2Mmf774IU_00077 -> id00015
    """
    speakers = set()

    for utt in utterances:
        m = re.match(r"(id\d+)_", utt)

        if m:
            speakers.add(m.group(1))

    return speakers


def ordered_pair(a, b):
    return (a, b)


def unordered_pair(a, b):
    return tuple(sorted((a, b)))


# ============================================================
# Load SEANet CSV
# ============================================================

def load_seanet_csv(csv_path, split="train"):
    """
    Expected SEANet format:

    train,train,id08108,RCODGWdfzdY/00197,0,
          train,id07077,oi72_HHDWY4/00288,
          0.8976636599379368,4.096

    Interpretation assumed:

        [0] dataset split
        [1] source1 split
        [2] source1 speaker
        [3] source1 utterance
        [4] source1 something / offset
        [5] source2 split
        [6] source2 speaker
        [7] source2 utterance
        [8] SNR
        [9] duration
    """

    samples = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):

            if not row:
                continue

            if len(row) < 10:
                print(
                    f"[WARNING] line {line_number}: "
                    f"expected >=10 columns, got {len(row)}"
                )
                continue

            dataset_split = row[0].strip()

            if dataset_split != split:
                continue

            s1_speaker = row[2].strip()
            s1_name = row[3].strip()

            s2_speaker = row[6].strip()
            s2_name = row[7].strip()

            try:
                snr = float(row[8])
            except ValueError:
                snr = None

            try:
                duration = float(row[9])
            except ValueError:
                duration = None

            s1 = canonical_utt(
                s1_speaker,
                s1_name,
            )

            s2 = canonical_utt(
                s2_speaker,
                s2_name,
            )

            samples.append({
                "s1": s1,
                "s2": s2,
                "snr": snr,
                "duration": duration,
                "line": line_number,
            })

    return samples


# ============================================================
# Load Swift-Net
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_swiftnet(swift_dir):
    """
    Expected:

        tr/
          mix.json
          s1.json
          s2.json

    s1/s2 format:

        [
            audio_path,
            mouth_npz_path,
            length
        ]

    mix format:

        [
            audio_path,
            length
        ]
    """

    s1_path = os.path.join(swift_dir, "s1.json")
    s2_path = os.path.join(swift_dir, "s2.json")
    mix_path = os.path.join(swift_dir, "mix.json")

    s1_data = load_json(s1_path)
    s2_data = load_json(s2_path)
    mix_data = load_json(mix_path)

    if not (
        len(s1_data)
        == len(s2_data)
        == len(mix_data)
    ):
        raise RuntimeError(
            "Swift-Net JSON lengths differ:\n"
            f"  s1  = {len(s1_data)}\n"
            f"  s2  = {len(s2_data)}\n"
            f"  mix = {len(mix_data)}"
        )

    samples = []

    for i, (s1_entry, s2_entry, mix_entry) in enumerate(
        zip(s1_data, s2_data, mix_data)
    ):

        s1_audio = s1_entry[0]
        s1_npz = s1_entry[1]
        s1_length = s1_entry[2]

        s2_audio = s2_entry[0]
        s2_npz = s2_entry[1]
        s2_length = s2_entry[2]

        mix_audio = mix_entry[0]
        mix_length = mix_entry[1]

        s1 = extract_utt_from_npz(s1_npz)
        s2 = extract_utt_from_npz(s2_npz)

        samples.append({
            "s1": s1,
            "s2": s2,
            "s1_audio": s1_audio,
            "s2_audio": s2_audio,
            "mix_audio": mix_audio,
            "s1_length": s1_length,
            "s2_length": s2_length,
            "mix_length": mix_length,
            "index": i,
        })

    return samples


# ============================================================
# Comparison
# ============================================================

def percentage(n, total):
    if total == 0:
        return 0.0

    return 100.0 * n / total


def compare(seanet, swift):
    # --------------------------------------------------------
    # Individual utterances
    # --------------------------------------------------------

    seanet_s1 = {x["s1"] for x in seanet}
    seanet_s2 = {x["s2"] for x in seanet}
    seanet_all = seanet_s1 | seanet_s2

    swift_s1 = {x["s1"] for x in swift}
    swift_s2 = {x["s2"] for x in swift}
    swift_all = swift_s1 | swift_s2

    # --------------------------------------------------------
    # Speakers
    # --------------------------------------------------------

    seanet_speakers = extract_speakers(seanet_all)
    swift_speakers = extract_speakers(swift_all)

    # --------------------------------------------------------
    # Ordered pairs
    # --------------------------------------------------------

    seanet_ordered = Counter(
        ordered_pair(x["s1"], x["s2"])
        for x in seanet
    )

    swift_ordered = Counter(
        ordered_pair(x["s1"], x["s2"])
        for x in swift
    )

    # --------------------------------------------------------
    # Unordered pairs
    # --------------------------------------------------------

    seanet_unordered = Counter(
        unordered_pair(x["s1"], x["s2"])
        for x in seanet
    )

    swift_unordered = Counter(
        unordered_pair(x["s1"], x["s2"])
        for x in swift
    )

    ordered_common = (
        set(seanet_ordered)
        & set(swift_ordered)
    )

    unordered_common = (
        set(seanet_unordered)
        & set(swift_unordered)
    )

    # Account for duplicates too
    ordered_common_samples = sum(
        min(
            seanet_ordered[p],
            swift_ordered[p],
        )
        for p in ordered_common
    )

    unordered_common_samples = sum(
        min(
            seanet_unordered[p],
            swift_unordered[p],
        )
        for p in unordered_common
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("VOXCELEB2 TRAINING SET COMPARISON")
    print("=" * 78)

    print()
    print("DATASET SIZE")
    print("-" * 78)

    print(f"SEANet samples       : {len(seanet):,}")
    print(f"Swift-Net samples    : {len(swift):,}")

    print()
    print("UNIQUE UTTERANCES")
    print("-" * 78)

    print(f"SEANet s1            : {len(seanet_s1):,}")
    print(f"SEANet s2            : {len(seanet_s2):,}")
    print(f"SEANet total         : {len(seanet_all):,}")

    print()

    print(f"Swift-Net s1         : {len(swift_s1):,}")
    print(f"Swift-Net s2         : {len(swift_s2):,}")
    print(f"Swift-Net total      : {len(swift_all):,}")

    print()
    print("UTTERANCE OVERLAP")
    print("-" * 78)

    common_utterances = seanet_all & swift_all

    print(
        f"Common utterances    : "
        f"{len(common_utterances):,}"
    )

    print(
        f"SEANet covered       : "
        f"{percentage(len(common_utterances), len(seanet_all)):.2f}%"
    )

    print(
        f"Swift-Net covered    : "
        f"{percentage(len(common_utterances), len(swift_all)):.2f}%"
    )

    print()
    print("SPEAKERS")
    print("-" * 78)

    common_speakers = (
        seanet_speakers
        & swift_speakers
    )

    print(
        f"SEANet speakers      : "
        f"{len(seanet_speakers):,}"
    )

    print(
        f"Swift-Net speakers   : "
        f"{len(swift_speakers):,}"
    )

    print(
        f"Common speakers      : "
        f"{len(common_speakers):,}"
    )

    print(
        f"SEANet speaker cov.  : "
        f"{percentage(len(common_speakers), len(seanet_speakers)):.2f}%"
    )

    print(
        f"Swift speaker cov.   : "
        f"{percentage(len(common_speakers), len(swift_speakers)):.2f}%"
    )

    print()
    print("EXACT MIXTURE PAIRS")
    print("-" * 78)

    print(
        f"Unique SEANet pairs  : "
        f"{len(seanet_ordered):,}"
    )

    print(
        f"Unique Swift pairs   : "
        f"{len(swift_ordered):,}"
    )

    print(
        f"Exact ordered match  : "
        f"{len(ordered_common):,}"
    )

    print(
        f"Matched SEANet       : "
        f"{percentage(ordered_common_samples, len(seanet)):.4f}%"
    )

    print(
        f"Matched Swift-Net    : "
        f"{percentage(ordered_common_samples, len(swift)):.4f}%"
    )

    print()
    print("PAIR OVERLAP IGNORING SOURCE ORDER")
    print("-" * 78)

    print(
        f"Common pairs         : "
        f"{len(unordered_common):,}"
    )

    print(
        f"Matched SEANet       : "
        f"{percentage(unordered_common_samples, len(seanet)):.4f}%"
    )

    print(
        f"Matched Swift-Net    : "
        f"{percentage(unordered_common_samples, len(swift)):.4f}%"
    )

    # --------------------------------------------------------
    # Exact dataset equality
    # --------------------------------------------------------

    print()
    print("DATASET EQUALITY")
    print("-" * 78)

    same_ordered_set = (
        set(seanet_ordered)
        == set(swift_ordered)
    )

    same_ordered_multiset = (
        seanet_ordered
        == swift_ordered
    )

    same_unordered_set = (
        set(seanet_unordered)
        == set(swift_unordered)
    )

    same_utterance_set = (
        seanet_all
        == swift_all
    )

    print(
        f"Same utterance set   : "
        f"{same_utterance_set}"
    )

    print(
        f"Same ordered pairs   : "
        f"{same_ordered_set}"
    )

    print(
        f"Same pairs + counts  : "
        f"{same_ordered_multiset}"
    )

    print(
        f"Same unordered pairs : "
        f"{same_unordered_set}"
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    seanet_duplicates = sum(
        count - 1
        for count in seanet_ordered.values()
        if count > 1
    )

    swift_duplicates = sum(
        count - 1
        for count in swift_ordered.values()
        if count > 1
    )

    print()
    print("DUPLICATES")
    print("-" * 78)

    print(
        f"SEANet duplicate pairs : "
        f"{seanet_duplicates:,}"
    )

    print(
        f"Swift duplicate pairs  : "
        f"{swift_duplicates:,}"
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print()
    print("EXAMPLES OF MATCHING PAIRS")
    print("-" * 78)

    for i, pair in enumerate(
        sorted(ordered_common)[:10]
    ):
        print(
            f"{i + 1:2d}. "
            f"{pair[0]}  +  {pair[1]}"
        )

    seanet_only = (
        set(seanet_ordered)
        - set(swift_ordered)
    )

    swift_only = (
        set(swift_ordered)
        - set(seanet_ordered)
    )

    print()
    print("EXAMPLES ONLY IN SEANET")
    print("-" * 78)

    for i, pair in enumerate(
        sorted(seanet_only)[:10]
    ):
        print(
            f"{i + 1:2d}. "
            f"{pair[0]}  +  {pair[1]}"
        )

    print()
    print("EXAMPLES ONLY IN SWIFT-NET")
    print("-" * 78)

    for i, pair in enumerate(
        sorted(swift_only)[:10]
    ):
        print(
            f"{i + 1:2d}. "
            f"{pair[0]}  +  {pair[1]}"
        )

    # --------------------------------------------------------
    # Additional diagnostic:
    # for every SEANet mixture, check whether BOTH utterances
    # exist somewhere in Swift, even if they aren't paired.
    # --------------------------------------------------------

    seanet_both_exist_swift = sum(
        1
        for x in seanet
        if (
            x["s1"] in swift_all
            and
            x["s2"] in swift_all
        )
    )

    swift_both_exist_seanet = sum(
        1
        for x in swift
        if (
            x["s1"] in seanet_all
            and
            x["s2"] in seanet_all
        )
    )

    print()
    print("SAME SOURCE POOL, DIFFERENT MIXTURES?")
    print("-" * 78)

    print(
        "SEANet mixtures whose BOTH sources exist in Swift-Net:"
    )

    print(
        f"    {seanet_both_exist_swift:,} / "
        f"{len(seanet):,} "
        f"({percentage(seanet_both_exist_swift, len(seanet)):.2f}%)"
    )

    print()

    print(
        "Swift-Net mixtures whose BOTH sources exist in SEANet:"
    )

    print(
        f"    {swift_both_exist_seanet:,} / "
        f"{len(swift):,} "
        f"({percentage(swift_both_exist_seanet, len(swift)):.2f}%)"
    )

    print()
    print("=" * 78)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare SEANet and Swift-Net "
            "VoxCeleb2 training datasets"
        )
    )

    parser.add_argument(
        "--seanet-csv",
        required=True,
        help="SEANet data_list.csv",
    )

    parser.add_argument(
        "--swift-tr",
        required=True,
        help=(
            "Swift-Net Vox2/tr directory "
            "containing mix.json, s1.json, s2.json"
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
        help="SEANet split to compare (default: train)",
    )

    args = parser.parse_args()

    print("Loading SEANet...")
    seanet = load_seanet_csv(
        args.seanet_csv,
        split=args.split,
    )

    print(
        f"Loaded {len(seanet):,} "
        f"SEANet samples"
    )

    print("Loading Swift-Net...")
    swift = load_swiftnet(
        args.swift_tr,
    )

    print(
        f"Loaded {len(swift):,} "
        f"Swift-Net samples"
    )

    compare(
        seanet,
        swift,
    )


if __name__ == "__main__":
    main()