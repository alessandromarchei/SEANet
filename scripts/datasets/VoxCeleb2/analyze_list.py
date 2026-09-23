#!/usr/bin/env python3

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_COLUMNS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detailed analysis of a SEANet/AV-TSE data_list.csv."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Input data_list.csv",
    )

    parser.add_argument(
        "--show-overlap",
        type=int,
        default=10,
        help="Show up to N examples for each overlap (default: 10).",
    )

    return parser.parse_args()


def pct(n, total):
    if total == 0:
        return 0.0
    return 100.0 * n / total


def load_csv(path):
    rows = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, 1):

            if not row:
                continue

            if len(row) < EXPECTED_COLUMNS:
                raise ValueError(
                    f"{path}:{line_no}: expected at least "
                    f"{EXPECTED_COLUMNS} columns, got {len(row)}:\n"
                    f"{row}"
                )

            split = row[0].strip()

            target_source_split = row[1].strip()
            target_speaker = row[2].strip()
            target_utt = row[3].strip()

            interferer_source_split = row[5].strip()
            interferer_speaker = row[6].strip()
            interferer_utt = row[7].strip()

            target_key = (
                target_source_split,
                target_speaker,
                target_utt,
            )

            interferer_key = (
                interferer_source_split,
                interferer_speaker,
                interferer_utt,
            )

            rows.append(
                {
                    "line_no": line_no,
                    "raw": row,
                    "split": split,

                    "target_source_split": target_source_split,
                    "target_speaker": target_speaker,
                    "target_utt": target_utt,
                    "target_key": target_key,

                    "interferer_source_split": interferer_source_split,
                    "interferer_speaker": interferer_speaker,
                    "interferer_utt": interferer_utt,
                    "interferer_key": interferer_key,
                }
            )

    return rows


def build_stats(rows):
    stats = defaultdict(
        lambda: {
            "rows": [],
            "target_speakers": set(),
            "interferer_speakers": set(),
            "all_speakers": set(),
            "target_utts": set(),
            "interferer_utts": set(),
            "all_utts": set(),
            "target_counter": Counter(),
            "interferer_counter": Counter(),
            "all_utt_counter": Counter(),
            "source_splits_target": Counter(),
            "source_splits_interferer": Counter(),
            "snrs": [],
            "durations": [],
        }
    )

    for r in rows:
        s = stats[r["split"]]

        s["rows"].append(r)

        s["target_speakers"].add(
            r["target_speaker"]
        )

        s["interferer_speakers"].add(
            r["interferer_speaker"]
        )

        s["all_speakers"].update(
            [
                r["target_speaker"],
                r["interferer_speaker"],
            ]
        )

        s["target_utts"].add(
            r["target_key"]
        )

        s["interferer_utts"].add(
            r["interferer_key"]
        )

        s["all_utts"].update(
            [
                r["target_key"],
                r["interferer_key"],
            ]
        )

        s["target_counter"][r["target_key"]] += 1
        s["interferer_counter"][r["interferer_key"]] += 1

        s["all_utt_counter"][r["target_key"]] += 1
        s["all_utt_counter"][r["interferer_key"]] += 1

        s["source_splits_target"][
            r["target_source_split"]
        ] += 1

        s["source_splits_interferer"][
            r["interferer_source_split"]
        ] += 1

        try:
            s["snrs"].append(float(r["raw"][8]))
        except ValueError:
            pass

        try:
            s["durations"].append(float(r["raw"][-1]))
        except ValueError:
            pass

    return stats


def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_basic_stats(rows, stats):
    print_header("DATASET OVERVIEW")

    print(f"Total mixtures: {len(rows):,}")

    split_counts = Counter(
        r["split"] for r in rows
    )

    print()
    print(
        f"{'Split':<12}"
        f"{'Mixtures':>12}"
        f"{'%':>10}"
        f"{'Speakers':>12}"
        f"{'Target utts':>15}"
        f"{'All utts':>15}"
    )

    print("-" * 76)

    for split in sorted(split_counts):

        s = stats[split]

        print(
            f"{split:<12}"
            f"{len(s['rows']):>12,}"
            f"{pct(len(s['rows']), len(rows)):>9.2f}%"
            f"{len(s['all_speakers']):>12,}"
            f"{len(s['target_utts']):>15,}"
            f"{len(s['all_utts']):>15,}"
        )


def print_source_stats(stats):
    print_header("SOURCE SPLITS")

    for split in sorted(stats):

        s = stats[split]

        print(f"\n[{split}]")

        print("  Target:")
        for name, count in sorted(
            s["source_splits_target"].items()
        ):
            print(
                f"    {name:<12} {count:>8,}"
            )

        print("  Interferer:")
        for name, count in sorted(
            s["source_splits_interferer"].items()
        ):
            print(
                f"    {name:<12} {count:>8,}"
            )


def print_duplicate_stats(stats):
    print_header("UTTERANCE REUSE WITHIN EACH SPLIT")

    for split in sorted(stats):

        s = stats[split]

        repeated_target = {
            k: v
            for k, v in s["target_counter"].items()
            if v > 1
        }

        repeated_any = {
            k: v
            for k, v in s["all_utt_counter"].items()
            if v > 1
        }

        target_as_interferer = (
            s["target_utts"]
            & s["interferer_utts"]
        )

        print(f"\n[{split}]")
        print(
            f"  Unique target utterances:       "
            f"{len(s['target_utts']):,}"
        )
        print(
            f"  Unique interferer utterances:   "
            f"{len(s['interferer_utts']):,}"
        )
        print(
            f"  Unique utterances overall:      "
            f"{len(s['all_utts']):,}"
        )
        print(
            f"  Repeated target utterances:     "
            f"{len(repeated_target):,}"
        )
        print(
            f"  Utterances used >1 time:        "
            f"{len(repeated_any):,}"
        )
        print(
            f"  Used as target AND interferer:  "
            f"{len(target_as_interferer):,}"
        )


def show_examples(values, n):
    for value in sorted(values)[:n]:

        if isinstance(value, tuple):
            value = "/".join(value)

        print(f"      {value}")

    if len(values) > n:
        print(
            f"      ... +{len(values) - n:,} more"
        )


def print_pairwise_analysis(stats, show_n):
    print_header("PAIRWISE SPLIT INDEPENDENCE")

    splits = sorted(stats)

    for i, a in enumerate(splits):
        for b in splits[i + 1:]:

            sa = stats[a]
            sb = stats[b]

            speaker_overlap = (
                sa["all_speakers"]
                & sb["all_speakers"]
            )

            target_speaker_overlap = (
                sa["target_speakers"]
                & sb["target_speakers"]
            )

            utterance_overlap = (
                sa["all_utts"]
                & sb["all_utts"]
            )

            target_utt_overlap = (
                sa["target_utts"]
                & sb["target_utts"]
            )

            cross_target_interferer_ab = (
                sa["target_utts"]
                & sb["interferer_utts"]
            )

            cross_target_interferer_ba = (
                sb["target_utts"]
                & sa["interferer_utts"]
            )

            print()
            print(f"{a.upper()}  <->  {b.upper()}")
            print("-" * 78)

            print(
                f"All-speaker overlap:              "
                f"{len(speaker_overlap):,}"
            )

            print(
                f"Target-speaker overlap:           "
                f"{len(target_speaker_overlap):,}"
            )

            print(
                f"Any utterance overlap:            "
                f"{len(utterance_overlap):,}"
            )

            print(
                f"Target utterance overlap:         "
                f"{len(target_utt_overlap):,}"
            )

            print(
                f"{a} target -> {b} interferer: "
                f"{len(cross_target_interferer_ab):,}"
            )

            print(
                f"{b} target -> {a} interferer: "
                f"{len(cross_target_interferer_ba):,}"
            )

            speaker_independent = (
                len(speaker_overlap) == 0
            )

            utterance_independent = (
                len(utterance_overlap) == 0
            )

            print()
            print(
                "Speaker-independent:              "
                f"{'YES' if speaker_independent else 'NO'}"
            )

            print(
                "Utterance-independent:            "
                f"{'YES' if utterance_independent else 'NO'}"
            )

            if speaker_overlap and show_n > 0:
                print()
                print("  Example shared speakers:")
                show_examples(
                    speaker_overlap,
                    show_n,
                )

            if utterance_overlap and show_n > 0:
                print()
                print("  Example shared utterances:")
                show_examples(
                    utterance_overlap,
                    show_n,
                )


def print_speaker_roles(stats):
    print_header("TARGET / INTERFERER SPEAKER ROLES")

    for split in sorted(stats):

        s = stats[split]

        both = (
            s["target_speakers"]
            & s["interferer_speakers"]
        )

        target_only = (
            s["target_speakers"]
            - s["interferer_speakers"]
        )

        interferer_only = (
            s["interferer_speakers"]
            - s["target_speakers"]
        )

        print(f"\n[{split}]")
        print(
            f"  Target speakers:       "
            f"{len(s['target_speakers']):,}"
        )
        print(
            f"  Interferer speakers:   "
            f"{len(s['interferer_speakers']):,}"
        )
        print(
            f"  Speakers in both roles:"
            f" {len(both):,}"
        )
        print(
            f"  Target-only speakers:  "
            f"{len(target_only):,}"
        )
        print(
            f"  Interferer-only:       "
            f"{len(interferer_only):,}"
        )


def print_numeric_stats(stats):
    print_header("SNR / DURATION")

    for split in sorted(stats):

        s = stats[split]

        print(f"\n[{split}]")

        if s["snrs"]:
            values = s["snrs"]
            print(
                f"  SNR:      "
                f"min={min(values):.3f}, "
                f"mean={sum(values)/len(values):.3f}, "
                f"max={max(values):.3f}"
            )

        if s["durations"]:
            values = s["durations"]
            print(
                f"  Duration: "
                f"min={min(values):.3f}s, "
                f"mean={sum(values)/len(values):.3f}s, "
                f"max={max(values):.3f}s"
            )


def print_final_verdict(stats):
    print_header("INDEPENDENCE SUMMARY")

    splits = sorted(stats)

    print(
        f"{'Pair':<22}"
        f"{'Speaker independent':>22}"
        f"{'Utterance independent':>25}"
    )

    print("-" * 72)

    for i, a in enumerate(splits):
        for b in splits[i + 1:]:

            speaker_overlap = (
                stats[a]["all_speakers"]
                & stats[b]["all_speakers"]
            )

            utterance_overlap = (
                stats[a]["all_utts"]
                & stats[b]["all_utts"]
            )

            print(
                f"{a + ' <-> ' + b:<22}"
                f"{('YES' if not speaker_overlap else 'NO'):>22}"
                f"{('YES' if not utterance_overlap else 'NO'):>25}"
            )


def main():
    args = parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)

    rows = load_csv(args.csv)

    if not rows:
        raise RuntimeError(
            f"No entries found in {args.csv}"
        )

    stats = build_stats(rows)

    print()
    print(f"File: {args.csv.resolve()}")

    print_basic_stats(rows, stats)
    print_source_stats(stats)
    print_speaker_roles(stats)
    print_duplicate_stats(stats)
    print_pairwise_analysis(
        stats,
        args.show_overlap,
    )
    print_numeric_stats(stats)
    print_final_verdict(stats)

    print()


if __name__ == "__main__":
    main()