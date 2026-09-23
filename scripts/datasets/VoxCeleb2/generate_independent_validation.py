#!/usr/bin/env python3

import argparse
import csv
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
}



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a speaker-independent validation split "
            "for a SEANet/VoxCeleb2 data_list.csv."
        )
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Original SEANet data_list.csv.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output data_list.csv.",
    )

    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help=(
            "Root containing clean VoxCeleb2 audio. "
            "Expected layout: split/idXXXXX/video/utterance.wav"
        ),
    )


    parser.add_argument(
        "--num-val-speakers",
        type=int,
        default=300,
        help="Number of unseen validation speakers.",
    )

    parser.add_argument(
        "--num-val-mixtures",
        type=int,
        default=5000,
        help="Number of validation mixtures.",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Minimum utterance duration in seconds.",
    )

    parser.add_argument(
        "--snr-min",
        type=float,
        default=-10.0,
        help="Minimum mixture SNR.",
    )

    parser.add_argument(
        "--snr-max",
        type=float,
        default=10.0,
        help="Maximum mixture SNR.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--audio-split",
        type=str,
        default="train",
        help=(
            "Source split name written into columns 1 and 5 "
            "(default: train)."
        ),
    )

    parser.add_argument(
        "--keep-original-val",
        action="store_true",
        help=(
            "Keep original validation rows too. "
            "Normally they are replaced."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_csv(path):
    rows = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        for line_no, row in enumerate(reader, 1):

            if not row:
                continue

            if len(row) < 10:
                raise RuntimeError(
                    f"{path}:{line_no}: "
                    f"expected >=10 columns, got {len(row)}"
                )

            rows.append(row)

    return rows


def speakers_for_split(rows, split):
    speakers = set()

    for row in rows:

        if row[0].strip() != split:
            continue

        speakers.add(row[2].strip())
        speakers.add(row[6].strip())

    return speakers


def utterance_key_from_path(path, root):
    """
    Convert:

        ROOT/train/id00001/abc/00001.wav

    to:

        ("train", "id00001", "abc/00001")

    The same representation is used by data_list.csv.
    """

    rel = path.relative_to(root)

    if len(rel.parts) < 4:
        return None

    split = rel.parts[0]
    speaker = rel.parts[1]

    utterance_parts = list(rel.parts[2:])

    utterance_parts[-1] = (
        Path(utterance_parts[-1]).stem
    )

    utterance = "/".join(
        utterance_parts
    )

    return (
        split,
        speaker,
        utterance,
    )


def scan_files(root, extensions):
    result = {}

    print()
    print(f"Scanning {root} ...")

    count = 0

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        key = utterance_key_from_path(
            path,
            root,
        )

        if key is None:
            continue

        result[key] = path
        count += 1

    print(
        f"Found {count:,} files."
    )

    return result


def wav_duration(path):
    """
    Fast WAV duration reader.

    Uses soundfile if available.
    """

    try:
        import soundfile as sf

        info = sf.info(str(path))

        if info.samplerate <= 0:
            return None

        return (
            info.frames
            / float(info.samplerate)
        )

    except Exception:
        return None


def visual_num_frames(path):
    try:
        import numpy as np

        obj = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if isinstance(obj, np.lib.npyio.NpzFile):

            keys = obj.files

            if not keys:
                obj.close()
                return None

            shape = obj[keys[0]].shape
            obj.close()

        else:
            shape = obj.shape

        if not shape:
            return None

        return int(shape[0])

    except Exception:
        return None


def collect_candidates(
    audio_files,
    forbidden_speakers,
    source_split,
    min_duration,
):
    """
    Collect usable audio utterances for the independent validation set.

    Visual embeddings are intentionally not checked here. They are
    precomputed independently and are expected to follow the same
    speaker/utterance naming convention used by the SEANet loader.
    """

    by_speaker = defaultdict(list)

    duration_rejected = 0
    forbidden_rejected = 0
    wrong_split = 0

    print()
    print(
        f"Audio utterances available: "
        f"{len(audio_files):,}"
    )

    for key in sorted(audio_files):

        split, speaker, utterance = key

        # We only want utterances coming from the requested
        # VoxCeleb2 source split, normally "train".
        if split != source_split:
            wrong_split += 1
            continue

        # Validation speakers must be completely independent
        # from both original train and original test speakers.
        if speaker in forbidden_speakers:
            forbidden_rejected += 1
            continue

        duration = wav_duration(
            audio_files[key]
        )

        if (
            duration is None
            or duration < min_duration
        ):
            duration_rejected += 1
            continue

        by_speaker[speaker].append(
            {
                "split": split,
                "speaker": speaker,
                "utterance": utterance,
                "duration": duration,
                "audio": audio_files[key],
            }
        )

    print(
        f"Utterances outside '{source_split}' ignored: "
        f"{wrong_split:,}"
    )

    print(
        f"Forbidden-speaker utterances:     "
        f"{forbidden_rejected:,}"
    )

    print(
        f"Too-short/unreadable utterances:  "
        f"{duration_rejected:,}"
    )

    print(
        f"Usable independent utterances:    "
        f"{sum(len(v) for v in by_speaker.values()):,}"
    )

    return by_speaker

def select_validation_speakers(
    candidates,
    n,
    rng,
):
    speakers = [
        speaker
        for speaker, utterances
        in candidates.items()
        if utterances
    ]

    if len(speakers) < n:
        raise RuntimeError(
            f"Requested {n:,} validation speakers, "
            f"but only {len(speakers):,} independent "
            f"speakers with usable audio+visual data "
            f"were found."
        )

    rng.shuffle(speakers)

    return speakers[:n]


def generate_mixtures(
    candidates,
    selected_speakers,
    n_mixtures,
    snr_min,
    snr_max,
    rng,
):
    rows = []

    seen_pairs = set()

    attempts = 0
    max_attempts = max(
        100000,
        n_mixtures * 100,
    )

    while (
        len(rows) < n_mixtures
        and attempts < max_attempts
    ):
        attempts += 1

        target_speaker, interferer_speaker = (
            rng.sample(
                selected_speakers,
                2,
            )
        )

        target = rng.choice(
            candidates[target_speaker]
        )

        interferer = rng.choice(
            candidates[interferer_speaker]
        )

        pair_key = (
            target["split"],
            target["speaker"],
            target["utterance"],
            interferer["split"],
            interferer["speaker"],
            interferer["utterance"],
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(
            pair_key
        )

        snr = rng.uniform(
            snr_min,
            snr_max,
        )

        # Maximum fully-overlapped duration.
        mixture_duration = min(
            target["duration"],
            interferer["duration"],
        )

        rows.append(
            [
                "val",

                target["split"],
                target["speaker"],
                target["utterance"],
                "0",

                interferer["split"],
                interferer["speaker"],
                interferer["utterance"],

                f"{snr:.15g}",
                f"{mixture_duration:.6f}",
            ]
        )

    if len(rows) != n_mixtures:
        raise RuntimeError(
            f"Could generate only {len(rows):,}/"
            f"{n_mixtures:,} unique mixtures."
        )

    return rows


def verify(
    train_rows,
    val_rows,
    test_rows,
):
    def speakers(rows):
        result = set()

        for r in rows:
            result.add(r[2])
            result.add(r[6])

        return result

    def utterances(rows):
        result = set()

        for r in rows:

            result.add(
                (
                    r[1],
                    r[2],
                    r[3],
                )
            )

            result.add(
                (
                    r[5],
                    r[6],
                    r[7],
                )
            )

        return result

    train_s = speakers(train_rows)
    val_s = speakers(val_rows)
    test_s = speakers(test_rows)

    train_u = utterances(train_rows)
    val_u = utterances(val_rows)
    test_u = utterances(test_rows)

    checks = {
        "train/val speakers":
            train_s & val_s,

        "train/test speakers":
            train_s & test_s,

        "val/test speakers":
            val_s & test_s,

        "train/val utterances":
            train_u & val_u,

        "train/test utterances":
            train_u & test_u,

        "val/test utterances":
            val_u & test_u,
    }

    print()
    print("Independence checks")
    print("-------------------")

    for name, overlap in checks.items():

        status = (
            "OK"
            if not overlap
            else "FAIL"
        )

        print(
            f"{name:<25}: "
            f"{len(overlap):>6,}  "
            f"{status}"
        )

        if overlap:
            raise RuntimeError(
                f"Independence violation: {name}"
            )


def main():
    args = parse_args()

    rng = random.Random(
        args.seed
    )

    if not args.input_csv.is_file():
        raise FileNotFoundError(
            args.input_csv
        )

    if not args.audio_root.is_dir():
        raise NotADirectoryError(
            args.audio_root
        )



    if (
        args.output_csv.exists()
        and not args.force
    ):
        raise FileExistsError(
            f"{args.output_csv} exists. "
            f"Use --force to overwrite."
        )

    rows = load_csv(
        args.input_csv
    )

    train_rows = [
        r for r in rows
        if r[0].strip() == "train"
    ]

    original_val_rows = [
        r for r in rows
        if r[0].strip() == "val"
    ]

    test_rows = [
        r for r in rows
        if r[0].strip() == "test"
    ]

    train_speakers = speakers_for_split(
        rows,
        "train",
    )

    test_speakers = speakers_for_split(
        rows,
        "test",
    )

    forbidden_speakers = (
        train_speakers
        | test_speakers
    )

    print()
    print("=" * 72)
    print("INDEPENDENT VALIDATION GENERATOR")
    print("=" * 72)

    print()
    print("Original CSV")
    print("------------")
    print(
        f"Train mixtures: {len(train_rows):,}"
    )
    print(
        f"Val mixtures:   {len(original_val_rows):,}"
    )
    print(
        f"Test mixtures:  {len(test_rows):,}"
    )

    print()
    print(
        f"Train speakers: {len(train_speakers):,}"
    )
    print(
        f"Test speakers:  {len(test_speakers):,}"
    )
    print(
        f"Forbidden:      {len(forbidden_speakers):,}"
    )

    # ========================================================
    # Scan dataset
    # ========================================================

    audio_files = scan_files(
        args.audio_root,
        AUDIO_EXTENSIONS,
    )

    candidates = collect_candidates(
        audio_files=audio_files,
        forbidden_speakers=forbidden_speakers,
        source_split=args.audio_split,
        min_duration=args.min_duration,
    )

    print()
    print(
        f"Independent candidate speakers: "
        f"{len(candidates):,}"
    )

    # ========================================================
    # Select speakers
    # ========================================================

    selected_speakers = (
        select_validation_speakers(
            candidates,
            args.num_val_speakers,
            rng,
        )
    )

    selected_utterances = sum(
        len(candidates[s])
        for s in selected_speakers
    )

    print(
        f"Selected validation speakers:   "
        f"{len(selected_speakers):,}"
    )

    print(
        f"Available validation utterances:"
        f" {selected_utterances:,}"
    )

    # ========================================================
    # Generate mixtures
    # ========================================================

    val_rows = generate_mixtures(
        candidates=candidates,
        selected_speakers=selected_speakers,
        n_mixtures=args.num_val_mixtures,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        rng=rng,
    )

    # ========================================================
    # Safety checks
    # ========================================================

    verify(
        train_rows,
        val_rows,
        test_rows,
    )

    # ========================================================
    # Output
    # ========================================================

    if args.keep_original_val:

        output_rows = (
            train_rows
            + original_val_rows
            + val_rows
            + test_rows
        )

    else:

        output_rows = (
            train_rows
            + val_rows
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

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print()
    print(
        f"Output: {args.output_csv}"
    )

    print()
    print(
        f"Train mixtures: {len(train_rows):,}"
    )

    if args.keep_original_val:
        print(
            f"Original val:   "
            f"{len(original_val_rows):,}"
        )

    print(
        f"New val:        {len(val_rows):,}"
    )

    print(
        f"Test mixtures:  {len(test_rows):,}"
    )

    print()
    print(
        f"Validation speakers: "
        f"{len(selected_speakers):,}"
    )

    print(
        f"SNR range: [{args.snr_min:g}, "
        f"{args.snr_max:g}] dB"
    )

    print(
        f"Minimum duration: "
        f"{args.min_duration:g} s"
    )

    print()


if __name__ == "__main__":
    main()