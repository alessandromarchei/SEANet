#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
    )

    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    video_root = Path(args.video_root).resolve()
    output = Path(args.output).resolve()

    videos = set()

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 4:
                continue

            mixture_split = row[0].strip()
            target_split = row[1].strip()
            speaker_id = row[2].strip()
            utterance = row[3].strip()

            if args.split is not None:
                if mixture_split != args.split:
                    continue
                if target_split != args.split:
                    continue

            path = video_root / speaker_id / f"{utterance}.mp4"

            if not path.exists():
                print(f"[MISSING] {path}")
                continue

            rel = path.relative_to(video_root).with_suffix("")

            videos.add(rel.as_posix())

    videos = sorted(videos)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as f:
        for x in videos:
            f.write(x + "\n")

    print()
    print("SEANet -> AV-HuBERT manifest")
    print("============================")
    print(f"CSV:       {csv_path}")
    print(f"Videos:    {len(videos)}")
    print(f"Output:    {output}")


if __name__ == "__main__":
    main()