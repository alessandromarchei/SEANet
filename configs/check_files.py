#!/usr/bin/env python3

import os
from pathlib import Path

CSV = "configs/data_list.csv"

RAW_ROOT = Path("/scratch_ssd/VoxCeleb2-2Mix/raw_audio")
CLEAN_ROOT = Path("/scratch_nvme/VoxCeleb2-2Mix/audio_clean/train")


print("Indexing audio_clean/train...")

# Index by filename
clean_by_filename = {}

for p in CLEAN_ROOT.rglob("*.wav"):
    clean_by_filename.setdefault(p.name, []).append(p)

print(f"Indexed {sum(len(v) for v in clean_by_filename.values())} WAVs")
print()


def flat_filename(speaker, utterance):
    # id01750 + AXSzm1Bn-Ws/00121
    # ->
    # id01750_AXSzm1Bn-Ws_00121.wav
    return f"{speaker}_{utterance.replace('/', '_')}.wav"


raw_found = set()
clean_found = set()
missing_everywhere = set()

examples_clean = []

with open(CSV) as f:
    for line_num, line in enumerate(f, 1):

        d = line.strip().split(",")

        entries = [
            (d[2], d[3]),
            (d[6], d[7]),
        ]

        for speaker, utterance in entries:

            filename = flat_filename(speaker, utterance)

            raw_path = RAW_ROOT / filename

            key = (speaker, utterance)

            # ----------------------------------------
            # Existing flattened raw_audio
            # ----------------------------------------

            if raw_path.is_file():
                raw_found.add(key)
                continue

            # ----------------------------------------
            # Search audio_clean/train
            # ----------------------------------------

            matches = clean_by_filename.get(filename, [])

            if matches:
                clean_found.add(key)

                if len(examples_clean) < 20:
                    examples_clean.append(
                        (filename, matches[0])
                    )

                continue

            # ----------------------------------------
            # Maybe clean tree uses just 00121.wav
            # ----------------------------------------

            basename = utterance.split("/")[-1] + ".wav"

            possible = []

            for p in clean_by_filename.get(basename, []):
                path_str = str(p)

                if speaker in path_str and utterance.split("/")[0] in path_str:
                    possible.append(p)

            if possible:
                clean_found.add(key)

                if len(examples_clean) < 20:
                    examples_clean.append(
                        (filename, possible[0])
                    )

                continue

            missing_everywhere.add(
                (speaker, utterance)
            )


print("========================================")
print("CSV audio coverage")
print("========================================")

print(f"Found in raw_audio:        {len(raw_found)}")
print(f"Recovered in audio_clean:  {len(clean_found)}")
print(f"Missing everywhere:        {len(missing_everywhere)}")

print()

if examples_clean:
    print("Examples recovered from audio_clean:")
    for expected, actual in examples_clean:
        print()
        print("Expected:")
        print(f"  {expected}")
        print("Found:")
        print(f"  {actual}")

if missing_everywhere:
    print()
    print("First 30 still missing:")

    for speaker, utterance in sorted(missing_everywhere)[:30]:
        print(f"  {speaker}/{utterance}")