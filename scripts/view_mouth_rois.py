#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Find files
# ============================================================

def find_npy_files(root: Path):
    files = sorted(root.rglob("*.npy"))

    if not files:
        raise RuntimeError(
            f"No .npy files found under:\n{root}"
        )

    return files


# ============================================================
# Load video ROI cache
# ============================================================

def load_rois(path: Path):
    arr = np.load(path)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected [T,H,W], got {arr.shape} in {path}"
        )

    if arr.dtype != np.uint8:
        # Only for visualization.
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


# ============================================================
# Draw
# ============================================================

def make_display(
    frame,
    file_path,
    root,
    video_idx,
    num_videos,
    frame_idx,
    num_frames,
    scale,
):
    h, w = frame.shape

    display = cv2.resize(
        frame,
        (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    display = cv2.cvtColor(
        display,
        cv2.COLOR_GRAY2BGR,
    )

    # Add information panel on top.
    header_h = 110

    canvas = np.zeros(
        (
            display.shape[0] + header_h,
            display.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[header_h:] = display

    try:
        relative_path = file_path.relative_to(root)
    except ValueError:
        relative_path = file_path

    lines = [
        f"Video: {video_idx + 1}/{num_videos}",
        f"Frame: {frame_idx + 1}/{num_frames}",
        str(relative_path),
        "A/D frame | W/S video | SPACE play | Q quit",
    ]

    y = 22

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return canvas


# ============================================================
# Main viewer
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for AV-HuBERT mouth ROI caches."
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing .npy [T,H,W] mouth ROI files.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=5,
        help="Display scale factor. Default: 5",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Playback FPS. Default: 25",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    files = find_npy_files(root)

    print("=" * 70)
    print("AV-HuBERT Mouth ROI Viewer")
    print("=" * 70)
    print(f"Root:         {root}")
    print(f"Videos:       {len(files)}")
    print(f"Display scale:{args.scale}x")
    print(f"Playback FPS: {args.fps}")
    print()
    print("Controls")
    print("--------")
    print("A / Left       Previous frame")
    print("D / Right      Next frame")
    print("W / Up         Previous video")
    print("S / Down       Next video")
    print("Space          Play / pause")
    print("R              Restart current video")
    print("Home           First video")
    print("End            Last video")
    print("Q / Esc        Quit")
    print("=" * 70)

    video_idx = 0
    frame_idx = 0
    playing = False

    rois = load_rois(files[video_idx])

    window_name = "AV-HuBERT Mouth ROI"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    while True:

        # ----------------------------------------------------
        # Clamp indices
        # ----------------------------------------------------

        video_idx = max(
            0,
            min(video_idx, len(files) - 1),
        )

        frame_idx = max(
            0,
            min(frame_idx, len(rois) - 1),
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        canvas = make_display(
            frame=rois[frame_idx],
            file_path=files[video_idx],
            root=root,
            video_idx=video_idx,
            num_videos=len(files),
            frame_idx=frame_idx,
            num_frames=len(rois),
            scale=args.scale,
        )

        cv2.imshow(
            window_name,
            canvas,
        )

        # ----------------------------------------------------
        # Playback
        # ----------------------------------------------------

        if playing:
            delay = max(
                1,
                int(1000.0 / args.fps),
            )
        else:
            delay = 0

        key = cv2.waitKeyEx(delay)

        # ----------------------------------------------------
        # Automatic playback
        # ----------------------------------------------------

        if playing and key == -1:
            frame_idx += 1

            if frame_idx >= len(rois):
                frame_idx = 0

                if video_idx < len(files) - 1:
                    video_idx += 1
                    rois = load_rois(files[video_idx])
                else:
                    playing = False
                    frame_idx = len(rois) - 1

            continue

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if key in (
            ord("q"),
            ord("Q"),
            27,  # ESC
        ):
            break

        # ----------------------------------------------------
        # Play / pause
        # ----------------------------------------------------

        elif key == ord(" "):
            playing = not playing

        # ----------------------------------------------------
        # Next frame
        # ----------------------------------------------------

        elif key in (
            ord("d"),
            ord("D"),
            2555904,  # right arrow
        ):
            playing = False

            frame_idx = min(
                frame_idx + 1,
                len(rois) - 1,
            )

        # ----------------------------------------------------
        # Previous frame
        # ----------------------------------------------------

        elif key in (
            ord("a"),
            ord("A"),
            2424832,  # left arrow
        ):
            playing = False

            frame_idx = max(
                frame_idx - 1,
                0,
            )

        # ----------------------------------------------------
        # Next video
        # ----------------------------------------------------

        elif key in (
            ord("s"),
            ord("S"),
            2621440,  # down arrow
        ):
            playing = False

            if video_idx < len(files) - 1:
                video_idx += 1
                rois = load_rois(files[video_idx])
                frame_idx = 0

        # ----------------------------------------------------
        # Previous video
        # ----------------------------------------------------

        elif key in (
            ord("w"),
            ord("W"),
            2490368,  # up arrow
        ):
            playing = False

            if video_idx > 0:
                video_idx -= 1
                rois = load_rois(files[video_idx])
                frame_idx = 0

        # ----------------------------------------------------
        # Restart video
        # ----------------------------------------------------

        elif key in (
            ord("r"),
            ord("R"),
        ):
            playing = False
            frame_idx = 0

        # ----------------------------------------------------
        # First video
        # ----------------------------------------------------

        elif key == 2359296:  # Home
            playing = False
            video_idx = 0
            rois = load_rois(files[video_idx])
            frame_idx = 0

        # ----------------------------------------------------
        # Last video
        # ----------------------------------------------------

        elif key == 2293760:  # End
            playing = False
            video_idx = len(files) - 1
            rois = load_rois(files[video_idx])
            frame_idx = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()