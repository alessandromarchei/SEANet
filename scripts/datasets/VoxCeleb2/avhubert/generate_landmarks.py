#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import face_alignment
from facenet_pytorch import MTCNN
from skimage import transform as tf


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate full-length Kai Li / Dolphin mouth ROIs."
    )

    p.add_argument(
        "--csv",
        required=True,
        help="SEANet configs/data_list.csv",
    )

    p.add_argument(
        "--video-root",
        required=True,
        help="Root containing train/... and test/... VoxCeleb2 videos",
    )

    p.add_argument(
        "--output-root",
        required=True,
        help="Output directory for full-length mouth NPZ files",
    )

    p.add_argument(
        "--mean-face",
        required=True,
        help="20words_mean_face.npy",
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    p.add_argument(
        "--detect-every",
        type=int,
        default=1,
    )

    p.add_argument(
        "--face-scale",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--max-files",
        type=int,
        default=0,
    )

    return p.parse_args()


args = parse_args()

video_root = Path(args.video_root)
output_root = Path(args.output_root)

output_root.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Kai constants -- DO NOT CHANGE
# ============================================================

STD_SIZE = (256, 256)

STABLE_POINTS = [33, 36, 39, 42, 45]

MOUTH_START = 48
MOUTH_STOP = 68

CROP_HEIGHT = 96
CROP_WIDTH = 96

WINDOW_MARGIN = 12

mean_face_landmarks = np.load(args.mean_face)


# ============================================================
# Original Kai transform.py
# ============================================================

def linear_interpolate(landmarks, start_idx, stop_idx):

    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]

    delta = stop_landmarks - start_landmarks

    for idx in range(1, stop_idx - start_idx):

        landmarks[start_idx + idx] = (
            start_landmarks
            + idx / float(stop_idx - start_idx) * delta
        )

    return landmarks


def warp_img(src, dst, img, std_size):

    tform = tf.estimate_transform(
        "similarity",
        src,
        dst,
    )

    warped = tf.warp(
        img,
        inverse_map=tform.inverse,
        output_shape=std_size,
    )

    warped = warped * 255
    warped = warped.astype("uint8")

    return warped, tform


def apply_transform(transform, img, std_size):

    warped = tf.warp(
        img,
        inverse_map=transform.inverse,
        output_shape=std_size,
    )

    warped = warped * 255
    warped = warped.astype("uint8")

    return warped


def cut_patch(
    img,
    landmarks,
    height,
    width,
    threshold=5,
):

    center_x, center_y = np.mean(
        landmarks,
        axis=0,
    )

    if center_y - height < 0:
        center_y = height

    if center_y - height < -threshold:
        raise Exception("too much bias in height")

    if center_x - width < 0:
        center_x = width

    if center_x - width < -threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height

    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")

    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width

    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    return np.copy(
        img[
            int(round(center_y) - round(height)):
            int(round(center_y) + round(height)),

            int(round(center_x) - round(width)):
            int(round(center_x) + round(width))
        ]
    )


# ============================================================
# Landmark interpolation -- Kai
# ============================================================

def landmarks_interpolate(landmarks):

    valid = [
        idx
        for idx, x in enumerate(landmarks)
        if x is not None
    ]

    if not valid:
        return None

    for idx in range(1, len(valid)):

        if valid[idx] - valid[idx - 1] == 1:
            continue

        landmarks = linear_interpolate(
            landmarks,
            valid[idx - 1],
            valid[idx],
        )

    valid = [
        idx
        for idx, x in enumerate(landmarks)
        if x is not None
    ]

    landmarks[:valid[0]] = (
        [landmarks[valid[0]]] * valid[0]
    )

    landmarks[valid[-1]:] = (
        [landmarks[valid[-1]]]
        * (len(landmarks) - valid[-1])
    )

    assert all(x is not None for x in landmarks)

    return landmarks


# ============================================================
# Kai mouth crop
# ============================================================

def crop_patch(frames, landmarks):

    q_frame = deque()
    q_landmarks = deque()

    sequence = []

    trans = None

    for frame_idx, frame in enumerate(frames):

        q_landmarks.append(
            landmarks[frame_idx]
        )

        q_frame.append(frame)

        if len(q_frame) == WINDOW_MARGIN:

            smoothed_landmarks = np.mean(
                q_landmarks,
                axis=0,
            )

            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()

            trans_frame, trans = warp_img(
                smoothed_landmarks[
                    STABLE_POINTS, :
                ],
                mean_face_landmarks[
                    STABLE_POINTS, :
                ],
                cur_frame,
                STD_SIZE,
            )

            trans_landmarks = trans(
                cur_landmarks
            )

            sequence.append(
                cut_patch(
                    trans_frame,
                    trans_landmarks[
                        MOUTH_START:MOUTH_STOP
                    ],
                    CROP_HEIGHT // 2,
                    CROP_WIDTH // 2,
                )
            )

    # --------------------------------------------------------
    # Exact Kai tail handling:
    # reuse last affine transform
    # --------------------------------------------------------

    if trans is None:

        smoothed_landmarks = np.mean(
            q_landmarks,
            axis=0,
        )

        cur_landmarks = q_landmarks.popleft()
        cur_frame = q_frame.popleft()

        trans_frame, trans = warp_img(
            smoothed_landmarks[
                STABLE_POINTS, :
            ],
            mean_face_landmarks[
                STABLE_POINTS, :
            ],
            cur_frame,
            STD_SIZE,
        )

        trans_landmarks = trans(
            cur_landmarks
        )

        sequence.append(
            cut_patch(
                trans_frame,
                trans_landmarks[
                    MOUTH_START:MOUTH_STOP
                ],
                CROP_HEIGHT // 2,
                CROP_WIDTH // 2,
            )
        )

    while q_frame:

        cur_frame = q_frame.popleft()

        trans_frame = apply_transform(
            trans,
            cur_frame,
            STD_SIZE,
        )

        trans_landmarks = trans(
            q_landmarks.popleft()
        )

        sequence.append(
            cut_patch(
                trans_frame,
                trans_landmarks[
                    MOUTH_START:MOUTH_STOP
                ],
                CROP_HEIGHT // 2,
                CROP_WIDTH // 2,
            )
        )

    return np.asarray(sequence)


# ============================================================
# Face box helper -- Kai
# ============================================================

def face2head(boxes, scale=1.0):

    new_boxes = []

    for box in boxes:

        width = box[2] - box[0]
        height = box[3] - box[1]

        cx = (box[2] + box[0]) / 2
        cy = (box[3] + box[1]) / 2

        square_width = int(
            max(width, height) * scale
        )

        new_boxes.append([
            cx - square_width / 2,
            cy - square_width / 2,
            cx + square_width / 2,
            cy + square_width / 2,
        ])

    return new_boxes


# ============================================================
# Read FULL video
# ============================================================

def read_video(path):

    cap = cv2.VideoCapture(str(path))

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # Kai converts BGR -> RGB before PIL
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            Image.fromarray(frame)
        )

    cap.release()

    return frames


# ============================================================
# Models
# ============================================================

device = torch.device(args.device)

print("Loading MTCNN...")

mtcnn = MTCNN(
    keep_all=True,
    device=device,
)

print("Loading face_alignment...")

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=str(device),
)


# ============================================================
# Process ONE utterance
# ============================================================

def process_video(video_path):

    original_frames = read_video(
        video_path
    )

    if not original_frames:
        raise RuntimeError(
            "video contains no frames"
        )

    face_frames = []
    landmarks = []

    last_box = None

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Kai original:
    #
    # frames[:int(video.fps * 2)]
    #
    # WE INTENTIONALLY PROCESS ALL FRAMES.
    #
    # Everything else stays equivalent.
    # --------------------------------------------------------

    for i, frame in enumerate(original_frames):

        if (
            i % args.detect_every == 0
            or last_box is None
        ):

            boxes, _ = mtcnn.detect(frame)

            if boxes is None or len(boxes) == 0:

                # Kai would fail here.
                # Keep previous box only when available.
                if last_box is None:
                    landmarks.append(None)
                    face_frames.append(None)
                    continue

                box = last_box

            else:

                box = face2head(
                    [boxes[0]],
                    args.face_scale,
                )[0]

                last_box = box

        else:

            box = last_box

        face = frame.crop(
            (
                box[0],
                box[1],
                box[2],
                box[3],
            )
        ).resize(
            (224, 224)
        )

        face_np = np.asarray(face)

        preds = fa.get_landmarks(
            face_np
        )

        if preds is None or len(preds) == 0:

            landmarks.append(None)

        else:

            landmarks.append(
                np.asarray(preds[0])
            )

        # Kai speaker video is written through OpenCV
        # as BGR. crop_patch reads BGR.
        face_bgr = cv2.cvtColor(
            face_np,
            cv2.COLOR_RGB2BGR,
        )

        face_frames.append(
            face_bgr
        )

    # Frames for which no initial face existed
    # cannot be processed correctly.
    valid_indices = [
        i
        for i, f in enumerate(face_frames)
        if f is not None
    ]

    if not valid_indices:
        raise RuntimeError(
            "no face detected"
        )

    first = valid_indices[0]

    if first > 0:

        for i in range(first):
            face_frames[i] = face_frames[first]

    landmarks = landmarks_interpolate(
        landmarks
    )

    if landmarks is None:
        raise RuntimeError(
            "no valid landmarks"
        )

    mouths = crop_patch(
        face_frames,
        landmarks,
    )

    # Exact Kai:
    # convert BGR mouth patches to grayscale
    gray = np.stack(
        [
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )
            for frame in mouths
        ],
        axis=0,
    )

    return gray


# ============================================================
# Read SEANet CSV
# ============================================================

items = {}

with open(args.csv, newline="") as f:

    reader = csv.reader(f)

    for row in reader:

        if not row:
            continue

        # SEANet:
        #
        # row[1] = source split
        # row[2] = target speaker
        # row[3] = target video/utterance

        source_split = row[1].strip()
        speaker = row[2].strip()
        utterance = row[3].strip()

        key = (
            source_split,
            speaker,
            utterance,
        )

        items[key] = None


items = list(items.keys())

if args.max_files > 0:
    items = items[:args.max_files]


print()
print("=" * 72)
print("KAI FULL-LENGTH MOUTH GENERATOR")
print("=" * 72)
print(f"CSV          : {args.csv}")
print(f"Video root   : {video_root}")
print(f"Output root  : {output_root}")
print(f"Unique target utterances: {len(items)}")
print()


# ============================================================
# Main
# ============================================================

success = 0
failed = []

for source_split, speaker, utterance in tqdm(
    items,
    dynamic_ncols=True,
):

    video_path = (
        video_root
        / source_split
        / speaker
        / f"{utterance}.mp4"
    )

    output_path = (
        output_root
        / speaker
        / f"{utterance}.npz"
    )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        continue

    try:

        if not video_path.exists():
            raise FileNotFoundError(
                video_path
            )

        mouths = process_video(
            video_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            output_path,
            data=mouths,
        )

        success += 1

    except Exception as exc:

        failed.append(
            (
                str(video_path),
                repr(exc),
            )
        )


print()
print("=" * 72)
print("DONE")
print("=" * 72)
print(f"Generated : {success}")
print(f"Failed    : {len(failed)}")

if failed:

    failure_path = (
        output_root / "failed.txt"
    )

    with open(failure_path, "w") as f:

        for path, error in failed:
            f.write(
                f"{path}\t{error}\n"
            )

    print(f"Failure log: {failure_path}")