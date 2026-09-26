#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict
from argparse import Namespace
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Generate cached AV-HuBERT visual embeddings directly "
        "from VoxCeleb2 face-track MP4 files."
    )
)

parser.add_argument(
    "--video_root",
    type=str,
    required=True,
    help=(
        "Root containing VoxCeleb2 face-track MP4s, e.g. "
        ".../orig/train"
    ),
)

parser.add_argument(
    "--output_root",
    type=str,
    required=True,
    help="Output root for [T,D] AV-HuBERT .npy embeddings.",
)

parser.add_argument(
    "--data_list",
    type=str,
    default=None,
    help="Optional SEANet data_list.csv.",
)

parser.add_argument(
    "--split",
    type=str,
    choices=["train", "val", "test"],
    default=None,
)

parser.add_argument(
    "--avhubert_root",
    type=str,
    required=True,
    help="Root of cloned facebookresearch/av_hubert repository.",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="AV-HuBERT checkpoint, e.g. large_vox_433h.pt",
)

parser.add_argument(
    "--landmark_predictor",
    type=str,
    required=True,
    help="dlib shape_predictor_68_face_landmarks.dat",
)

parser.add_argument(
    "--mean_face",
    type=str,
    required=True,
    help="20words_mean_face.npy",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
)

parser.add_argument(
    "--workers",
    type=int,
    default=8,
    help="CPU workers for decoding + landmarks + alignment.",
)

parser.add_argument(
    "--prefetch_factor",
    type=int,
    default=2,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Maximum AV-HuBERT batch size for equal-length videos.",
)

parser.add_argument(
    "--max_videos",
    type=int,
    default=0,
)

parser.add_argument(
    "--overwrite",
    action="store_true",
)

parser.add_argument(
    "--save_mouth_rois",
    action="store_true",
    help="Optionally save [T,96,96] mouth ROI uint8 for debugging.",
)

parser.add_argument(
    "--mouth_root",
    type=str,
    default=None,
)

parser.add_argument(
    "--window_margin",
    type=int,
    default=12,
    help="Temporal landmark smoothing window. AV-HuBERT style.",
)

parser.add_argument(
    "--crop_size",
    type=int,
    default=96,
    help="Aligned mouth ROI before AV-HuBERT center crop.",
)

args = parser.parse_args()


# ============================================================
# Basic checks
# ============================================================

video_root = Path(args.video_root).resolve()
output_root = Path(args.output_root).resolve()
avhubert_root = Path(args.avhubert_root).resolve()
checkpoint = Path(args.checkpoint).resolve()
mean_face_path = Path(args.mean_face).resolve()
predictor_path = Path(args.landmark_predictor).resolve()

if not video_root.is_dir():
    raise FileNotFoundError(video_root)

if not avhubert_root.is_dir():
    raise FileNotFoundError(avhubert_root)

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

if not mean_face_path.is_file():
    raise FileNotFoundError(mean_face_path)

if not predictor_path.is_file():
    raise FileNotFoundError(predictor_path)

if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable.")

output_root.mkdir(
    parents=True,
    exist_ok=True,
)

if args.save_mouth_rois:

    if args.mouth_root is None:
        raise ValueError(
            "--mouth_root required with --save_mouth_rois"
        )

    mouth_root = Path(args.mouth_root).resolve()

    mouth_root.mkdir(
        parents=True,
        exist_ok=True,
    )

device = torch.device(args.device)


# ============================================================
# AV-HuBERT paths
# ============================================================

AVHUBERT_USER_DIR = avhubert_root / "avhubert"
FAIRSEQ_ROOT = avhubert_root / "fairseq"
AVHUBERT_UTILS_PATH = AVHUBERT_USER_DIR / "utils.py"


def add_path(path):

    path = str(path)

    if path not in sys.path:
        sys.path.insert(0, path)


add_path(avhubert_root)
add_path(FAIRSEQ_ROOT)


# ============================================================
# Load AV-HuBERT utils
# ============================================================

def load_module_from_path(
    module_name,
    module_path,
):

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(module_path),
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Unable to import {module_path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


avhubert_utils = load_module_from_path(
    "avhubert_local_utils",
    AVHUBERT_UTILS_PATH,
)


# ============================================================
# Fairseq
# ============================================================

from fairseq import checkpoint_utils
from fairseq import utils as fairseq_utils


# ============================================================
# PyTorch >= 2.6 legacy checkpoint compatibility
# ============================================================

def patch_torch_load():

    original = torch.load

    if getattr(
        original,
        "_avhubert_legacy_patch",
        False,
    ):
        return

    def wrapped(*a, **kw):

        kw.setdefault(
            "weights_only",
            False,
        )

        return original(
            *a,
            **kw,
        )

    wrapped._avhubert_legacy_patch = True

    torch.load = wrapped


patch_torch_load()


# ============================================================
# Load AV-HuBERT ONCE
# ============================================================

print()
print("Loading AV-HuBERT")
print("==================")
print(f"Repository: {avhubert_root}")
print(f"Checkpoint: {checkpoint}")


fairseq_utils.import_user_module(
    Namespace(
        user_dir=str(AVHUBERT_USER_DIR)
    )
)


models, saved_cfg, task = (
    checkpoint_utils
    .load_model_ensemble_and_task(
        [str(checkpoint)]
    )
)


if len(models) != 1:

    raise RuntimeError(
        f"Expected one model, got {len(models)}"
    )


model = models[0]


# Fine-tuned seq2seq checkpoints wrap AV-HuBERT.
if hasattr(model, "decoder"):

    print(
        "Detected fine-tuned checkpoint; "
        "using encoder.w2v_model."
    )

    model = model.encoder.w2v_model


model = model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad_(False)


print()
print("AV-HuBERT task configuration")
print("----------------------------")
print(
    f"image_crop_size: "
    f"{task.cfg.image_crop_size}"
)
print(
    f"image_mean:      "
    f"{task.cfg.image_mean}"
)
print(
    f"image_std:       "
    f"{task.cfg.image_std}"
)
print()


# ============================================================
# Mean face
# ============================================================

mean_face = np.load(
    mean_face_path
).astype(np.float32)


if mean_face.shape != (68, 2):

    raise RuntimeError(
        f"Expected mean face [68,2], "
        f"got {mean_face.shape}"
    )


# ============================================================
# AV-HuBERT alignment constants
# ============================================================

# Stable points commonly used by the AV-HuBERT / lipreading
# preprocessing pipeline:
#
# nose + eye corners

STABLE_POINTS = [
    33,
    36,
    39,
    42,
    45,
]

MOUTH_START = 48
MOUTH_END = 68

STD_SIZE = (256, 256)


# ============================================================
# SEANet CSV
# ============================================================

def load_videos_from_data_list(
    csv_path,
    video_root,
    split=None,
):

    videos = set()

    with open(
        csv_path,
        "r",
        newline="",
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            if len(row) < 4:
                continue

            mixture_split = row[0].strip()
            target_split = row[1].strip()
            speaker_id = row[2].strip()
            utterance = row[3].strip()

            if (
                split is not None
                and mixture_split != split
            ):
                continue

            if (
                split is not None
                and target_split != split
            ):
                continue

            path = (
                video_root
                / speaker_id
                / f"{utterance}.mp4"
            )

            videos.add(path)

    return sorted(videos)


# ============================================================
# Landmark utilities
# ============================================================

# Predictor is instantiated lazily PER worker.
#
# This avoids trying to pickle a dlib predictor.
_WORKER_PREDICTOR = None


def get_landmark_predictor():

    global _WORKER_PREDICTOR

    if _WORKER_PREDICTOR is None:

        _WORKER_PREDICTOR = (
            dlib.shape_predictor(
                str(predictor_path)
            )
        )

    return _WORKER_PREDICTOR


def shape_to_numpy(shape):

    coords = np.empty(
        (68, 2),
        dtype=np.float32,
    )

    for i in range(68):

        coords[i, 0] = shape.part(i).x
        coords[i, 1] = shape.part(i).y

    return coords


def detect_landmarks_face_track(
    gray,
):

    """
    The MP4 is already a VoxCeleb face track.

    Therefore we deliberately skip the expensive CNN/HOG
    face detector and give the entire image to the 68-point
    landmark predictor.
    """

    h, w = gray.shape[:2]

    rect = dlib.rectangle(
        0,
        0,
        w - 1,
        h - 1,
    )

    predictor = get_landmark_predictor()

    shape = predictor(
        gray,
        rect,
    )

    return shape_to_numpy(shape)


# ============================================================
# Missing landmark interpolation
# ============================================================

def interpolate_landmarks(
    landmarks,
):

    """
    landmarks:
        list of either [68,2] arrays or None

    Linear interpolation for isolated failures.
    """

    n = len(landmarks)

    valid = [
        i
        for i, lm in enumerate(landmarks)
        if lm is not None
    ]

    if len(valid) == 0:

        raise RuntimeError(
            "No valid landmarks in video."
        )

    # Fill before first valid frame.
    first = valid[0]

    for i in range(first):

        landmarks[i] = (
            landmarks[first].copy()
        )

    # Fill after last valid frame.
    last = valid[-1]

    for i in range(last + 1, n):

        landmarks[i] = (
            landmarks[last].copy()
        )

    # Interior interpolation.
    for a, b in zip(
        valid[:-1],
        valid[1:],
    ):

        if b == a + 1:
            continue

        lm_a = landmarks[a]
        lm_b = landmarks[b]

        span = b - a

        for i in range(a + 1, b):

            alpha = (
                (i - a)
                / span
            )

            landmarks[i] = (
                (1.0 - alpha) * lm_a
                + alpha * lm_b
            )

    return np.stack(
        landmarks,
        axis=0,
    ).astype(np.float32)


# ============================================================
# Temporal smoothing
# ============================================================

def smooth_landmarks(
    landmarks,
    window_margin,
):

    """
    Moving-average landmark smoothing.

    [T,68,2] -> [T,68,2]
    """

    T = landmarks.shape[0]

    result = np.empty_like(
        landmarks
    )

    for t in range(T):

        start = max(
            0,
            t - window_margin,
        )

        end = min(
            T,
            t + window_margin + 1,
        )

        result[t] = (
            landmarks[start:end]
            .mean(axis=0)
        )

    return result


# ============================================================
# Similarity transform
# ============================================================

def estimate_similarity_transform(
    src,
    dst,
):

    """
    Estimate similarity transform:
        x' = s R x + t

    using OpenCV estimateAffinePartial2D.
    """

    matrix, _ = cv2.estimateAffinePartial2D(
        src.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.LMEDS,
    )

    if matrix is None:

        raise RuntimeError(
            "Could not estimate face alignment."
        )

    return matrix.astype(np.float32)


def transform_points(
    points,
    matrix,
):

    ones = np.ones(
        (
            points.shape[0],
            1,
        ),
        dtype=np.float32,
    )

    homogeneous = np.concatenate(
        [
            points.astype(np.float32),
            ones,
        ],
        axis=1,
    )

    return (
        homogeneous
        @ matrix.T
    )


# ============================================================
# Crop mouth
# ============================================================

def crop_mouth_patch(
    image,
    landmarks,
    crop_size=96,
):

    mouth = landmarks[
        MOUTH_START:MOUTH_END
    ]

    center = mouth.mean(
        axis=0
    )

    half = crop_size // 2

    cx = float(center[0])
    cy = float(center[1])

    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half

    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # Padding if crop touches border.
    pad_left = max(
        0,
        -x1,
    )

    pad_top = max(
        0,
        -y1,
    )

    pad_right = max(
        0,
        x2 - image.shape[1],
    )

    pad_bottom = max(
        0,
        y2 - image.shape[0],
    )

    if (
        pad_left
        or pad_top
        or pad_right
        or pad_bottom
    ):

        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

        x1 += pad_left
        x2 += pad_left

        y1 += pad_top
        y2 += pad_top

    patch = image[
        y1:y2,
        x1:x2,
    ]

    if patch.shape != (
        crop_size,
        crop_size,
    ):

        raise RuntimeError(
            f"Unexpected mouth crop "
            f"{patch.shape}"
        )

    return patch


# ============================================================
# Decode + AV-HuBERT-like alignment
# ============================================================

def preprocess_video(
    video_path,
):

    """
    MP4 face track
        ->
    grayscale frames
        ->
    68 landmarks
        ->
    interpolation
        ->
    temporal smoothing
        ->
    similarity alignment to mean face
        ->
    96x96 mouth ROI

    Returns:
        uint8 [T,96,96]
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open {video_path}"
        )

    frames = []
    landmarks = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        frames.append(gray)

        try:

            lm = (
                detect_landmarks_face_track(
                    gray
                )
            )

        except Exception:

            lm = None

        landmarks.append(lm)

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            "No decoded frames."
        )

    # --------------------------------------------------------
    # Landmark interpolation
    # --------------------------------------------------------

    landmarks = interpolate_landmarks(
        landmarks
    )

    # --------------------------------------------------------
    # Smooth landmarks
    # --------------------------------------------------------

    smoothed = smooth_landmarks(
        landmarks,
        args.window_margin,
    )

    # --------------------------------------------------------
    # Align each frame
    # --------------------------------------------------------

    mouth_rois = []

    destination = mean_face[
        STABLE_POINTS
    ]

    for t, frame in enumerate(frames):

        # Transformation is estimated from smoothed stable
        # landmarks.
        source = smoothed[
            t,
            STABLE_POINTS,
        ]

        matrix = (
            estimate_similarity_transform(
                source,
                destination,
            )
        )

        # Align original grayscale frame.
        aligned = cv2.warpAffine(
            frame,
            matrix,
            (
                STD_SIZE[1],
                STD_SIZE[0],
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Transform ORIGINAL landmarks using the smoothed
        # alignment transform.
        aligned_landmarks = (
            transform_points(
                landmarks[t],
                matrix,
            )
        )

        mouth = crop_mouth_patch(
            aligned,
            aligned_landmarks,
            crop_size=args.crop_size,
        )

        mouth_rois.append(mouth)

    mouth_rois = np.stack(
        mouth_rois,
        axis=0,
    )

    return mouth_rois


# ============================================================
# Dataset
# ============================================================

class VideoDataset(Dataset):

    def __init__(
        self,
        videos,
    ):

        self.videos = list(videos)

    def __len__(self):

        return len(
            self.videos
        )

    def __getitem__(
        self,
        index,
    ):

        path = self.videos[index]

        try:

            rois = preprocess_video(
                path
            )

            return {
                "ok": True,
                "video_path": str(path),
                "rois": rois,
                "frames": rois.shape[0],
                "error": "",
            }

        except Exception as exc:

            return {
                "ok": False,
                "video_path": str(path),
                "rois": None,
                "frames": 0,
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }


def collate_single(items):

    return items[0]


# ============================================================
# Prepare AV-HuBERT input
# ============================================================

def prepare_avhubert_batch(
    samples,
):

    """
    samples all have same T.

    Input:
        uint8 [B,T,96,96]

    Output:
        float32 [B,1,T,88,88]
    """

    rois = np.stack(
        [
            x["rois"]
            for x in samples
        ],
        axis=0,
    )

    x = torch.from_numpy(
        rois
    ).float()

    # --------------------------------------------------------
    # AV-HuBERT inference preprocessing
    #
    # Normalize(0,255)
    # CenterCrop(88)
    # Normalize(mean,std)
    # --------------------------------------------------------

    x = x / 255.0

    crop = int(
        task.cfg.image_crop_size
    )

    H = x.shape[-2]
    W = x.shape[-1]

    if H < crop or W < crop:

        raise RuntimeError(
            f"ROI {H}x{W} smaller than "
            f"AV-HuBERT crop {crop}"
        )

    top = (
        H - crop
    ) // 2

    left = (
        W - crop
    ) // 2

    x = x[
        :,
        :,
        top:top + crop,
        left:left + crop,
    ]

    x = (
        x
        - float(task.cfg.image_mean)
    ) / float(task.cfg.image_std)

    # [B,T,H,W]
    # ->
    # [B,1,T,H,W]

    x = (
        x
        .unsqueeze(1)
        .contiguous()
    )

    return x.to(
        device,
        non_blocking=True,
    )


# ============================================================
# AV-HuBERT inference
# ============================================================

@torch.inference_mode()
def process_batch(
    samples,
):

    if len(samples) == 0:
        return 0

    T = int(
        samples[0]["frames"]
    )

    if any(
        int(x["frames"]) != T
        for x in samples
    ):

        raise RuntimeError(
            "Mixed temporal lengths."
        )

    x = prepare_avhubert_batch(
        samples
    )

    # --------------------------------------------------------
    # AV-HuBERT visual-only
    # --------------------------------------------------------

    features, _ = (
        model.extract_finetune(
            source={
                "video": x,
                "audio": None,
            },
            padding_mask=None,
            output_layer=None,
        )
    )

    # Expected:
    #
    # [B,T,D]

    if not torch.is_tensor(
        features
    ):

        raise RuntimeError(
            "AV-HuBERT returned "
            f"{type(features)}"
        )

    features = (
        features
        .detach()
        .float()
        .cpu()
    )

    if features.ndim != 3:

        raise RuntimeError(
            "Expected AV-HuBERT output "
            f"[B,T,D], got "
            f"{tuple(features.shape)}"
        )

    B, out_T, D = (
        features.shape
    )

    if B != len(samples):

        raise RuntimeError(
            f"Batch mismatch: "
            f"{B} != {len(samples)}"
        )

    if out_T != T:

        raise RuntimeError(
            f"Temporal mismatch: "
            f"input={T}, output={out_T}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    for i, sample in enumerate(
        samples
    ):

        video_path = Path(
            sample["video_path"]
        )

        relative = (
            video_path
            .relative_to(video_root)
            .with_suffix(".npy")
        )

        output_path = (
            output_root
            / relative
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        embedding = (
            features[i]
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        np.save(
            output_path,
            embedding,
        )

        # Optional debug mouth ROI.
        if args.save_mouth_rois:

            roi_path = (
                mouth_root
                / relative
            )

            roi_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                roi_path,
                sample["rois"],
            )

    return B, D


# ============================================================
# Find input videos
# ============================================================

if args.data_list is not None:

    videos = (
        load_videos_from_data_list(
            args.data_list,
            video_root,
            args.split,
        )
    )

else:

    videos = sorted(
        video_root.rglob(
            "*.mp4"
        )
    )


videos = [
    x
    for x in videos
    if x.is_file()
]


if args.max_videos > 0:

    videos = videos[
        :args.max_videos
    ]


# ============================================================
# Skip cached
# ============================================================

todo = []
skipped = 0


for path in videos:

    relative = (
        path
        .relative_to(video_root)
        .with_suffix(".npy")
    )

    dst = (
        output_root
        / relative
    )

    if (
        dst.exists()
        and not args.overwrite
    ):

        skipped += 1
        continue

    todo.append(path)


print()
print("AV-HuBERT embedding generation")
print("==============================")
print(f"Input root:       {video_root}")
print(f"Output root:      {output_root}")
print(f"Checkpoint:       {checkpoint}")
print(f"Mean face:        {mean_face_path}")
print(f"Landmarks:        {predictor_path}")
print(f"Videos total:     {len(videos)}")
print(f"Already done:     {skipped}")
print(f"To process:       {len(todo)}")
print(f"CPU workers:      {args.workers}")
print(f"GPU batch size:   {args.batch_size}")
print(f"Device:           {device}")
print(
    f"AV-HuBERT crop:   "
    f"{task.cfg.image_crop_size}"
)
print()


# ============================================================
# DataLoader
# ============================================================

dataset = VideoDataset(
    todo
)


loader_kwargs = dict(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.workers,
    collate_fn=collate_single,
    pin_memory=False,
)


if args.workers > 0:

    loader_kwargs.update(
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )


loader = DataLoader(
    **loader_kwargs
)


# ============================================================
# Temporal buckets
# ============================================================

buckets = defaultdict(list)

success = 0
failed = 0
num_batches = 0
embedding_dim = None


# ============================================================
# Main
# ============================================================

for sample in tqdm(
    loader,
    total=len(dataset),
    desc="AV-HuBERT",
    dynamic_ncols=True,
):

    if not sample["ok"]:

        failed += 1

        tqdm.write(
            "\nFAILED:\n"
            f"  {sample['video_path']}\n"
            f"  {sample['error']}\n"
        )

        continue

    T = int(
        sample["frames"]
    )

    buckets[T].append(
        sample
    )

    if (
        len(buckets[T])
        < args.batch_size
    ):

        continue

    batch = buckets[T]

    buckets[T] = []

    try:

        processed, D = (
            process_batch(
                batch
            )
        )

        success += processed
        num_batches += 1
        embedding_dim = D

    except Exception as exc:

        failed += len(batch)

        tqdm.write(
            "\nFAILED GPU BATCH:\n"
            f"  T={T}\n"
            f"  B={len(batch)}\n"
            f"  {type(exc).__name__}: "
            f"{exc}\n"
        )


# ============================================================
# Flush incomplete buckets
# ============================================================

remaining = sum(
    len(x)
    for x in buckets.values()
)


if remaining:

    print()
    print(
        f"Flushing {remaining} videos "
        "from incomplete temporal buckets..."
    )


for T in sorted(
    buckets.keys()
):

    batch = buckets[T]

    if not batch:
        continue

    try:

        processed, D = (
            process_batch(
                batch
            )
        )

        success += processed
        num_batches += 1
        embedding_dim = D

    except Exception as exc:

        failed += len(batch)

        print(
            "\nFAILED FLUSH BATCH:\n"
            f"  T={T}\n"
            f"  B={len(batch)}\n"
            f"  {type(exc).__name__}: "
            f"{exc}"
        )


# ============================================================
# Summary
# ============================================================

print()
print("Done")
print("====")
print(f"Success:          {success}")
print(f"Failed:           {failed}")
print(f"Already cached:   {skipped}")
print(f"GPU batches:      {num_batches}")

if embedding_dim is not None:

    print(
        f"Embedding dim:    "
        f"{embedding_dim}"
    )

print()