#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
import csv

# ============================================================
# Add SEANet repository root to Python path
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Generate SEANet/VoxMix visual embeddings from VoxCeleb2 "
        "224x224 face-track MP4 files at an arbitrary temporal FPS."
    )
)

parser.add_argument(
    "--video_root",
    type=str,
    required=True,
    help=(
        "Root containing idXXXXX/video_id/*.mp4, e.g. "
        "/scratch_nvme/VoxCeleb2-2Mix/orig/train"
    ),
)

parser.add_argument(
    "--output_root",
    type=str,
    required=True,
    help=(
        "Output root. Directory structure relative to video_root "
        "is preserved."
    ),
)

parser.add_argument(
    "--visual_frontend",
    type=str,
    required=True,
    help="Path to visual_frontend.pt",
)

parser.add_argument(
    "--fps",
    type=float,
    default=25.0,
    help="Target FPS BEFORE the visual frontend. Default: 25",
)

parser.add_argument(
    "--data_list",
    type=str,
    default=None,
    help=(
        "Optional SEANet data_list.csv. "
        "If provided, process only target-speaker videos "
        "referenced by the CSV instead of recursively scanning video_root."
    ),
)

parser.add_argument(
    "--split",
    type=str,
    default=None,
    choices=["train", "val", "test"],
    help=(
        "Optional mixture split to select from data_list.csv. "
        "For example --split train."
    ),
)


parser.add_argument(
    "--source_fps",
    type=float,
    default=25.0,
    help=(
        "Expected source FPS of VoxCeleb2 face-track MP4 files. "
        "Default: 25"
    ),
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
)

parser.add_argument(
    "--overwrite",
    action="store_true",
)

parser.add_argument(
    "--max_videos",
    type=int,
    default=0,
    help="Process at most N videos. 0 = all.",
)

parser.add_argument(
    "--save_rois",
    action="store_true",
    help=(
        "Also save the selected 112x112 grayscale input frames "
        "for debugging."
    ),
)

parser.add_argument(
    "--roi_root",
    type=str,
    default="",
    help="Root for debugging ROI .npy files.",
)

parser.add_argument(
    "--normalization",
    type=str,
    default="mean_std",
    choices=[
        "zero_one",
        "minus_one_one",
        "raw",
        "mean_std",
    ],
    help=(
        "Pixel normalization before VisualFrontend: "
        "zero_one=x/255, "
        "minus_one_one=(x/255-0.5)/0.5, "
        "raw=x, "
        "mean_std=(x/255-mean)/std"
    ),
)

parser.add_argument(
    "--pixel_mean",
    type=float,
    default=0.4161,
    help="Mean used when --normalization mean_std",
)

parser.add_argument(
    "--pixel_std",
    type=float,
    default=0.1688,
    help="Std used when --normalization mean_std",
)

parser.add_argument(
    "--spatial_preprocess",
    type=str,
    default="center_crop",
    choices=[
        "center_crop",
        "resize",
    ],
    help=(
        "center_crop: resize to 224 then central 112x112 crop; "
        "resize: resize complete frame directly to 112x112"
    ),
)

parser.add_argument(
    "--grayscale",
    type=str,
    default="opencv",
    choices=[
        "opencv",
        "average",
        "red",
        "green",
        "blue",
    ],
    help="Method used to convert BGR video frame to one channel.",
)

parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Optional single MP4 to process.",
)


args = parser.parse_args()



def load_videos_from_data_list(
    csv_path,
    video_root,
    split=None,
):
    """
    Load only TARGET-speaker utterances required by SEANet.

    Expected data_list.csv format:

        col 0: mixture split
        col 1: target source split
        col 2: target speaker ID
        col 3: target utterance path (video_id/utterance_id)

    Example:

        train,train,id08108,RCODGWdfzdY/00197,...

    becomes:

        <video_root>/id08108/RCODGWdfzdY/00197.mp4

    NOTE:
        video_root is assumed to already point to the source split,
        e.g.:
            /scratch_nvme/VoxCeleb2-2Mix/orig/train
    """

    csv_path = Path(csv_path)
    video_root = Path(video_root)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Data list not found: {csv_path}"
        )

    videos = set()

    total_rows = 0
    selected_rows = 0

    with open(csv_path, "r", newline="") as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            total_rows += 1

            if len(row) < 4:
                raise RuntimeError(
                    f"Malformed CSV row {total_rows}: {row}"
                )

            mixture_split = row[0].strip()
            target_split = row[1].strip()
            speaker_id = row[2].strip()
            utterance = row[3].strip()

            # Select requested mixture split
            if split is not None and mixture_split != split:
                continue

            # Since video_root already points to /train,
            # make sure CSV target is also train.
            if split is not None and target_split != split:
                continue

            selected_rows += 1

            video_path = (
                video_root
                / speaker_id
                / f"{utterance}.mp4"
            )

            videos.add(video_path)

    videos = sorted(videos)

    print()
    print("Data-list filtering")
    print("-------------------")
    print(f"CSV:             {csv_path}")
    print(f"CSV rows:        {total_rows}")
    print(f"Selected rows:   {selected_rows}")
    print(f"Unique targets:  {len(videos)}")
    print()

    return videos

# ============================================================
# Checks
# ============================================================

if args.fps <= 0:
    raise ValueError("--fps must be > 0")

if args.fps > args.source_fps:
    raise ValueError(
        f"Target FPS ({args.fps}) cannot be greater than "
        f"source FPS ({args.source_fps})."
    )

if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but no CUDA GPU is available.")

video_root = Path(args.video_root).resolve()
output_root = Path(args.output_root).resolve()

if not video_root.is_dir():
    raise FileNotFoundError(video_root)

if not os.path.isfile(args.visual_frontend):
    raise FileNotFoundError(args.visual_frontend)

output_root.mkdir(parents=True, exist_ok=True)

if args.save_rois:
    if not args.roi_root:
        raise ValueError(
            "--roi_root must be specified when --save_rois is used."
        )

    roi_root = Path(args.roi_root).resolve()
    roi_root.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load visual frontend
# ============================================================

def load_visual_frontend(path, device):
    """
    Load the exact SEANet visual frontend architecture and its state_dict.
    """

    from pretrain_networks.visual_frontend import VisualFrontend

    print(f"Loading visual frontend architecture...")
    model = VisualFrontend()

    print(f"Loading visual frontend weights: {path}")
    state_dict = torch.load(
        path,
        map_location="cpu",
    )

    # visual_frontend.pt is directly an OrderedDict/state_dict
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print("Visual frontend loaded successfully.")

    return model

device = torch.device(args.device)

visual_frontend = load_visual_frontend(
    args.visual_frontend,
    device,
)


# ============================================================
# Temporal sampling
# ============================================================

def temporal_indices(
    num_frames,
    source_fps,
    target_fps,
):
    """
    Generate indices corresponding to a regularly sampled
    target-FPS timeline.

    Example:

        source = 25 FPS
        target = 20 FPS

        source times:
            0, 40, 80, 120, 160, 200, ... ms

        target times:
            0, 50, 100, 150, 200, ... ms

    Each requested target timestamp is mapped to the closest
    available source frame.
    """

    if num_frames <= 0:
        return np.empty(
            (0,),
            dtype=np.int64,
        )

    if abs(target_fps - source_fps) < 1e-8:
        return np.arange(
            num_frames,
            dtype=np.int64,
        )

    # Duration represented by the frame sequence.
    duration = num_frames / source_fps

    num_output_frames = int(
        np.floor(
            duration * target_fps
            + 1e-8
        )
    )

    num_output_frames = max(
        num_output_frames,
        1,
    )

    target_times = (
        np.arange(
            num_output_frames,
            dtype=np.float64,
        )
        / target_fps
    )

    indices = np.rint(
        target_times * source_fps
    ).astype(np.int64)

    indices = np.clip(
        indices,
        0,
        num_frames - 1,
    )

    # Rounding should not create duplicates when target <= source,
    # but remove them defensively.
    indices = np.unique(indices)

    return indices


# ============================================================
# Video preprocessing
# ============================================================

def convert_to_grayscale(frame, method):
    """
    Input:
        OpenCV BGR frame [H,W,3]

    Output:
        grayscale [H,W]
    """

    if method == "opencv":
        # OpenCV:
        # Y ≈ 0.114 B + 0.587 G + 0.299 R
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

    elif method == "average":
        gray = frame.astype(np.float32).mean(axis=2)

    elif method == "red":
        gray = frame[:, :, 2]

    elif method == "green":
        gray = frame[:, :, 1]

    elif method == "blue":
        gray = frame[:, :, 0]

    else:
        raise ValueError(
            f"Unknown grayscale method: {method}"
        )

    return gray


def read_video_112(video_path):
    """
    Decode MP4 and produce one-channel 112x112 frames.

    Returns:
        [T,112,112]
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # BGR -> one channel
        # ----------------------------------------------------

        frame = convert_to_grayscale(
            frame,
            args.grayscale,
        )

        # ----------------------------------------------------
        # Spatial preprocessing
        # ----------------------------------------------------

        if args.spatial_preprocess == "center_crop":

            # Force original expected spatial size
            frame = cv2.resize(
                frame,
                (224, 224),
                interpolation=cv2.INTER_LINEAR,
            )

            # Central 112x112
            frame = frame[
                56:168,
                56:168,
            ]

        elif args.spatial_preprocess == "resize":

            # Resize the complete face-track frame directly
            frame = cv2.resize(
                frame,
                (112, 112),
                interpolation=cv2.INTER_LINEAR,
            )

        else:
            raise ValueError(
                f"Unknown spatial preprocessing: "
                f"{args.spatial_preprocess}"
            )

        if frame.shape != (112, 112):
            raise RuntimeError(
                f"Unexpected frame shape {frame.shape} "
                f"for {video_path}"
            )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames decoded from {video_path}"
        )

    return np.stack(
        frames,
        axis=0,
    )


# ============================================================
# Prepare input for visual frontend
# ============================================================
def prepare_visual_tensor(frames, device):
    """
    Input:
        frames: [T,112,112]

    Output expected by VisualFrontend:
        [T,B,C,H,W]

    with:
        B = 1
        C = 1

    VisualFrontend internally transforms this into:
        [B,C,T,H,W]
    """

    x = torch.from_numpy(
        frames
    ).float()

    # ========================================================
    # Pixel normalization
    # ========================================================

    if args.normalization == "zero_one":

        # [0,255] -> [0,1]
        x = x / 255.0

    elif args.normalization == "minus_one_one":

        # [0,255] -> [-1,1]
        x = x / 255.0
        x = (x - 0.5) / 0.5

    elif args.normalization == "raw":

        # Leave pixels approximately [0,255]
        pass

    elif args.normalization == "mean_std":

        x = x / 255.0

        x = (
            x - args.pixel_mean
        ) / args.pixel_std

    else:

        raise ValueError(
            f"Unknown normalization: "
            f"{args.normalization}"
        )

    # ========================================================
    # [T,H,W]
    #     ->
    # [T,B,C,H,W]
    #
    # B=1
    # C=1
    # ========================================================

    x = x.unsqueeze(1).unsqueeze(2)

    return x.to(
        device,
        non_blocking=True,
    )


# ============================================================
# Normalize frontend output shape
# ============================================================
def normalize_embedding_shape(output, expected_frames):
    """
    VisualFrontend returns:
        [T, B, 512]

    We process one video at a time, therefore B=1.

    Return:
        [T, 512]
    """

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise RuntimeError(
            f"Visual frontend returned {type(output)}, expected Tensor."
        )

    output = output.detach().float().cpu()

    if output.ndim != 3:
        raise RuntimeError(
            f"Expected VisualFrontend output [T,B,512], "
            f"got {tuple(output.shape)}"
        )

    T, B, D = output.shape

    if B != 1:
        raise RuntimeError(
            f"Expected batch size 1, got shape {tuple(output.shape)}"
        )

    if D != 512:
        raise RuntimeError(
            f"Expected embedding dimension 512, "
            f"got shape {tuple(output.shape)}"
        )

    # [T,1,512] -> [T,512]
    output = output[:, 0, :]

    if output.shape[0] != expected_frames:
        raise RuntimeError(
            f"Temporal length mismatch: "
            f"input frames={expected_frames}, "
            f"output embeddings={output.shape[0]}"
        )

    return output.numpy()


# ============================================================
# Process one video
# ============================================================

@torch.inference_mode()
def process_video(
    video_path,
    output_path,
):
    # --------------------------------------------------------
    # 1. Decode + spatial preprocessing at original FPS
    # --------------------------------------------------------

    frames = read_video_112(
        video_path
    )

    original_frames = frames.shape[0]

    # --------------------------------------------------------
    # 2. TRUE temporal downsampling BEFORE Conv3D
    # --------------------------------------------------------

    indices = temporal_indices(
        num_frames=original_frames,
        source_fps=args.source_fps,
        target_fps=args.fps,
    )

    frames = frames[indices]

    sampled_frames = frames.shape[0]

    # --------------------------------------------------------
    # Optional debugging cache
    # --------------------------------------------------------

    if args.save_rois:

        relative = (
            video_path
            .relative_to(video_root)
            .with_suffix(".npy")
        )

        roi_path = (
            roi_root
            / relative
        )

        roi_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            roi_path,
            frames,
        )

    # --------------------------------------------------------
    # 3. Visual frontend
    #
    # IMPORTANT:
    # Conv3D sees the temporally downsampled frames.
    # --------------------------------------------------------

    x = prepare_visual_tensor(
        frames,
        device,
    )

    output = visual_frontend(x)

    embedding = normalize_embedding_shape(
        output,
        expected_frames=sampled_frames,
    )

    # --------------------------------------------------------
    # 4. Save exactly as SEANet cache:
    #
    # idXXXXX/video_id/utterance.npy
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        embedding.astype(
            np.float32,
            copy=False,
        ),
    )

    return {
        "original_frames": original_frames,
        "sampled_frames": sampled_frames,
        "embedding_frames": embedding.shape[0],
        "embedding_dim": embedding.shape[1],
    }


# ============================================================
# Find videos
# ============================================================

if args.video is not None:

    videos = [
        Path(args.video).resolve()
    ]

elif args.data_list is not None:

    videos = load_videos_from_data_list(
        csv_path=args.data_list,
        video_root=video_root,
        split=args.split,
    )

else:

    print(
        "WARNING: no --data_list provided. "
        "Scanning all MP4 files recursively."
    )

    videos = sorted(
        video_root.rglob("*.mp4")
    )

if args.max_videos > 0:
    videos = videos[
        :args.max_videos
    ]

if len(videos) == 0:
    raise RuntimeError(
        f"No MP4 files found under {video_root}"
    )



existing_videos = []
missing_videos = []

for path in videos:
    if path.is_file():
        existing_videos.append(path)
    else:
        missing_videos.append(path)

print(f"Existing videos: {len(existing_videos)}")
print(f"Missing videos:  {len(missing_videos)}")

if missing_videos:
    print("\nFirst missing videos:")

    for path in missing_videos[:20]:
        print(f"  {path}")

    if len(missing_videos) > 20:
        print(f"  ... and {len(missing_videos) - 20} more")

videos = existing_videos



print()
print("SEANet visual embedding generation")
print("==================================")
print(f"Input root:      {video_root}")
print(f"Output root:     {output_root}")
print(f"Frontend:        {args.visual_frontend}")
print(f"Source FPS:      {args.source_fps}")
print(f"Target FPS:      {args.fps}")
print(f"Videos:          {len(videos)}")
print(f"Device:          {device}")
print()


# ============================================================
# Main loop
# ============================================================

success = 0
skipped = 0
failed = 0


for video_path in tqdm(
    videos,
    desc=f"Visual {args.fps:g} FPS",
    dynamic_ncols=True,
):

    relative = (
        video_path
        .relative_to(video_root)
        .with_suffix(".npy")
    )

    output_path = (
        output_root
        / relative
    )

    # --------------------------------------------------------
    # Existing file
    # --------------------------------------------------------

    if (
        output_path.exists()
        and not args.overwrite
    ):
        skipped += 1
        continue

    try:

        stats = process_video(
            video_path,
            output_path,
        )

        success += 1

    except Exception as exc:

        failed += 1

        tqdm.write(
            f"\nFAILED: {video_path}\n"
            f"  {type(exc).__name__}: {exc}\n"
        )


# ============================================================
# Summary
# ============================================================

print()
print("Finished")
print("========")
print(f"Successful: {success}")
print(f"Skipped:    {skipped}")
print(f"Failed:     {failed}")
print(f"Output:     {output_root}")