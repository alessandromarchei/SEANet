#!/usr/bin/env python3

import os
import sys
import argparse
import importlib.util
from argparse import Namespace
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# PATHS / IMPORTS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent


def add_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(module_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load {module_name} from {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate per-frame AV-HuBERT embeddings directly "
            "from Kai/Dolphin mouth ROI .npz files."
        )
    )

    parser.add_argument(
        "--mouth-root",
        required=True,
        type=Path,
        help=(
            "Root containing mouth .npz files. "
            "Each file must contain data=[T,H,W]."
        ),
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help=(
            "Output root. One float32 [T,D] .npy "
            "will be generated for each mouth .npz."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="AV-HuBERT checkpoint, e.g. large_vox_433h.pt",
    )

    parser.add_argument(
        "--avhubert-root",
        required=True,
        type=Path,
        help="Root of the av_hubert repository.",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Maximum number of same-length utterances "
            "processed together."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="0 = all files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no-buckets",
        action="store_true",
        help="Process one utterance at a time.",
    )

    return parser.parse_args()


# ============================================================
# PYTORCH LEGACY CHECKPOINT COMPATIBILITY
# ============================================================

def patch_torch_load():

    original = torch.load

    if getattr(
        original,
        "_avhubert_legacy_patch",
        False,
    ):
        return

    def wrapped(*args, **kwargs):
        kwargs.setdefault(
            "weights_only",
            False,
        )
        return original(
            *args,
            **kwargs,
        )

    wrapped._avhubert_legacy_patch = True

    torch.load = wrapped


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    mouth_root = args.mouth_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    avhubert_root = args.avhubert_root.resolve()

    user_dir = (
        avhubert_root
        / "avhubert"
    )

    fairseq_root = (
        avhubert_root
        / "fairseq"
    )

    avhubert_utils_path = (
        user_dir
        / "utils.py"
    )

    if not mouth_root.is_dir():
        raise FileNotFoundError(
            f"Mouth root not found: {mouth_root}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    if not user_dir.is_dir():
        raise FileNotFoundError(
            f"AV-HuBERT user dir not found: {user_dir}"
        )

    add_path(avhubert_root)
    add_path(fairseq_root)

    avhubert_utils = load_module_from_path(
        "avhubert_local_utils",
        avhubert_utils_path,
    )

    from fairseq import checkpoint_utils
    from fairseq import utils as fairseq_utils

    patch_torch_load()

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but CUDA is unavailable."
        )

    if device.type == "cuda":
        torch.cuda.set_device(
            device
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD AV-HUBERT ONCE
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING AV-HUBERT")
    print("=" * 80)

    fairseq_utils.import_user_module(
        Namespace(
            user_dir=str(user_dir)
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

    # Same logic as Dolphin.
    if hasattr(model, "decoder"):

        print(
            "Fine-tuned checkpoint detected."
        )

        print(
            "Using model.encoder.w2v_model"
        )

        model = (
            model
            .encoder
            .w2v_model
        )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # ========================================================
    # EXACT DOLPHIN / AV-HUBERT TRANSFORM
    # ========================================================

    crop_size = int(
        task.cfg.image_crop_size
    )

    image_mean = float(
        task.cfg.image_mean
    )

    image_std = float(
        task.cfg.image_std
    )

    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(
            0.0,
            255.0,
        ),

        avhubert_utils.CenterCrop(
            (
                crop_size,
                crop_size,
            )
        ),

        avhubert_utils.Normalize(
            image_mean,
            image_std,
        ),
    ])

    print()
    print("Checkpoint")
    print("-" * 80)
    print(checkpoint)

    print()
    print("Preprocessing")
    print("-" * 80)
    print(
        f"Center crop : {crop_size}x{crop_size}"
    )
    print(
        f"Mean        : {image_mean}"
    )
    print(
        f"Std         : {image_std}"
    )
    print(
        f"Device      : {device}"
    )

    # ========================================================
    # FIND MOUTH FILES
    # ========================================================

    all_files = sorted(
        mouth_root.rglob("*.npz")
    )

    if not all_files:
        raise RuntimeError(
            f"No .npz files found under {mouth_root}"
        )

    todo = []

    already_done = 0

    for path in all_files:

        relative = (
            path
            .relative_to(mouth_root)
            .with_suffix(".npy")
        )

        output_path = (
            output_root
            / relative
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            already_done += 1
            continue

        todo.append(path)

    if args.max_files > 0:
        todo = todo[
            :args.max_files
        ]

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)
    print(
        f"Mouth root    : {mouth_root}"
    )
    print(
        f"Output root   : {output_root}"
    )
    print(
        f"Files found   : {len(all_files)}"
    )
    print(
        f"Already done  : {already_done}"
    )
    print(
        f"To process    : {len(todo)}"
    )
    print(
        f"Batch size    : {args.batch_size}"
    )

    if not todo:
        print("\nNothing to do.")
        return

    # ========================================================
    # READ MOUTH
    # ========================================================

    def load_mouth(path):

        with np.load(
            path,
            allow_pickle=False,
        ) as npz:

            if "data" not in npz.files:
                raise RuntimeError(
                    f"{path}: missing key 'data'. "
                    f"Keys={npz.files}"
                )

            frames = npz["data"]

        if frames.ndim != 3:
            raise RuntimeError(
                f"{path}: expected [T,H,W], "
                f"got {frames.shape}"
            )

        if (
            frames.shape[1] < crop_size
            or frames.shape[2] < crop_size
        ):
            raise RuntimeError(
                f"{path}: ROI {frames.shape[1:]} "
                f"is smaller than AV-HuBERT "
                f"crop {crop_size}x{crop_size}"
            )

        return frames

    # ========================================================
    # SCAN TEMPORAL LENGTHS
    #
    # Kai mouths are often all T=50, which is ideal:
    # this will become essentially one giant bucket.
    # ========================================================

    buckets = defaultdict(list)

    failures = []

    print()
    print("Scanning mouth shapes...")

    for path in tqdm(
        todo,
        desc="Scan",
        dynamic_ncols=True,
    ):

        try:

            with np.load(
                path,
                allow_pickle=False,
            ) as npz:

                if "data" not in npz.files:
                    raise RuntimeError(
                        "missing key 'data'"
                    )

                shape = npz["data"].shape

            if len(shape) != 3:
                raise RuntimeError(
                    f"expected [T,H,W], got {shape}"
                )

            T, H, W = shape

            if (
                H < crop_size
                or W < crop_size
            ):
                raise RuntimeError(
                    f"ROI {H}x{W} smaller than "
                    f"{crop_size}x{crop_size}"
                )

            buckets[int(T)].append(
                path
            )

        except Exception as exc:

            failures.append(
                (
                    str(path),
                    str(exc),
                )
            )

    print()
    print(
        f"Temporal buckets : {len(buckets)}"
    )

    for T in sorted(buckets)[:20]:
        print(
            f"  T={T:<5} : "
            f"{len(buckets[T]):>8} files"
        )

    # ========================================================
    # PREPARE BATCH
    # ========================================================

    def prepare_batch(paths):

        transformed = []

        expected_T = None

        for path in paths:

            frames = load_mouth(
                path
            )

            if expected_T is None:
                expected_T = frames.shape[0]

            if frames.shape[0] != expected_T:
                raise RuntimeError(
                    "Mixed temporal lengths in batch."
                )

            # -----------------------------------------------
            # EXACT same preprocessing as Dolphin script:
            #
            # torch.tensor(np.load(path)["data"])
            # transform(frames)
            #
            # [T,96,96]
            #      ↓
            # [T,88,88]
            # -----------------------------------------------

            frames = torch.tensor(
                frames,
                dtype=torch.float32,
            )

            frames = transform(
                frames
            )

            transformed.append(
                frames
            )

        # [B,T,H,W]
        batch = torch.stack(
            transformed,
            dim=0,
        )

        # AV-HuBERT:
        #
        # [B,1,T,H,W]
        batch = (
            batch
            .unsqueeze(1)
            .contiguous()
        )

        return batch.to(
            device,
            non_blocking=True,
        )

    # ========================================================
    # AV-HUBERT INFERENCE
    # ========================================================

    @torch.inference_mode()
    def extract_batch(paths):

        video = prepare_batch(
            paths
        )

        B = video.shape[0]

        feature, _ = model.extract_finetune(
            source={
                "video": video,
                "audio": None,
            },
            padding_mask=None,
            output_layer=None,
        )

        if feature.ndim != 3:
            raise RuntimeError(
                "Expected AV-HuBERT output "
                f"[B,T,D], got {tuple(feature.shape)}"
            )

        if feature.shape[0] != B:
            raise RuntimeError(
                "AV-HuBERT batch dimension mismatch: "
                f"{feature.shape[0]} != {B}"
            )

        return (
            feature
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save_embedding(
        mouth_path,
        embedding,
    ):

        relative = (
            mouth_path
            .relative_to(mouth_root)
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

        np.save(
            output_path,
            embedding.astype(
                np.float32,
                copy=False,
            ),
        )

    # ========================================================
    # PROCESS
    # ========================================================

    total_valid = sum(
        len(paths)
        for paths in buckets.values()
    )

    success = 0
    embedding_dim = None

    print()
    print("=" * 80)
    print("AV-HUBERT INFERENCE")
    print("=" * 80)
    print()

    with tqdm(
        total=total_valid,
        desc="AV-HuBERT",
        dynamic_ncols=True,
    ) as progress:

        for T in sorted(buckets):

            paths = buckets[T]

            batch_size = (
                1
                if args.no_buckets
                else args.batch_size
            )

            for start in range(
                0,
                len(paths),
                batch_size,
            ):

                batch_paths = paths[
                    start:
                    start + batch_size
                ]

                try:

                    features = extract_batch(
                        batch_paths
                    )

                    embedding_dim = (
                        features.shape[-1]
                    )

                    for path, feature in zip(
                        batch_paths,
                        features,
                    ):

                        save_embedding(
                            path,
                            feature,
                        )

                        success += 1

                except torch.cuda.OutOfMemoryError:

                    torch.cuda.empty_cache()

                    print(
                        "\nCUDA OOM for batch of "
                        f"{len(batch_paths)} files "
                        f"with T={T}."
                    )

                    print(
                        "Retrying individually..."
                    )

                    for path in batch_paths:

                        try:

                            feature = extract_batch(
                                [path]
                            )[0]

                            embedding_dim = (
                                feature.shape[-1]
                            )

                            save_embedding(
                                path,
                                feature,
                            )

                            success += 1

                        except Exception as exc:

                            failures.append(
                                (
                                    str(path),
                                    repr(exc),
                                )
                            )

                except Exception as exc:

                    # Retry individually so one corrupt file
                    # does not destroy the whole batch.

                    for path in batch_paths:

                        try:

                            feature = extract_batch(
                                [path]
                            )[0]

                            embedding_dim = (
                                feature.shape[-1]
                            )

                            save_embedding(
                                path,
                                feature,
                            )

                            success += 1

                        except Exception as sub_exc:

                            failures.append(
                                (
                                    str(path),
                                    repr(sub_exc),
                                )
                            )

                progress.update(
                    len(batch_paths)
                )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Generated     : {success}"
    )
    print(
        f"Failed        : {len(failures)}"
    )
    print(
        f"Embedding dim : {embedding_dim}"
    )

    if failures:

        failure_path = (
            output_root
            / "failed_avhubert.txt"
        )

        with failure_path.open(
            "w"
        ) as f:

            for path, error in failures:

                f.write(
                    f"{path}\t{error}\n"
                )

        print(
            f"Failure log   : {failure_path}"
        )


if __name__ == "__main__":
    main()