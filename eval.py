#!/usr/bin/env python3

import argparse
import csv
import os
import re
import time
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from dataLoader import init_loader
from loss import loss_speech

from model.dprnn import dprnn
from model.muse import muse
from model.avsep import avsep
from model.seanet import seanet


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate an audio-visual target speaker extraction model "
        "and optionally save WAV examples."
    )
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model-only checkpoint, e.g. model_0136.model",
)

parser.add_argument(
    "--backbone",
    type=str,
    required=True,
    choices=[
        "seanet",
        "dprnn",
        "muse",
        "avsep",
    ],
)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

parser.add_argument(
    "--data_list",
    type=str,
    required=True,
)

parser.add_argument(
    "--audio_path",
    type=str,
    required=True,
)

parser.add_argument(
    "--visual_path",
    type=str,
    required=True,
)

parser.add_argument(
    "--musan_path",
    type=str,
    default="",
)

parser.add_argument(
    "--length",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=6,
)

parser.add_argument(
    "--n_cpu",
    type=int,
    default=12,
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

parser.add_argument(
    "--output_dir",
    type=str,
    default="eval_outputs",
    help="Directory where evaluation results are saved",
)

parser.add_argument(
    "--save_audio",
    type=int,
    default=20,
    help=(
        "Number of test examples for which WAV files are saved. "
        "Use 0 to disable."
    ),
)


# ------------------------------------------------------------
# Precision / performance
# ------------------------------------------------------------

parser.add_argument(
    "--precision",
    choices=[
        "fp32",
        "fp16",
        "bf16",
    ],
    default="bf16",
)

parser.add_argument(
    "--compile",
    action="store_true",
    help="Enable torch.compile",
)

parser.add_argument(
    "--compile_mode",
    choices=[
        "default",
        "reduce-overhead",
        "max-autotune",
    ],
    default="default",
)

parser.add_argument(
    "--tf32",
    action=argparse.BooleanOptionalAction,
    default=True,
)


args = parser.parse_args()


# ============================================================
# Basic checks
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required by this evaluation script."
    )

if not os.path.isfile(args.model):
    raise FileNotFoundError(
        f"Model checkpoint not found: {args.model}"
    )

if not os.path.isfile(args.data_list):
    raise FileNotFoundError(
        f"Dataset list not found: {args.data_list}"
    )

os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# Performance configuration
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = args.tf32
torch.backends.cudnn.allow_tf32 = args.tf32
torch.backends.cudnn.benchmark = True

if args.tf32:

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass


# ============================================================
# Precision
# ============================================================

if args.precision == "bf16":

    amp_dtype = torch.bfloat16
    use_amp = True

elif args.precision == "fp16":

    amp_dtype = torch.float16
    use_amp = True

else:

    amp_dtype = None
    use_amp = False


def autocast_context():

    if not use_amp:
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
    )


# ============================================================
# Build model
# ============================================================

def build_model():

    if args.backbone == "seanet":

        model = seanet(
            256,
            40,
            64,
            128,
            100,
            6,
        )

    elif args.backbone == "dprnn":

        model = dprnn()

    elif args.backbone == "muse":

        model = muse(
            M=800
        )

    elif args.backbone == "avsep":

        model = avsep()

    else:

        raise ValueError(
            f"Unknown backbone: {args.backbone}"
        )

    return model.cuda()


# ============================================================
# Load model-only checkpoint
#
# Compatible with:
#
#   model.xxx
#   model._orig_mod.xxx
#   _orig_mod.xxx
#   xxx
#
# ============================================================

def load_model_checkpoint(
    model,
    checkpoint_path,
):

    print(
        f"Loading model: {checkpoint_path}"
    )

    loaded_state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Also tolerate a full checkpoint if one is accidentally
    # passed here.
    if (
        isinstance(loaded_state, dict)
        and "model" in loaded_state
        and isinstance(
            loaded_state["model"],
            dict,
        )
    ):

        print(
            "Full training checkpoint detected; "
            "loading model weights only."
        )

        loaded_state = loaded_state["model"]

    model_state = model.state_dict()

    cleaned_state = {}

    for name, param in loaded_state.items():

        original_name = name

        if name.startswith(
            "model._orig_mod."
        ):

            name = name[
                len("model._orig_mod.") :
            ]

        elif name.startswith(
            "_orig_mod."
        ):

            name = name[
                len("_orig_mod.") :
            ]

        elif name.startswith(
            "model."
        ):

            name = name[
                len("model.") :
            ]

        if name not in model_state:

            print(
                f"Skipping unknown parameter: "
                f"{original_name}"
            )

            continue

        if (
            model_state[name].shape
            != param.shape
        ):

            print(
                f"Skipping shape mismatch: "
                f"{original_name}: "
                f"{param.shape} != "
                f"{model_state[name].shape}"
            )

            continue

        cleaned_state[name] = param

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state,
            strict=False,
        )
    )

    print(
        f"Loaded {len(cleaned_state)} tensors."
    )

    if missing:

        print(
            f"WARNING: {len(missing)} "
            f"missing parameters."
        )

        for name in missing[:20]:
            print(f"  {name}")

    if unexpected:

        print(
            f"WARNING: {len(unexpected)} "
            f"unexpected parameters."
        )

    if len(cleaned_state) == 0:

        raise RuntimeError(
            "No model parameters were loaded."
        )


# ============================================================
# Read test sample names from CSV
#
# SEANet CSV:
#
#   data[0] = split
#   data[2] = target speaker
#   data[3] = target utterance
#
# label_name:
#
#   data[2] + "/" + data[3]
#
# This is used ONLY for naming the saved examples.
# ============================================================

def load_test_sample_names(
    csv_path,
):

    samples = []

    with open(
        csv_path,
        "r",
        newline="",
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if len(row) < 4:
                continue

            split = (
                row[0]
                .strip()
                .lower()
            )

            if split != "test":
                continue

            speaker = (
                row[2]
                .strip()
            )

            utterance = (
                row[3]
                .strip()
            )

            target_name = os.path.join(
                speaker,
                utterance,
            )

            samples.append(
                target_name
            )

    return samples


# ============================================================
# Safe directory name
# ============================================================

def sanitize_sample_name(
    name,
):

    name = str(name)

    name = name.replace(
        "\\",
        "__",
    )

    name = name.replace(
        "/",
        "__",
    )

    name = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        name,
    )

    if name.lower().endswith(
        ".wav"
    ):

        name = name[:-4]

    return name


# ============================================================
# Save WAV
# ============================================================

def save_wav(
    path,
    tensor,
):

    audio = (
        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    # Remove possible singleton dimensions.
    audio = np.squeeze(audio)

    sf.write(
        path,
        audio,
        16000,
        subtype="PCM_16",
    )


# ============================================================
# Forward
# ============================================================

def forward_model(
    model,
    audio,
    face,
):

    current_B = audio.shape[0]

    with autocast_context():

        if args.backbone == "seanet":

            out_speech, _ = model(
                audio,
                face,
                M=current_B,
            )

            estimate = out_speech[
                -current_B:,
                :
            ]

        elif args.backbone == "dprnn":

            out_speech = model(
                audio,
                face,
                M=current_B,
            )

            estimate = out_speech[
                -current_B:,
                :
            ]

        elif args.backbone == "muse":

            _, estimate = model(
                audio,
                face,
            )

        elif args.backbone == "avsep":

            estimate = model(
                audio,
                face,
            )

        else:

            raise ValueError(
                f"Unknown backbone: "
                f"{args.backbone}"
            )

    return estimate


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    Loader,
    test_names,
):

    criterion = loss_speech().cuda()

    model.eval()

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    sisdr_total = 0.0
    sdr_total = 0.0

    sisdri_total = 0.0
    sdri_total = 0.0

    num_batches = 0
    num_samples = 0
    num_saved = 0

    dataset_offset = 0

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.inference_mode():

        pbar = tqdm(
            Loader,
            total=len(Loader),
            desc="Test",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in pbar:

            # Existing SEANet loader returns:
            #
            # audio
            # face
            # speech
            # noise
            # muse_label

            if len(batch) < 5:

                raise RuntimeError(
                    "Unexpected test loader output."
                )

            (
                audio,
                face,
                speech,
                noise,
                _,
            ) = batch[:5]

            current_B = audio.shape[0]

            # ------------------------------------------------
            # GPU
            # ------------------------------------------------

            audio = audio.cuda(
                non_blocking=True
            )

            face = face.cuda(
                non_blocking=True
            )

            speech = speech.cuda(
                non_blocking=True
            )

            noise = noise.cuda(
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            estimate = forward_model(
                model,
                audio,
                face,
            )

            # ------------------------------------------------
            # Metrics in FP32
            # ------------------------------------------------

            estimate_fp32 = (
                estimate.float()
            )

            target_fp32 = (
                speech.float()
            )

            mixture_fp32 = (
                audio.float()
            )

            result_estimate = (
                criterion.forward_eval_light(
                    estimate_fp32,
                    target_fp32,
                )
            )

            result_mixture = (
                criterion.forward_eval_light(
                    mixture_fp32,
                    target_fp32,
                )
            )

            batch_sisdr = (
                result_estimate[
                    "sisdr"
                ]
                .detach()
                .float()
                .item()
            )

            batch_sdr = (
                result_estimate[
                    "sdr"
                ]
                .detach()
                .float()
                .item()
            )

            mixture_sisdr = (
                result_mixture[
                    "sisdr"
                ]
                .detach()
                .float()
                .item()
            )

            mixture_sdr = (
                result_mixture[
                    "sdr"
                ]
                .detach()
                .float()
                .item()
            )

            batch_sisdri = (
                batch_sisdr
                - mixture_sisdr
            )

            batch_sdri = (
                batch_sdr
                - mixture_sdr
            )

            sisdr_total += batch_sisdr
            sdr_total += batch_sdr

            sisdri_total += batch_sisdri
            sdri_total += batch_sdri

            num_batches += 1
            num_samples += current_B

            # ------------------------------------------------
            # Save examples
            # ------------------------------------------------

            for i in range(current_B):

                global_index = (
                    dataset_offset
                    + i
                )

                if (
                    num_saved
                    >= args.save_audio
                ):

                    break

                # --------------------------------------------
                # Original target name
                # --------------------------------------------

                if (
                    global_index
                    < len(test_names)
                ):

                    original_name = (
                        test_names[
                            global_index
                        ]
                    )

                else:

                    original_name = (
                        f"sample_{global_index:06d}"
                    )

                safe_name = (
                    sanitize_sample_name(
                        original_name
                    )
                )

                # Add dataset index to guarantee uniqueness.
                sample_dir_name = (
                    f"{global_index:06d}_"
                    f"{safe_name}"
                )

                sample_dir = os.path.join(
                    args.output_dir,
                    sample_dir_name,
                )

                os.makedirs(
                    sample_dir,
                    exist_ok=True,
                )

                # --------------------------------------------
                # WAV files
                # --------------------------------------------

                save_wav(
                    os.path.join(
                        sample_dir,
                        "mixture.wav",
                    ),
                    audio[i],
                )

                save_wav(
                    os.path.join(
                        sample_dir,
                        "estimate.wav",
                    ),
                    estimate_fp32[i],
                )

                save_wav(
                    os.path.join(
                        sample_dir,
                        "target.wav",
                    ),
                    target_fp32[i],
                )

                save_wav(
                    os.path.join(
                        sample_dir,
                        "interferer.wav",
                    ),
                    noise[i],
                )

                # --------------------------------------------
                # Per-sample metrics
                # --------------------------------------------

                sample_estimate = (
                    criterion.forward_eval_light(
                        estimate_fp32[
                            i:i + 1
                        ],
                        target_fp32[
                            i:i + 1
                        ],
                    )
                )

                sample_mixture = (
                    criterion.forward_eval_light(
                        mixture_fp32[
                            i:i + 1
                        ],
                        target_fp32[
                            i:i + 1
                        ],
                    )
                )

                sample_sisdr = (
                    sample_estimate[
                        "sisdr"
                    ]
                    .detach()
                    .float()
                    .item()
                )

                sample_sdr = (
                    sample_estimate[
                        "sdr"
                    ]
                    .detach()
                    .float()
                    .item()
                )

                sample_input_sisdr = (
                    sample_mixture[
                        "sisdr"
                    ]
                    .detach()
                    .float()
                    .item()
                )

                sample_input_sdr = (
                    sample_mixture[
                        "sdr"
                    ]
                    .detach()
                    .float()
                    .item()
                )

                sample_sisdri = (
                    sample_sisdr
                    - sample_input_sisdr
                )

                sample_sdri = (
                    sample_sdr
                    - sample_input_sdr
                )

                # --------------------------------------------
                # Information
                # --------------------------------------------

                info_path = os.path.join(
                    sample_dir,
                    "info.txt",
                )

                with open(
                    info_path,
                    "w",
                ) as f:

                    f.write(
                        f"dataset_index: "
                        f"{global_index}\n"
                    )

                    f.write(
                        f"target_original: "
                        f"{original_name}\n"
                    )

                    f.write(
                        f"SI-SDR: "
                        f"{sample_sisdr:.4f} dB\n"
                    )

                    f.write(
                        f"SDR: "
                        f"{sample_sdr:.4f} dB\n"
                    )

                    f.write(
                        f"Input SI-SDR: "
                        f"{sample_input_sisdr:.4f} dB\n"
                    )

                    f.write(
                        f"Input SDR: "
                        f"{sample_input_sdr:.4f} dB\n"
                    )

                    f.write(
                        f"SI-SDRi: "
                        f"{sample_sisdri:.4f} dB\n"
                    )

                    f.write(
                        f"SDRi: "
                        f"{sample_sdri:.4f} dB\n"
                    )

                num_saved += 1

            dataset_offset += current_B

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            mean_sisdr = (
                sisdr_total
                / num_batches
            )

            mean_sdr = (
                sdr_total
                / num_batches
            )

            mean_sisdri = (
                sisdri_total
                / num_batches
            )

            pbar.set_postfix(
                sisdr=f"{mean_sisdr:.3f}",
                sdr=f"{mean_sdr:.3f}",
                sisdri=f"{mean_sisdri:.3f}",
                saved=(
                    f"{num_saved}/"
                    f"{args.save_audio}"
                ),
            )

    # ========================================================
    # Final results
    # ========================================================

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    mean_sisdr = (
        sisdr_total
        / max(num_batches, 1)
    )

    mean_sdr = (
        sdr_total
        / max(num_batches, 1)
    )

    mean_sisdri = (
        sisdri_total
        / max(num_batches, 1)
    )

    mean_sdri = (
        sdri_total
        / max(num_batches, 1)
    )

    results = (
        "Test results\n"
        "============\n"
        f"Model:      {args.model}\n"
        f"Backbone:   {args.backbone}\n"
        f"Samples:    {num_samples}\n"
        f"Batches:    {num_batches}\n"
        f"Time:       {elapsed / 60.0:.2f} min\n"
        "\n"
        f"SI-SDR:     {mean_sisdr:.4f} dB\n"
        f"SDR:        {mean_sdr:.4f} dB\n"
        f"SI-SDRi:    {mean_sisdri:.4f} dB\n"
        f"SDRi:       {mean_sdri:.4f} dB\n"
        "\n"
        f"Saved WAV:  {num_saved}\n"
    )

    print()
    print(results)

    results_path = os.path.join(
        args.output_dir,
        "results.txt",
    )

    with open(
        results_path,
        "w",
    ) as f:

        f.write(results)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("SEANet evaluation")
    print("==================")
    print(f"Model:       {args.model}")
    print(f"Backbone:    {args.backbone}")
    print(f"Precision:   {args.precision}")
    print(f"TF32:        {args.tf32}")
    print(f"Compile:     {args.compile}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Save audio:  {args.save_audio}")
    print(f"Output:      {args.output_dir}")
    print()

    # ========================================================
    # Model
    # ========================================================

    model = build_model()

    load_model_checkpoint(
        model,
        args.model,
    )

    # Compile AFTER loading weights.
    if args.compile:

        print(
            f"Compiling model "
            f"(mode={args.compile_mode})..."
        )

        model = torch.compile(
            model,
            mode=args.compile_mode,
        )

    # ========================================================
    # Dataset
    # ========================================================

    # init_loader() expects the same args used by main.py.
    #
    # These dummy/default attributes are included in case the
    # existing loader expects them.

    args.eval = True
    args.epoch = 1

    print(
        "Initializing test DataLoader..."
    )

    loader_args = init_loader(
        args
    )

    if not hasattr(
        loader_args,
        "testLoader",
    ):

        raise RuntimeError(
            "init_loader() did not create args.testLoader."
        )

    test_loader = (
        loader_args.testLoader
    )

    print(
        f"Test dataset samples: "
        f"{len(test_loader.dataset)}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # ========================================================
    # Original sample names
    # ========================================================

    test_names = load_test_sample_names(
        args.data_list
    )

    print(
        f"Test names from CSV: "
        f"{len(test_names)}"
    )

    if (
        len(test_names)
        != len(test_loader.dataset)
    ):

        print()
        print(
            "WARNING: number of test rows in CSV "
            "does not match test dataset size."
        )

        print(
            "WAV evaluation is still valid, but "
            "saved source names may not align."
        )

        print()

    # ========================================================
    # Evaluation
    # ========================================================

    evaluate(
        model,
        test_loader,
        test_names,
    )


if __name__ == "__main__":
    main()