#!/usr/bin/env python3

import argparse
import importlib
import math
import os
import sys

import torch
import torch.nn as nn

from ptflops import get_model_complexity_info

# ============================================================
# Add SEANet repository root to Python path
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)




class AVModelWrapper(nn.Module):
    """
    Wrapper used only for ptflops.

    It gives ptflops a simple forward signature while preserving
    positional arguments for the original AV-TSE model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, av_inputs):
        audio, visual = av_inputs
        return self.model(audio, visual)

        
# ============================================================
# Utilities
# ============================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


def n_audio_latents(
    duration,
    sample_rate=16000,
    kernel_size=40,
    stride=20,
):
    samples = int(round(duration * sample_rate))

    return (
        math.floor(
            (samples - kernel_size) / stride
        ) + 1
    )


def human_number(x):
    if x >= 1e9:
        return f"{x / 1e9:.3f} G"
    if x >= 1e6:
        return f"{x / 1e6:.3f} M"
    if x >= 1e3:
        return f"{x / 1e3:.3f} K"

    return str(x)


# ============================================================
# Model loader
# ============================================================

def load_av_model(name):

    name = name.lower()

    candidates = {
        "muse": [
            ("model.muse", "muse"),
        ],

        "seanet": [
            ("model.seanet", "seanet"),
        ],

        "dprnn": [
            ("model.dprnn", "dprnn"),
        ],

        "av_sepformer": [
            ("model.avsep", "avsep"),
        ],
    }

    if name not in candidates:
        raise ValueError(
            f"Unknown model: {name}"
        )

    errors = []

    for module_name, class_name in candidates[name]:

        try:
            module = importlib.import_module(module_name)

            if not hasattr(module, class_name):
                continue

            cls = getattr(module, class_name)

            model = cls()

            print(
                f"[OK] Loaded {module_name}.{class_name}"
            )

            return model

        except Exception as e:
            errors.append(
                f"{module_name}.{class_name}: {e}"
            )

    raise RuntimeError(
        "Could not instantiate model.\n" +
        "\n".join(errors)
    )


# ============================================================
# Visual frontend
# ============================================================

def load_visual_frontend(weights=None):

    try:
        from pretrain_networks.visual_frontend import VisualFrontend
    except ImportError:

        raise RuntimeError(
            "\nCould not import VisualFrontend.\n\n"
            "Copy the MuSE pretrain_networks directory into "
            "the SEANet repository:\n\n"
            "SEANet/pretrain_networks/\n"
            "    visual_frontend.py\n"
            "    visual_frontend.pt\n"
        )

    model = VisualFrontend()

    if weights is not None:

        state = torch.load(
            weights,
            map_location="cpu"
        )

        model.load_state_dict(state)

    return model

# ============================================================
# AV model
# ============================================================
def profile_av_model(
    model,
    duration,
    fps,
    sample_rate,
    device,
):

    n_samples = int(
        round(duration * sample_rate)
    )

    n_frames = max(
        1,
        int(round(duration * fps))
    )

    model = model.to(device).eval()

    wrapped_model = AVModelWrapper(model)
    wrapped_model = wrapped_model.to(device).eval()

    def input_constructor(input_res):

        audio = torch.randn(
            1,
            n_samples,
            device=device
        )

        visual = torch.randn(
            1,
            n_frames,
            512,
            device=device
        )

        # ptflops will call:
        #
        # wrapped_model(av_inputs=(audio, visual))
        #
        # and the wrapper will call:
        #
        # original_model(audio, visual)

        return {
            "av_inputs": (audio, visual)
        }

    with torch.no_grad():

        macs, params = get_model_complexity_info(
            wrapped_model,
            (1,),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=True,
            backend="pytorch",
        )

    if macs is None:
        raise RuntimeError(
            "ptflops failed to compute MACs. "
            "Check the error printed above."
        )

    return {
        "macs": macs,
        "gmacs": macs / 1e9,
        "gmacs_per_second": (
            macs / duration / 1e9
        ),
        "frames": n_frames,
        "samples": n_samples,
        "params_ptflops": params,
    }


# ============================================================
# Visual frontend wrapper
# ============================================================

class VisualFrontendWrapper(nn.Module):

    def __init__(self, frontend):
        super().__init__()

        self.frontend = frontend

    def forward(self, x):
        return self.frontend(x)


def profile_visual_frontend(
    frontend,
    duration,
    fps,
    device,
    roi_size=112,
):

    n_frames = max(
        1,
        int(round(duration * fps))
    )

    frontend = frontend.to(device).eval()

    # VisualFrontend expects:
    #
    # [T, B, C, H, W]
    #
    # and internally converts it to:
    #
    # [B, C, T, H, W]
    #
    # Therefore:
    #
    # [n_frames, 1, 1, 112, 112]

    def input_constructor(input_res):

        video = torch.randn(
            n_frames,
            1,
            1,
            roi_size,
            roi_size,
            device=device
        )

        # VisualFrontend.forward(inputBatch)
        return {
            "inputBatch": video
        }

    with torch.no_grad():

        macs, params = get_model_complexity_info(
            frontend,
            (1,),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=True,
            backend="pytorch",
        )

    return {
        "macs": macs,
        "gmacs": macs / 1e9,
        "gmacs_per_second":
            macs / duration / 1e9,
        "frames": n_frames,
        "params_ptflops": params,
    }

# ============================================================
# Pretty printing
# ============================================================

def print_header(
    model_name,
    duration,
    fps,
    sample_rate,
):

    samples = int(duration * sample_rate)

    visual_frames = int(
        round(duration * fps)
    )

    audio_latents = n_audio_latents(
        duration,
        sample_rate
    )

    print()
    print("=" * 76)
    print(f"MODEL              : {model_name}")
    print(f"Duration           : {duration:.3f} s")
    print(f"Sample rate        : {sample_rate} Hz")
    print(f"Audio samples      : {samples}")
    print(f"Visual FPS         : {fps}")
    print(f"Visual frames      : {visual_frames}")
    print(f"Audio latent steps : {audio_latents}")
    print(
        f"Visual/audio ratio : "
        f"{audio_latents / max(visual_frames,1):.2f}"
    )
    print("=" * 76)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "muse",
            "dprnn",
            "av_sepformer",
            "seanet",
        ],
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Audio duration in seconds",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--fps",
        nargs="+",
        type=float,
        default=[25.0],
    )

    parser.add_argument(
        "--visual-frontend",
        type=str,
        default=None,
        help="Path to visual_frontend.pt",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # --------------------------------------------------------
    # Load DeepAVSR frontend once
    # --------------------------------------------------------

    visual_frontend = None

    if args.visual_frontend:

        visual_frontend = load_visual_frontend(
            args.visual_frontend
        )

        vf_total, vf_trainable = count_params(
            visual_frontend
        )

        print()
        print("DeepAVSR Visual Frontend")
        print("-" * 76)
        print(
            f"Parameters : "
            f"{vf_total:,} "
            f"({vf_total / 1e6:.3f} M)"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for model_name in args.models:

        model = load_av_model(model_name)

        total_params, trainable_params = (
            count_params(model)
        )

        print()
        print("#" * 76)
        print(
            f"# {model_name.upper()}"
        )
        print("#" * 76)

        print(
            f"Backend parameters : "
            f"{total_params:,} "
            f"({total_params / 1e6:.3f} M)"
        )

        for fps in args.fps:

            print_header(
                model_name,
                args.duration,
                fps,
                args.sample_rate,
            )

            # -----------------------------------------------
            # AV-TSE backend
            # -----------------------------------------------

            backend = profile_av_model(
                model=model,
                duration=args.duration,
                fps=fps,
                sample_rate=args.sample_rate,
                device=device,
            )

            print()
            print("AV-TSE BACKEND")
            print("-" * 76)

            print(
                f"GMACs / input       : "
                f"{backend['gmacs']:.4f}"
            )

            print(
                f"GMACs / second      : "
                f"{backend['gmacs_per_second']:.4f}"
            )

            # if backend["unsupported"]:
            #     print(
            #         "Unsupported ops      : "
            #         f"{backend['unsupported']}"
            #     )

            # -----------------------------------------------
            # Visual frontend
            # -----------------------------------------------

            if visual_frontend is not None:

                vf = profile_visual_frontend(
                    frontend=visual_frontend,
                    duration=args.duration,
                    fps=fps,
                    device=device,
                )

                print()
                print("DEEPAVSR VISUAL FRONTEND")
                print("-" * 76)

                print(
                    f"Frames              : "
                    f"{vf['frames']}"
                )

                print(
                    f"GMACs / input       : "
                    f"{vf['gmacs']:.4f}"
                )

                print(
                    f"GMACs / second      : "
                    f"{vf['gmacs_per_second']:.4f}"
                )

                print(
                    f"GMACs / frame avg.  : "
                    f"{vf['macs'] / vf['frames'] / 1e9:.4f}"
                )

                # if vf["unsupported"]:
                #     print(
                #         "Unsupported ops      : "
                #         f"{vf['unsupported']}"
                #     )

                # -------------------------------------------
                # Total
                # -------------------------------------------

                total_gmacs = (
                    backend["gmacs"]
                    +
                    vf["gmacs"]
                )

                total_gmacs_s = (
                    backend["gmacs_per_second"]
                    +
                    vf["gmacs_per_second"]
                )

                total_params_e2e = (
                    total_params
                    +
                    vf_total
                )

                print()
                print("END-TO-END")
                print("-" * 76)

                print(
                    f"Total parameters    : "
                    f"{total_params_e2e:,} "
                    f"({total_params_e2e / 1e6:.3f} M)"
                )

                print(
                    f"Total GMACs/input   : "
                    f"{total_gmacs:.4f}"
                )

                print(
                    f"Total GMACs/sec     : "
                    f"{total_gmacs_s:.4f}"
                )

                visual_percentage = (
                    100.0
                    * vf["gmacs_per_second"]
                    / total_gmacs_s
                )
                
                print(
                    f"Visual frontend     : "
                    f"{visual_percentage:.2f}% "
                    f"of compute"
                )


if __name__ == "__main__":
    main()