#!/usr/bin/env python3

import argparse
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from model.seanet import seanet
from pretrain_networks.visual_frontend import VisualFrontend


# ============================================================
# Wrappers
# ============================================================

class SEANetWrapper(nn.Module):
    """
    SEANet without visual frontend.

    Inputs:
        mixture : [B, T_audio]
        visual  : [B, T_visual, 512]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mixture, visual):

        batch_size = mixture.shape[0]

        speech, noise = self.model(
            mixture,
            visual,
            M=batch_size,
        )

        return speech, noise


def inspect_onnx_metadata(path, max_nodes=50):
    import onnx

    model = onnx.load(
        path,
        load_external_data=False,
    )

    print()
    print("=" * 100)
    print("ONNX NODE METADATA")
    print("=" * 100)

    for i, node in enumerate(model.graph.node[:max_nodes]):

        print()
        print(
            f"[{i:04d}] "
            f"{node.op_type:<25} "
            f"name={node.name}"
        )

        if not node.metadata_props:
            print("    metadata: NONE")
            continue

        for prop in node.metadata_props:
            print(
                f"    {prop.key}: "
                f"{prop.value}"
            )


class FullSEANetWrapper(nn.Module):
    """
    Complete pipeline:

        raw video
            |
            v
        VisualFrontend
            |
            v
        visual embeddings
            |
            v
          SEANet

    External video convention:

        [B, T, 1, H, W]

    Original VisualFrontend convention:

        [T, B, 1, H, W]

    Original VisualFrontend output:

        [T, B, 512]

    SEANet visual input:

        [B, T, 512]
    """

    def __init__(
        self,
        visual_frontend,
        seanet_model,
    ):
        super().__init__()

        self.visual_frontend = visual_frontend
        self.seanet = seanet_model

    def forward(self, mixture, video):

        # [B,T,1,H,W]
        # ->
        # [T,B,1,H,W]

        video = video.permute(
            1, 0, 2, 3, 4
        )

        # [T,B,1,H,W]
        # ->
        # [T,B,512]

        visual = self.visual_frontend(
            video
        )

        # [T,B,512]
        # ->
        # [B,T,512]

        visual = visual.permute(
            1, 0, 2
        )

        batch_size = mixture.shape[0]

        speech, noise = self.seanet(
            mixture,
            visual,
            M=batch_size,
        )

        return speech, noise


# ============================================================
# Checkpoint
# ============================================================

def extract_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in (
        "model",
        "state_dict",
        "model_state_dict",
    ):
        if (
            key in checkpoint
            and isinstance(checkpoint[key], dict)
        ):
            return checkpoint[key]

    return checkpoint


def clean_state_dict(state_dict):

    cleaned = OrderedDict()

    prefixes = (
        "model._orig_mod.",
        "_orig_mod.",
        "model.",
        "module.",
    )

    for name, tensor in state_dict.items():

        new_name = name

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_name.startswith(prefix):

                    new_name = new_name[
                        len(prefix):
                    ]

                    changed = True
                    break

        cleaned[new_name] = tensor

    return cleaned


def load_weights(model, path, name):

    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    print(f"Checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Loaded tensors:        {len(state_dict)}"
    )
    print(
        f"Missing parameters:    {len(missing)}"
    )
    print(
        f"Unexpected parameters: {len(unexpected)}"
    )

    if missing:

        print("\nMissing:")

        for x in missing[:30]:
            print(f"  {x}")

    if unexpected:

        print("\nUnexpected:")

        for x in unexpected[:30]:
            print(f"  {x}")

    if missing or unexpected:

        raise RuntimeError(
            f"{name}: checkpoint/model mismatch."
        )

    print("Checkpoint matches model.")


# ============================================================
# Export
# ============================================================


def simplify_onnx_model(
    input_path,
    output_path=None,
):

    try:
        import onnx
        import onnxsim

    except ImportError as exc:
        raise RuntimeError(
            "ONNX simplification requested but "
            "onnx/onnxsim is not installed.\n"
            "Install with:\n"
            "  pip install onnx onnxsim"
        ) from exc

    if output_path is None:

        base, ext = os.path.splitext(
            input_path
        )

        output_path = (
            base + "_simplified" + ext
        )

    print()
    print("=" * 80)
    print("ONNX SIMPLIFICATION")
    print("=" * 80)

    print(
        f"Input:  {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    original_size = (
        os.path.getsize(input_path)
        / 1024**2
    )

    print(
        f"Original size: "
        f"{original_size:.2f} MiB"
    )

    print(
        "Loading ONNX model...",
        flush=True,
    )

    model = onnx.load(
        input_path,
        load_external_data=True,
    )

    original_nodes = len(
        model.graph.node
    )

    original_initializers = len(
        model.graph.initializer
    )

    print(
        f"Original nodes:        "
        f"{original_nodes}"
    )

    print(
        f"Original initializers: "
        f"{original_initializers}"
    )

    print(
        "Running ONNX Simplifier...",
        flush=True,
    )

    t0 = time.perf_counter()

    simplified_model, check = (
        onnxsim.simplify(
            model,
        )
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    if not check:

        raise RuntimeError(
            "ONNX Simplifier validation failed."
        )

    print(
        f"Simplification completed "
        f"in {elapsed:.2f} s"
    )

    # Validate again with ONNX itself.

    onnx.checker.check_model(
        simplified_model
    )

    print(
        "Simplified ONNX checker: OK"
    )

    # Save everything in ONE .onnx file.
    #
    # No external-data file.

    onnx.save_model(
        simplified_model,
        output_path,
        save_as_external_data=False,
    )

    simplified_nodes = len(
        simplified_model.graph.node
    )

    simplified_initializers = len(
        simplified_model.graph.initializer
    )

    simplified_size = (
        os.path.getsize(output_path)
        / 1024**2
    )

    print()
    print("-" * 80)

    print(
        f"Nodes:        "
        f"{original_nodes} -> "
        f"{simplified_nodes}"
    )

    print(
        f"Initializers: "
        f"{original_initializers} -> "
        f"{simplified_initializers}"
    )

    print(
        f"Size:         "
        f"{original_size:.2f} -> "
        f"{simplified_size:.2f} MiB"
    )

    if original_nodes > 0:

        reduction = (
            100.0
            * (
                original_nodes
                - simplified_nodes
            )
            / original_nodes
        )

        print(
            f"Node reduction: "
            f"{reduction:.2f}%"
        )

    print("-" * 80)

    print()
    print(
        f"Simplified ONNX: "
        f"{output_path}"
    )

    return output_path


def export(args):

    print()
    print("=" * 80)
    print("SEANET ONNX EXPORT")
    print("=" * 80)

    # ========================================================
    # SEANet
    # ========================================================

    model = seanet(
        256,
        40,
        64,
        128,
        100,
        6,
    )

    load_weights(
        model,
        args.checkpoint,
        "SEANet",
    )

    model.eval()

    # ========================================================
    # Input sizes
    # ========================================================

    audio_samples = int(
        args.duration * 16000
    )

    visual_frames = int(
        args.duration * args.fps
    )

    mixture = torch.randn(
        1,
        audio_samples,
        dtype=torch.float32,
    )

    # ========================================================
    # Full model
    # ========================================================

    if args.visual_frontend:

        print()
        print("=" * 80)
        print("MODE")
        print("=" * 80)

        print(
            "Exporting COMPLETE model:"
        )
        print(
            "Video -> VisualFrontend -> SEANet -> Audio"
        )

        frontend = VisualFrontend()

        load_weights(
            frontend,
            args.visual_frontend,
            "Visual Frontend",
        )

        frontend.eval()

        wrapper = FullSEANetWrapper(
            frontend,
            model,
        )

        video = torch.randn(
            1,
            visual_frames,
            1,
            args.height,
            args.width,
            dtype=torch.float32,
        )

        inputs = (
            mixture,
            video,
        )

        input_names = [
            "mixture",
            "video",
        ]

        print()
        print(
            f"Audio input: {tuple(mixture.shape)}"
        )
        print(
            f"Video input: {tuple(video.shape)}"
        )

    # ========================================================
    # Backend only
    # ========================================================

    else:

        print()
        print("=" * 80)
        print("MODE")
        print("=" * 80)

        print(
            "No visual frontend supplied."
        )
        print(
            "Exporting SEANet backend only."
        )

        wrapper = SEANetWrapper(
            model
        )

        visual = torch.randn(
            1,
            visual_frames,
            512,
            dtype=torch.float32,
        )

        inputs = (
            mixture,
            visual,
        )

        input_names = [
            "mixture",
            "visual_embeddings",
        ]

        print()
        print(
            f"Audio input:  {tuple(mixture.shape)}"
        )
        print(
            f"Visual input: {tuple(visual.shape)}"
        )

    wrapper.eval()

    # ========================================================
    # Sanity forward
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 80)
    print("PYTORCH FORWARD")
    print("=" * 80)

    print(f"Device: {device}")
    print("Running forward...", flush=True)

    wrapper = wrapper.to(device)

    device_inputs = tuple(
        x.to(device)
        for x in inputs
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.inference_mode():

        speech, noise = wrapper(
            *device_inputs
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - t0
    )

    print(
        f"Forward completed in "
        f"{elapsed:.3f} s"
    )

    print(
        f"Speech output: "
        f"{tuple(speech.shape)}"
    )

    print(
        f"Noise output:  "
        f"{tuple(noise.shape)}"
    )

    # ========================================================
    # Move back to CPU for export
    # ========================================================

    print()
    print(
        "Moving model back to CPU...",
        flush=True,
    )

    wrapper = wrapper.cpu()

    inputs = tuple(
        x.cpu()
        for x in inputs
    )

    # Release GPU tensors

    del device_inputs
    del speech
    del noise

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # ONNX export
    # ========================================================

    output_dir = os.path.dirname(
        os.path.abspath(
            args.output
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("ONNX EXPORT")
    print("=" * 80)

    print(
        f"Output: {args.output}"
    )
    print(
        f"Opset:  {args.opset}"
    )

    print(
        "Exporting graph...",
        flush=True,
    )

    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        inputs,
        args.output,

        input_names=input_names,

        output_names=[
            "estimated_speech",
            "estimated_noise",
        ],

        opset_version=args.opset,

        dynamo=True,

        external_data=False,
    )
    inspect_onnx_metadata(
        args.output,
        max_nodes=50,
    )
    elapsed = (
        time.perf_counter()
        - t0
    )

    print(
        f"Export completed in "
        f"{elapsed:.2f} s"
    )

    # ========================================================
    # Validation
    # ========================================================

    try:

        import onnx

        print()
        print("=" * 80)
        print("ONNX CHECK")
        print("=" * 80)

        onnx_model = onnx.load(
            args.output
        )

        onnx.checker.check_model(
            onnx_model
        )

        print("ONNX checker: OK")

        print(
            f"Nodes: "
            f"{len(onnx_model.graph.node)}"
        )

        print(
            f"Initializers: "
            f"{len(onnx_model.graph.initializer)}"
        )

    except ImportError:

        print(
            "onnx not installed; "
            "validation skipped."
        )

    size_mb = (
        os.path.getsize(args.output)
        / 1024**2
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )

    print()
    print(
        f"ONNX: {args.output}"
    )

    if args.simplify:

        simplified_path = args.output

        simplify_onnx_model(
            input_path=args.output,
            output_path=simplified_path,
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--visual-frontend",
        default=None,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--simplify",
        action="store_true",
        help=(
            "Run ONNX Simplifier after export and save "
            "a *_simplified.onnx model."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not os.path.isfile(
        args.checkpoint
    ):
        raise FileNotFoundError(
            args.checkpoint
        )

    if (
        args.visual_frontend
        and not os.path.isfile(
            args.visual_frontend
        )
    ):
        raise FileNotFoundError(
            args.visual_frontend
        )

    export(args)


if __name__ == "__main__":
    main()