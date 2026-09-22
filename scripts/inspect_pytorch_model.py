#!/usr/bin/env python3

"""
inspect_pytorch_model.py

Professional PyTorch model/checkpoint inspector.

Supports:
    - .pt
    - .pth
    - .model
    - arbitrary torch.save() checkpoints

Can inspect:
    1. Serialized nn.Module
    2. Checkpoint dictionaries containing:
         model
         module
         net
         network
         state_dict
         model_state_dict
         weights
         ...
    3. Raw state_dict / OrderedDict

When an nn.Module is available:
    - complete module hierarchy
    - leaf-layer table
    - input/output tensor shapes
    - parameter count
    - trainable parameter count
    - parameter memory
    - buffers
    - kernel / stride / padding / dilation / groups
    - approximate MACs per layer
    - approximate FLOPs
    - activation memory
    - totals

When only a state_dict is available:
    - parameter names
    - tensor shapes
    - dtype
    - number of elements
    - memory
    - inferred layer grouping
    - inferred Conv/Linear information when possible

Dependencies:
    pip install torch rich

Example:
    python inspect_pytorch_model.py model.pth

    python inspect_pytorch_model.py model.pt \
        --input-shape 1 3 224 224

    python inspect_pytorch_model.py checkpoint.pth \
        --input-shape 1 1 16000 \
        --device cuda

    python inspect_pytorch_model.py model.pth \
        --input-shape 1 3 88 88 \
        --depth 4
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


console = Console()


# =============================================================================
# Formatting utilities
# =============================================================================

def human_number(n: int) -> str:
    n = int(n)

    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f} B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f} M"
    if n >= 1_000:
        return f"{n / 1_000:.3f} K"

    return str(n)


def human_bytes(num_bytes: float) -> str:
    num_bytes = float(num_bytes)

    units = ["B", "KiB", "MiB", "GiB", "TiB"]

    for unit in units:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0

    return f"{num_bytes:.2f} PiB"


def human_macs(n: Optional[int]) -> str:
    if n is None:
        return "-"

    n = float(n)

    if n >= 1e12:
        return f"{n / 1e12:.3f} T"
    if n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"

    return f"{n:.0f}"


def shape_to_str(shape) -> str:
    if shape is None:
        return "-"

    if isinstance(shape, torch.Size):
        shape = tuple(shape)

    if isinstance(shape, tuple):
        return "×".join(str(x) for x in shape)

    if isinstance(shape, list):
        return "[" + ", ".join(shape_to_str(x) for x in shape) + "]"

    return str(shape)


def dtype_size(dtype: torch.dtype) -> int:
    try:
        return torch.empty((), dtype=dtype).element_size()
    except Exception:
        return 0


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def tuple_str(value) -> str:
    if value is None:
        return "-"

    if isinstance(value, tuple):
        return "×".join(map(str, value))

    return str(value)


# =============================================================================
# Checkpoint loading
# =============================================================================

STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "weights",
    "params",
    "model_weights",
    "network_state_dict",
    "net_state_dict",
)

MODEL_KEYS = (
    "model",
    "module",
    "net",
    "network",
)


def torch_load_compat(path: str, device: str):
    """
    PyTorch >= 2.6 defaults to weights_only=True.
    We explicitly try weights_only=False because this is a local inspection
    utility and checkpoints may contain nn.Module objects.

    WARNING:
        Never load untrusted torch pickle checkpoints.
    """

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Older PyTorch
        return torch.load(
            path,
            map_location=device,
        )


def looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, (dict, OrderedDict)):
        return False

    if not obj:
        return False

    tensor_count = sum(
        isinstance(v, torch.Tensor)
        for v in obj.values()
    )

    return tensor_count > 0 and tensor_count >= max(1, len(obj) // 2)


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str):
    if not state_dict:
        return state_dict

    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {
            k[len(prefix):]: v
            for k, v in state_dict.items()
        }

    return state_dict


def normalize_state_dict(state_dict):
    """
    Common cleanup for DataParallel / DDP checkpoints.
    """

    for prefix in (
        "module.",
        "_orig_mod.",
    ):
        state_dict = strip_prefix(state_dict, prefix)

    return state_dict


@dataclass
class LoadedCheckpoint:
    raw: Any
    model: Optional[nn.Module]
    state_dict: Optional[Dict[str, torch.Tensor]]
    source: str


def extract_checkpoint(obj: Any) -> LoadedCheckpoint:

    # -------------------------------------------------------------------------
    # Direct nn.Module
    # -------------------------------------------------------------------------

    if isinstance(obj, nn.Module):
        return LoadedCheckpoint(
            raw=obj,
            model=obj,
            state_dict=obj.state_dict(),
            source="Serialized nn.Module",
        )

    # -------------------------------------------------------------------------
    # Raw state_dict
    # -------------------------------------------------------------------------

    if looks_like_state_dict(obj):
        sd = normalize_state_dict(dict(obj))

        return LoadedCheckpoint(
            raw=obj,
            model=None,
            state_dict=sd,
            source="Raw state_dict",
        )

    # -------------------------------------------------------------------------
    # Dictionary checkpoint
    # -------------------------------------------------------------------------

    if isinstance(obj, dict):

        # Look for embedded nn.Module
        for key in MODEL_KEYS:
            if key in obj and isinstance(obj[key], nn.Module):
                model = obj[key]

                return LoadedCheckpoint(
                    raw=obj,
                    model=model,
                    state_dict=model.state_dict(),
                    source=f"Checkpoint['{key}'] -> nn.Module",
                )

        # Look for known state_dict keys
        for key in STATE_DICT_KEYS:
            if key in obj and looks_like_state_dict(obj[key]):
                sd = normalize_state_dict(dict(obj[key]))

                return LoadedCheckpoint(
                    raw=obj,
                    model=None,
                    state_dict=sd,
                    source=f"Checkpoint['{key}'] -> state_dict",
                )

        # Search one level deeper
        for key, value in obj.items():
            if isinstance(value, dict):

                for nested_key in STATE_DICT_KEYS:
                    if (
                        nested_key in value
                        and looks_like_state_dict(value[nested_key])
                    ):
                        sd = normalize_state_dict(
                            dict(value[nested_key])
                        )

                        return LoadedCheckpoint(
                            raw=obj,
                            model=None,
                            state_dict=sd,
                            source=(
                                f"Checkpoint['{key}']"
                                f"['{nested_key}'] -> state_dict"
                            ),
                        )

    return LoadedCheckpoint(
        raw=obj,
        model=None,
        state_dict=None,
        source="Unknown",
    )


# =============================================================================
# State-dict inspection
# =============================================================================

def infer_tensor_role(name: str, tensor: torch.Tensor) -> str:
    lname = name.lower()

    if lname.endswith("bias"):
        return "Bias"

    if lname.endswith("running_mean"):
        return "BN mean"

    if lname.endswith("running_var"):
        return "BN variance"

    if lname.endswith("num_batches_tracked"):
        return "BN counter"

    if lname.endswith("weight"):

        if tensor.ndim == 4:
            return "Conv2D weight"

        if tensor.ndim == 5:
            return "Conv3D weight"

        if tensor.ndim == 3:
            return "Conv1D / tensor"

        if tensor.ndim == 2:
            return "Linear / embedding"

        if tensor.ndim == 1:
            return "Norm / scale"

        return "Weight"

    return "Tensor"


def infer_kernel(tensor: torch.Tensor) -> str:

    if tensor.ndim == 3:
        return str(tuple(tensor.shape[2:]))

    if tensor.ndim == 4:
        return str(tuple(tensor.shape[2:]))

    if tensor.ndim == 5:
        return str(tuple(tensor.shape[2:]))

    return "-"


def print_checkpoint_metadata(obj: Any):

    if not isinstance(obj, dict):
        return

    table = Table(
        title="Checkpoint Metadata",
        box=box.ROUNDED,
        header_style="bold cyan",
    )

    table.add_column("Key", style="bold")
    table.add_column("Type")
    table.add_column("Value / Description")

    for key, value in obj.items():

        if isinstance(value, torch.Tensor):
            description = (
                f"Tensor {shape_to_str(value.shape)} "
                f"{value.dtype}"
            )

        elif isinstance(value, nn.Module):
            description = value.__class__.__name__

        elif isinstance(value, dict):
            description = f"dict ({len(value)} keys)"

        elif isinstance(value, (list, tuple)):
            description = f"{type(value).__name__} ({len(value)} elements)"

        elif isinstance(value, (str, int, float, bool, type(None))):
            description = repr(value)

            if len(description) > 100:
                description = description[:97] + "..."

        else:
            description = type(value).__name__

        table.add_row(
            str(key),
            type(value).__name__,
            description,
        )

    console.print(table)


def print_state_dict_summary(
    state_dict: Dict[str, torch.Tensor],
):

    table = Table(
        title="State Dictionary — Parameters & Buffers",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold magenta",
    )

    table.add_column("#", justify="right", style="dim")
    table.add_column("Tensor name", style="cyan", overflow="fold")
    table.add_column("Role", style="yellow")
    table.add_column("Shape", style="green")
    table.add_column("Kernel", justify="center")
    table.add_column("DType")
    table.add_column("Elements", justify="right")
    table.add_column("Memory", justify="right")

    total_elements = 0
    total_bytes = 0

    for idx, (name, tensor) in enumerate(state_dict.items()):

        if not isinstance(tensor, torch.Tensor):
            continue

        elements = tensor.numel()
        memory = tensor_bytes(tensor)

        total_elements += elements
        total_bytes += memory

        table.add_row(
            str(idx),
            name,
            infer_tensor_role(name, tensor),
            shape_to_str(tensor.shape),
            infer_kernel(tensor),
            str(tensor.dtype).replace("torch.", ""),
            f"{elements:,}",
            human_bytes(memory),
        )

    console.print(table)

    summary = Table(
        title="State Dictionary Summary",
        box=box.DOUBLE_EDGE,
        header_style="bold green",
    )

    summary.add_column("Metric")
    summary.add_column("Value", justify="right")

    summary.add_row(
        "Total tensors",
        str(len(state_dict)),
    )

    summary.add_row(
        "Total stored elements",
        f"{total_elements:,}",
    )

    summary.add_row(
        "Total stored elements",
        human_number(total_elements),
    )

    summary.add_row(
        "Tensor storage",
        human_bytes(total_bytes),
    )

    console.print(summary)


# =============================================================================
# Module hierarchy
# =============================================================================

def build_module_tree(model: nn.Module) -> Tree:

    root = Tree(
        f"[bold cyan]{model.__class__.__name__}[/bold cyan]"
    )

    def recurse(module: nn.Module, tree: Tree):

        for name, child in module.named_children():

            params = sum(
                p.numel()
                for p in child.parameters(recurse=False)
            )

            label = (
                f"[green]{name}[/green]: "
                f"[yellow]{child.__class__.__name__}[/yellow]"
            )

            if params:
                label += (
                    f"  [dim]({human_number(params)} params)[/dim]"
                )

            branch = tree.add(label)
            recurse(child, branch)

    recurse(model, root)

    return root


# =============================================================================
# Forward analysis
# =============================================================================

@dataclass
class LayerRecord:
    index: int
    name: str
    layer_type: str

    input_shape: Any
    output_shape: Any

    kernel: str
    stride: str
    padding: str
    dilation: str
    groups: str

    params: int
    trainable: int

    param_bytes: int
    output_bytes: int

    macs: Optional[int]


def extract_shape(obj):

    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)

    if isinstance(obj, (list, tuple)):
        return [
            extract_shape(x)
            for x in obj
        ]

    if isinstance(obj, dict):
        return {
            k: extract_shape(v)
            for k, v in obj.items()
        }

    return type(obj).__name__


def recursive_tensor_bytes(obj) -> int:

    if isinstance(obj, torch.Tensor):
        return tensor_bytes(obj)

    if isinstance(obj, (list, tuple)):
        return sum(
            recursive_tensor_bytes(x)
            for x in obj
        )

    if isinstance(obj, dict):
        return sum(
            recursive_tensor_bytes(x)
            for x in obj.values()
        )

    return 0


def first_tensor(obj) -> Optional[torch.Tensor]:

    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for x in obj:
            result = first_tensor(x)
            if result is not None:
                return result

    if isinstance(obj, dict):
        for x in obj.values():
            result = first_tensor(x)
            if result is not None:
                return result

    return None


def calculate_macs(
    module: nn.Module,
    inputs,
    output,
) -> Optional[int]:

    out = first_tensor(output)

    if out is None:
        return None

    # -------------------------------------------------------------------------
    # Conv1d / Conv2d / Conv3d
    #
    # MACs =
    # output elements *
    # (Cin/groups * kernel elements)
    # -------------------------------------------------------------------------

    if isinstance(
        module,
        (nn.Conv1d, nn.Conv2d, nn.Conv3d),
    ):

        kernel_ops = math.prod(module.kernel_size)

        kernel_ops *= (
            module.in_channels // module.groups
        )

        return int(
            out.numel() * kernel_ops
        )

    # -------------------------------------------------------------------------
    # Linear
    # -------------------------------------------------------------------------

    if isinstance(module, nn.Linear):

        output_vectors = (
            out.numel() // module.out_features
        )

        return int(
            output_vectors
            * module.in_features
            * module.out_features
        )

    # -------------------------------------------------------------------------
    # BatchNorm
    #
    # Not normally counted as MACs in NN complexity reports.
    # -------------------------------------------------------------------------

    return None


def module_attributes(module: nn.Module):

    kernel = getattr(module, "kernel_size", None)
    stride = getattr(module, "stride", None)
    padding = getattr(module, "padding", None)
    dilation = getattr(module, "dilation", None)
    groups = getattr(module, "groups", None)

    return (
        tuple_str(kernel),
        tuple_str(stride),
        tuple_str(padding),
        tuple_str(dilation),
        str(groups) if groups is not None else "-",
    )


def analyze_forward(
    model: nn.Module,
    dummy_input,
) -> Tuple[List[LayerRecord], Any]:

    records: List[LayerRecord] = []
    hooks = []

    module_names = {
        module: name
        for name, module in model.named_modules()
    }

    leaf_modules = [
        module
        for module in model.modules()
        if len(list(module.children())) == 0
    ]

    def hook_fn(module, inputs, output):

        name = module_names.get(module, "?")

        params = sum(
            p.numel()
            for p in module.parameters(recurse=False)
        )

        trainable = sum(
            p.numel()
            for p in module.parameters(recurse=False)
            if p.requires_grad
        )

        param_bytes = sum(
            tensor_bytes(p)
            for p in module.parameters(recurse=False)
        )

        kernel, stride, padding, dilation, groups = (
            module_attributes(module)
        )

        macs = calculate_macs(
            module,
            inputs,
            output,
        )

        records.append(
            LayerRecord(
                index=len(records),
                name=name,
                layer_type=module.__class__.__name__,
                input_shape=extract_shape(inputs),
                output_shape=extract_shape(output),
                kernel=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                params=params,
                trainable=trainable,
                param_bytes=param_bytes,
                output_bytes=recursive_tensor_bytes(output),
                macs=macs,
            )
        )

    for module in leaf_modules:
        hooks.append(
            module.register_forward_hook(hook_fn)
        )

    model.eval()

    try:
        with torch.inference_mode():

            if isinstance(dummy_input, tuple):
                output = model(*dummy_input)
            else:
                output = model(dummy_input)

    finally:
        for hook in hooks:
            hook.remove()

    return records, output


# =============================================================================
# Detailed model table
# =============================================================================

def print_layer_table(records: List[LayerRecord]):

    table = Table(
        title="Forward Graph — Leaf Layers",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )

    table.add_column("#", justify="right", style="dim")
    table.add_column("Layer", style="cyan", overflow="fold")
    table.add_column("Type", style="yellow")

    table.add_column("Input shape", style="green")
    table.add_column("Output shape", style="green")

    table.add_column("Kernel", justify="center")
    table.add_column("Stride", justify="center")
    table.add_column("Pad", justify="center")
    table.add_column("Groups", justify="right")

    table.add_column("Params", justify="right")
    table.add_column("MACs", justify="right", style="magenta")
    table.add_column("Output mem", justify="right")

    for r in records:

        table.add_row(
            str(r.index),
            r.name,
            r.layer_type,
            shape_to_str(r.input_shape),
            shape_to_str(r.output_shape),
            r.kernel,
            r.stride,
            r.padding,
            r.groups,
            human_number(r.params),
            human_macs(r.macs),
            human_bytes(r.output_bytes),
        )

    console.print(table)


# =============================================================================
# Model totals
# =============================================================================

def print_model_totals(
    model: nn.Module,
    records: Optional[List[LayerRecord]] = None,
):

    params = list(model.parameters())
    buffers = list(model.buffers())

    total_params = sum(
        p.numel()
        for p in params
    )

    trainable_params = sum(
        p.numel()
        for p in params
        if p.requires_grad
    )

    frozen_params = (
        total_params - trainable_params
    )

    parameter_bytes = sum(
        tensor_bytes(p)
        for p in params
    )

    buffer_elements = sum(
        b.numel()
        for b in buffers
    )

    buffer_bytes = sum(
        tensor_bytes(b)
        for b in buffers
    )

    total_macs = None
    activation_bytes = None

    if records is not None:

        known_macs = [
            r.macs
            for r in records
            if r.macs is not None
        ]

        total_macs = sum(known_macs)

        activation_bytes = sum(
            r.output_bytes
            for r in records
        )

    table = Table(
        title="Model Summary",
        box=box.DOUBLE_EDGE,
        header_style="bold green",
    )

    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row(
        "Total parameters",
        f"{total_params:,}  ({human_number(total_params)})",
    )

    table.add_row(
        "Trainable parameters",
        f"{trainable_params:,}  ({human_number(trainable_params)})",
    )

    table.add_row(
        "Frozen parameters",
        f"{frozen_params:,}  ({human_number(frozen_params)})",
    )

    table.add_row(
        "Parameter memory",
        human_bytes(parameter_bytes),
    )

    table.add_row(
        "Buffers",
        f"{buffer_elements:,}",
    )

    table.add_row(
        "Buffer memory",
        human_bytes(buffer_bytes),
    )

    table.add_row(
        "Model tensors memory",
        human_bytes(parameter_bytes + buffer_bytes),
    )

    if total_macs is not None:

        table.add_row(
            "Known MACs / forward",
            human_macs(total_macs),
        )

        table.add_row(
            "Approx. FLOPs / forward",
            human_macs(2 * total_macs),
        )

    if activation_bytes is not None:

        table.add_row(
            "Cumulative layer outputs",
            human_bytes(activation_bytes),
        )

    console.print(table)


# =============================================================================
# File information
# =============================================================================

def print_file_info(
    path: str,
    loaded: LoadedCheckpoint,
):

    file_size = os.path.getsize(path)

    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold blue",
    )

    table.add_column("Property")
    table.add_column("Value")

    table.add_row(
        "File",
        os.path.abspath(path),
    )

    table.add_row(
        "Checkpoint size",
        human_bytes(file_size),
    )

    table.add_row(
        "Detected format",
        loaded.source,
    )

    table.add_row(
        "PyTorch",
        torch.__version__,
    )

    table.add_row(
        "CUDA available",
        str(torch.cuda.is_available()),
    )

    if loaded.model is not None:

        table.add_row(
            "Model class",
            loaded.model.__class__.__name__,
        )

    console.print(
        Panel(
            table,
            title="[bold]PyTorch Model Inspector[/bold]",
            border_style="blue",
        )
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect PyTorch models and checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to .pt/.pth/.model checkpoint",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for loading / dummy forward",
    )

    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Dummy model input shape. "
            "Example: --input-shape 1 3 224 224"
        ),
    )

    
    parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Printmetadata also",
    )


    parser.add_argument(
        "--dtype",
        choices=[
            "float32",
            "float16",
            "bfloat16",
        ],
        default="float32",
        help="Dummy input dtype",
    )

    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Do not print module hierarchy",
    )

    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Print the regular PyTorch model representation",
    )

    return parser.parse_args()


def resolve_dtype(name: str):

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    return mapping[name]


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    console.print()

    if not os.path.isfile(args.checkpoint):

        console.print(
            f"[bold red]ERROR:[/bold red] "
            f"Checkpoint not found: {args.checkpoint}"
        )

        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():

        console.print(
            "[yellow]CUDA requested but unavailable. "
            "Falling back to CPU.[/yellow]"
        )

        args.device = "cpu"

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------

    with console.status(
        "[bold cyan]Loading checkpoint...[/bold cyan]"
    ):

        try:
            raw = torch_load_compat(
                args.checkpoint,
                args.device,
            )

        except Exception:

            console.print(
                "[bold red]Failed to load checkpoint.[/bold red]"
            )

            console.print_exception()

            sys.exit(1)

    loaded = extract_checkpoint(raw)

    print_file_info(
        args.checkpoint,
        loaded,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    if args.print_metadata :
        print_checkpoint_metadata(raw)

    # -------------------------------------------------------------------------
    # Could not identify anything
    # -------------------------------------------------------------------------

    if (
        loaded.model is None
        and loaded.state_dict is None
    ):

        console.print(
            Panel(
                "[red]Could not identify an nn.Module or state_dict "
                "inside this checkpoint.[/red]\n\n"
                f"Root object type: {type(raw).__name__}",
                title="Unsupported checkpoint structure",
            )
        )

        sys.exit(2)

    # -------------------------------------------------------------------------
    # state_dict is always useful
    # -------------------------------------------------------------------------

    if loaded.state_dict is not None:

        print_state_dict_summary(
            loaded.state_dict
        )

    # -------------------------------------------------------------------------
    # state_dict only
    # -------------------------------------------------------------------------

    if loaded.model is None:


        return

    # -------------------------------------------------------------------------
    # Full nn.Module inspection
    # -------------------------------------------------------------------------

    model = loaded.model
    model = model.to(args.device)
    model.eval()

    if not args.no_tree:

        console.print(
            Panel(
                build_module_tree(model),
                title="Module Hierarchy",
                border_style="cyan",
            )
        )

    if args.full_model:

        console.print(
            Panel(
                str(model),
                title="PyTorch Model Representation",
                border_style="magenta",
            )
        )

    # -------------------------------------------------------------------------
    # No dummy input
    # -------------------------------------------------------------------------

    if args.input_shape is None:

        print_model_totals(model)

        console.print(
            Panel(
                "[yellow]No --input-shape supplied.[/yellow]\n\n"
                "The architecture and parameters were inspected, but "
                "a forward pass was not executed.\n\n"
                "For activation shapes and MACs, run for example:\n\n"
                "[bold cyan]"
                f"python {os.path.basename(__file__)} "
                f"{args.checkpoint} "
                "--input-shape 1 3 224 224"
                "[/bold cyan]",
                title="Forward analysis skipped",
                border_style="yellow",
            )
        )

        return

    # -------------------------------------------------------------------------
    # Dummy forward
    # -------------------------------------------------------------------------

    dtype = resolve_dtype(args.dtype)

    dummy = torch.randn(
        *args.input_shape,
        device=args.device,
        dtype=dtype,
    )

    console.print(
        Panel(
            f"Input shape : [green]{shape_to_str(dummy.shape)}[/green]\n"
            f"DType       : [green]{dummy.dtype}[/green]\n"
            f"Device      : [green]{dummy.device}[/green]",
            title="Dummy Forward Input",
            border_style="green",
        )
    )

    try:

        with console.status(
            "[bold cyan]Running instrumented forward pass...[/bold cyan]"
        ):

            records, output = analyze_forward(
                model,
                dummy,
            )

    except Exception as exc:

        console.print(
            Panel(
                f"[red]Forward pass failed:[/red]\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "The checkpoint/model itself was loaded correctly. "
                "The most likely reason is that the model requires "
                "multiple inputs or an input shape different from the "
                "one supplied.",
                title="Forward Error",
                border_style="red",
            )
        )

        print_model_totals(model)

        if console.is_terminal:
            console.print(
                "[dim]Use --full-model to inspect the architecture "
                "and determine the required inputs.[/dim]"
            )

        return

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print_layer_table(records)

    print_model_totals(
        model,
        records,
    )

    console.print(
        Panel(
            f"[bold]Final output[/bold]\n\n"
            f"Shape: [green]{shape_to_str(extract_shape(output))}[/green]\n"
            f"Memory: [green]{human_bytes(recursive_tensor_bytes(output))}[/green]",
            border_style="green",
        )
    )

    console.print(
        Panel(
            "[dim]"
            "MACs are analytically estimated for Conv1d/2d/3d and "
            "Linear layers. Operations implemented through functional "
            "calls, custom CUDA kernels, attention internals, einsum, "
            "matmul, interpolation, normalization and other operators "
            "may not be included. FLOPs are reported using the common "
            "approximation FLOPs ≈ 2 × MACs."
            "[/dim]",
            title="Complexity Notes",
            border_style="dim",
        )
    )


if __name__ == "__main__":
    main()