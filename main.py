#!/usr/bin/env python3

import argparse
import os

import torch

from tools import *
from trainer import *
from dataLoader import *
from logger import WandbLogger


# ============================================================
# Experiment directory
# ============================================================

def create_experiment_directory(exp_name, root="runs"):
    """
    Create:
        runs/<exp_name>

    If it already exists:
        runs/<exp_name>_1
        runs/<exp_name>_2
        ...
    """

    os.makedirs(root, exist_ok=True)

    base_name = exp_name
    candidate_name = base_name
    candidate_path = os.path.join(
        root,
        candidate_name,
    )

    idx = 1

    while os.path.exists(candidate_path):

        candidate_name = (
            f"{base_name}_{idx}"
        )

        candidate_path = os.path.join(
            root,
            candidate_name,
        )

        idx += 1

    os.makedirs(
        candidate_path,
        exist_ok=False,
    )

    return candidate_name, candidate_path


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Audio-visual target speaker extraction."
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

parser.add_argument(
    "--batch_size",
    type=int,
    default=15,
    help="Micro-batch size",
)

parser.add_argument(
    "--grad_accum_steps",
    type=int,
    default=1,
    help="Number of micro-batches accumulated before optimizer step",
)

parser.add_argument(
    "--max_epoch",
    type=int,
    default=150,
    help="Maximum number of epochs",
)

parser.add_argument(
    "--n_cpu",
    type=int,
    default=12,
    help="Number of DataLoader workers",
)

parser.add_argument(
    "--val_step",
    type=int,
    default=2,
    help="Validate and save every N epochs",
)

parser.add_argument(
    "--length",
    type=float,
    default=4,
    help="Training data length",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0010,
    help="Initial learning rate",
)

parser.add_argument(
    "--lr_decay",
    type=float,
    default=0.97,
    help="Learning rate decay every val_step epochs",
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.10,
    help="Weight for auxiliary loss",
)


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

parser.add_argument(
    "--exp_name",
    type=str,
    default="",
    help="Experiment name for a new training run",
)

parser.add_argument(
    "--init_model",
    type=str,
    default="",
    help="Initialize model from checkpoint",
)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

parser.add_argument(
    "--data_list",
    type=str,
    default="",
)

parser.add_argument(
    "--visual_path",
    type=str,
    default="",
)

parser.add_argument(
    "--audio_path",
    type=str,
    default="",
)

parser.add_argument(
    "--musan_path",
    type=str,
    default="",
)

parser.add_argument(
    "--backbone",
    type=str,
    default="",
)

parser.add_argument(
    "--eval",
    dest="eval",
    action="store_true",
    help="Evaluation only",
)


# ============================================================
# Performance
# ============================================================

parser.add_argument(
    "--precision",
    choices=[
        "fp32",
        "fp16",
        "bf16",
    ],
    default="bf16",
    help="Training precision",
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
    help="Enable TF32",
)


# ============================================================
# W&B
# ============================================================

parser.add_argument(
    "--wandb_project",
    type=str,
    default="SEANNet",
)

parser.add_argument(
    "--no_wandb",
    action="store_true",
)

parser.add_argument(
    "--wandb_audio_samples",
    type=int,
    default=10,
)

parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Resume training from a full checkpoint, e.g. runs/exp/last.pt",
)

parser.add_argument(
    "--start_epoch",
    type=int,
    default=1,
    help="Starting epoch when initializing from a model-only checkpoint",
)

# ============================================================
# Parse arguments
# ============================================================

args = parser.parse_args()

if not args.resume and not args.exp_name:
    parser.error(
        "--exp_name is required when starting a new experiment"
    )
    
# ============================================================
# Experiment name / path
# ============================================================

if args.resume:

    # Resume existing experiment.
    resume_path = os.path.abspath(args.resume)

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {resume_path}"
        )

    # last.pt is inside the experiment directory.
    args.save_path = os.path.dirname(resume_path)

    args.exp_name = os.path.basename(
        os.path.normpath(args.save_path)
    )

    args.wandb_name = args.exp_name

    print()
    print("Resuming experiment")
    print("-------------------")
    print(f"Experiment: {args.exp_name}")
    print(f"Directory:  {args.save_path}")
    print(f"Checkpoint: {resume_path}")
    print()

else:

    # Brand-new experiment.
    actual_exp_name, save_path = create_experiment_directory(
        args.exp_name,
        root="runs",
    )

    args.save_path = save_path
    args.exp_name = actual_exp_name
    args.wandb_name = actual_exp_name


print()
print("Experiment")
print("----------")
print(f"Name:       {args.exp_name}")
print(f"Directory:  {args.save_path}")
print(f"W&B name:   {args.wandb_name}")
print()


# ============================================================
# Original SEANet initialization
# ============================================================

args = init_system(args)

s = init_trainer(args)


resume_info = None

if args.resume:

    resume_info = s.load_training_checkpoint(
        args.resume
    )

    args.epoch = resume_info["next_epoch"]


# ============================================================
# Logger
# ============================================================

resume_wandb_id = None

if resume_info is not None:
    resume_wandb_id = resume_info.get(
        "wandb_run_id"
    )

logger = WandbLogger(
    args,
    resume_run_id=resume_wandb_id,
)

s.logger = logger


# ============================================================
# Data
#
# IMPORTANT:
# Initialize ONCE.
# Do not recreate DataLoaders every epoch.
# ============================================================

args = init_loader(args)


# ============================================================
# Evaluation only
# ============================================================

if args.eval:

    s.eval_network(
        "Test",
        args,
    )

    logger.finish()

    raise SystemExit(0)


# ============================================================
# Training
# ============================================================
# ============================================================
# Training
# ============================================================

while args.epoch <= args.max_epoch:

    # --------------------------------------------------------
    # Train one complete epoch
    # --------------------------------------------------------

    s.train_network(args)


    # --------------------------------------------------------
    # Validation + model-only checkpoint
    #
    # Every val_step epochs:
    #
    #   model_0002.model
    #   model_0004.model
    #   model_0006.model
    #   ...
    #
    # These contain model weights only.
    # --------------------------------------------------------

    if args.epoch % args.val_step == 0:

        s.eval_network(
            "Val",
            args,
        )

        model_path = os.path.join(
            args.model_save_path,
            f"model_{args.epoch:04d}.model",
        )

        s.save_parameters(
            model_path
        )


    # --------------------------------------------------------
    # Full resume checkpoint
    #
    # Saved EVERY epoch and always overwrites:
    #
    #   runs/<experiment>/last.pt
    #
    # Contains:
    #   - model
    #   - optimizer
    #   - scheduler
    #   - GradScaler
    #   - epoch
    #   - W&B run ID
    #
    # This is the file used by --resume.
    # --------------------------------------------------------

    last_checkpoint_path = os.path.join(
        args.save_path,
        "last.pt",
    )

    s.save_training_checkpoint(
        path=last_checkpoint_path,
        epoch=args.epoch,
        wandb_run_id=logger.run_id,
    )


    # --------------------------------------------------------
    # Next epoch
    # --------------------------------------------------------

    args.epoch += 1


logger.finish()