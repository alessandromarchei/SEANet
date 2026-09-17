#!/usr/bin/env python3

import argparse
import os

import torch

from tools import *
from trainer import *
from dataLoader import *
from logger import WandbLogger


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Audio-visual target speaker extraction."
)

parser.add_argument('--batch_size', type=int,   default=15,       help='Batch size for training and validation')
parser.add_argument('--max_epoch',  type=int,   default=150,      help='Maximum number of epochs')
parser.add_argument('--n_cpu',      type=int,   default=12,       help='Number of loader threads')
parser.add_argument('--val_step',   type=int,   default=3,        help='Every [val_step] epochs: Validation, update learning rate and save model')
parser.add_argument('--length',     type=float, default=4,        help='Training data length')
parser.add_argument('--lr',         type=float, default=0.0010,   help='Init learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,     help='Learning rate decay every [val_step] epochs')
parser.add_argument("--alpha",      type=float, default=0.10,     help='Weight for the loss_auxil')
parser.add_argument('--init_model', type=str,   default="",       help='Init model from pretrain')
parser.add_argument('--save_path',  type=str,   default="",       help='Path to save the clean list')
parser.add_argument('--data_list',  type=str,   default="",       help='The path of the training list')
parser.add_argument('--visual_path',type=str,   default="",       help='The path of the lip embs')
parser.add_argument('--audio_path', type=str,   default="",       help='The path of the clean audio')
parser.add_argument('--musan_path', type=str,   default="",       help='The path for the musan dataset for augmentation, can ignore if do not use')
parser.add_argument('--backbone',   type=str,    default="")
parser.add_argument('--eval',       dest='eval', action='store_true', help='Do evaluation only')

parser.add_argument( "--grad_accum_steps", type=int, default=1, help="Number of micro-batches to accumulate before optimizer step",)


# ============================================================
# Performance options
# ============================================================

parser.add_argument("--precision",choices=["fp32", "fp16", "bf16"],default="bf16",help="Training precision",)
parser.add_argument( "--compile", action="store_true", help="Enable torch.compile",)
parser.add_argument( "--compile_mode", choices=[ "default", "reduce-overhead", "max-autotune", ], default="default",)
parser.add_argument( "--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32",)

# ============================================================
# W&B
# ============================================================

parser.add_argument( "--wandb_project", type=str, default="SEANNet")
parser.add_argument( "--wandb_name", type=str, default="",)
parser.add_argument( "--no_wandb", action="store_true",)
parser.add_argument( "--wandb_audio_samples", type=int, default=10,)

args = init_system(parser.parse_args())
s = init_trainer(args)

# ============================================================
# Logger
# ============================================================

logger = WandbLogger(args)

s.logger = logger


# ============================================================
# Data
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

    quit()


# ============================================================
# Training
# ============================================================

while args.epoch < args.max_epoch:

    args = init_loader(args)

    s.train_network(args)

    if args.epoch % args.val_step == 0:

        model_path = (
            args.model_save_path
            + "/model_%04d.model"
            % args.epoch
        )

        s.save_parameters(model_path)

        s.eval_network(
            "Val",
            args,
        )

    args.epoch += 1


logger.finish()