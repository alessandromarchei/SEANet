import random
from pathlib import Path

import torch
import wandb
import numpy as np

class WandbLogger:
    def __init__(
        self,
        args,
        resume_run_id=None,
    ):  
        self.args = args
        self.enabled = not getattr(args, "no_wandb", False)

        if not self.enabled:
            return

        config = {
            "backbone": args.backbone,
            "batch_size": args.batch_size,
            "max_epoch": args.max_epoch,
            "length": args.length,
            "lr": args.lr,
            "lr_decay": args.lr_decay,
            "alpha": args.alpha,
            "val_step": args.val_step,
            "n_cpu": args.n_cpu,
            "precision": args.precision,
            "torch_compile": args.compile,
            "compile_mode": args.compile_mode,
            "tf32": args.tf32,
        }

        if resume_run_id is not None:

            wandb.init(
                project=args.wandb_project,
                name=args.exp_name,
                id=resume_run_id,
                resume="must",
                config=config,
                dir=args.save_path,
            )

        else:

            wandb.init(
                project=args.wandb_project,
                name=args.exp_name,
                config=config,
                dir=args.save_path,
            )

        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("system/*", step_metric="epoch")

    def watch_model(self, model):
        if not self.enabled:
            return

        # Logging gradients every iteration is expensive.
        # Keep this sparse.
        wandb.watch(
            model,
            log="gradients",
            log_freq=500,
            log_graph=False,
        )

    def log_train(
        self,
        epoch,
        loss,
        lr,
        epoch_time,
        samples_per_sec=None,
        batches_per_sec=None,
    ):
        if not self.enabled:
            return

        data = {
            "epoch": epoch,
            "train/loss": loss,
            "train/lr": lr,
            "train/epoch_time_sec": epoch_time,
        }

        if samples_per_sec is not None:
            data["train/samples_per_sec"] = samples_per_sec

        if batches_per_sec is not None:
            data["train/batches_per_sec"] = batches_per_sec

        if torch.cuda.is_available():
            data.update({
                "system/gpu_memory_allocated_GB":
                    torch.cuda.memory_allocated() / 1024**3,

                "system/gpu_memory_reserved_GB":
                    torch.cuda.memory_reserved() / 1024**3,

                "system/gpu_max_memory_allocated_GB":
                    torch.cuda.max_memory_allocated() / 1024**3,
            })

        wandb.log(data)

    def log_validation(
        self,
        epoch,
        sisdr,
        sdr,
        sisdri,
        sdri,
        val_time=None,
    ):
        if not self.enabled:
            return

        data = {
            "epoch": epoch,
            "val/SI-SDR": sisdr,
            "val/SDR": sdr,
            "val/SI-SDRi": sisdri,
            "val/SDRi": sdri,
        }

        if val_time is not None:
            data["val/time_sec"] = val_time

        wandb.log(data)

    def log_audio_examples(
        self,
        epoch,
        examples,
        sample_rate=16000,
    ):
        """
        examples:
        [
            {
                "mixture": Tensor,
                "estimate": Tensor,
                "target": Tensor,
                "name": str
            },
            ...
        ]

        The model estimate is peak-normalized ONLY for listening
        in W&B. The original tensor is not modified.
        """

        if not self.enabled:
            return

        table = wandb.Table(
            columns=[
                "sample",
                "mixture",
                "estimate",
                "ground_truth",
            ]
        )

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                x = (
                    x.detach()
                    .float()
                    .cpu()
                    .squeeze()
                    .numpy()
                )

            return x

        def normalize_for_listening(x, peak_target=0.95):
            """
            Peak-normalize audio for listening.

            Equivalent to the normalization used by save_wav(...):
                audio = 0.95 * audio / peak

            This does NOT affect training or evaluation metrics.
            """
            x = to_numpy(x)

            if not np.all(np.isfinite(x)):
                raise ValueError(
                    "Non-finite values found in audio sent to W&B"
                )

            peak = np.max(np.abs(x))

            if peak > 0:
                x = peak_target * x / peak

            return x

        for i, example in enumerate(examples):

            name = example.get(
                "name",
                f"sample_{i:02d}",
            )

            mixture = to_numpy(
                example["mixture"]
            )

            # Normalize model output ONLY for listening
            estimate = normalize_for_listening(
                example["estimate"]
            )

            target = to_numpy(
                example["target"]
            )

            table.add_data(
                name,

                wandb.Audio(
                    mixture,
                    sample_rate=sample_rate,
                    caption=f"{name} - mixture",
                ),

                wandb.Audio(
                    estimate,
                    sample_rate=sample_rate,
                    caption=f"{name} - SEANet",
                ),

                wandb.Audio(
                    target,
                    sample_rate=sample_rate,
                    caption=f"{name} - GT",
                ),
            )

        wandb.log({
            "epoch": epoch,
            "val/audio_examples": table,
        })

    def finish(self):
        if self.enabled:
            wandb.finish()

    @property
    def run_id(self):

        if not self.enabled:
            return None

        return wandb.run.id