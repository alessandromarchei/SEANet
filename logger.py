import random
from pathlib import Path

import torch
import wandb


class WandbLogger:
    def __init__(self, args):
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

        for i, example in enumerate(examples):
            name = example.get("name", f"sample_{i:02d}")

            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    x = x.detach().float().cpu().squeeze().numpy()
                return x

            table.add_data(
                name,

                wandb.Audio(
                    to_numpy(example["mixture"]),
                    sample_rate=sample_rate,
                    caption=f"{name} - mixture",
                ),

                wandb.Audio(
                    to_numpy(example["estimate"]),
                    sample_rate=sample_rate,
                    caption=f"{name} - SEANet",
                ),

                wandb.Audio(
                    to_numpy(example["target"]),
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