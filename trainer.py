import os
import sys
import time
import random
from contextlib import nullcontext
from collections import OrderedDict

import torch
import torch.nn as nn

from tools import *
from loss import *

from model.dprnn import dprnn
from model.muse import muse
from model.avsep import avsep
from model.seanet import seanet
from tqdm.auto import tqdm

import numpy as np
import soundfile as sf


# ============================================================
# Trainer initialization
# ============================================================

def init_trainer(args):

    s = trainer(args)

    args.epoch = args.start_epoch

    if args.init_model != "":

        print(
            f"Model {args.init_model} loaded from pretrain!"
        )

        s.load_parameters(
            args.init_model
        )

    elif len(args.modelfiles) >= 1:

        print(
            f"Model {args.modelfiles[-1]} "
            f"loaded from previous state!"
        )

        args.epoch = (
            int(
                os.path.splitext(
                    os.path.basename(
                        args.modelfiles[-1]
                    )
                )[0][6:]
            )
            + 1
        )

        s.load_parameters(
            args.modelfiles[-1]
        )

    return s

def save_wav(
    path,
    tensor,
    normalize=False,
    subtype="FLOAT",
):
    audio = (
        tensor
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    audio = np.squeeze(audio)

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"Non-finite values found while saving {path}"
        )

    if normalize:
        peak = np.max(np.abs(audio))

        if peak > 0:
            audio = 0.95 * audio / peak

    sf.write(
        path,
        audio,
        16000,
        subtype=subtype,
    )

# ============================================================
# Trainer
# ============================================================

class trainer(nn.Module):

    def __init__(self, args):

        super(trainer, self).__init__()

        # ====================================================
        # Performance configuration
        # ====================================================

        self.precision = getattr(
            args,
            "precision",
            "fp32",
        )

        self.use_compile = getattr(
            args,
            "compile",
            False,
        )

        self.compile_mode = getattr(
            args,
            "compile_mode",
            "default",
        )

        self.use_tf32 = getattr(
            args,
            "tf32",
            True,
        )

        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = self.use_tf32
            torch.backends.cudnn.allow_tf32 = self.use_tf32

            if self.use_tf32:

                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            # Good when input shapes are mostly constant.
            torch.backends.cudnn.benchmark = True

        # ====================================================
        # AMP
        # ====================================================

        if self.precision == "bf16":

            self.amp_dtype = torch.bfloat16
            self.use_amp = torch.cuda.is_available()

        elif self.precision == "fp16":

            self.amp_dtype = torch.float16
            self.use_amp = torch.cuda.is_available()

        elif self.precision == "fp32":

            self.amp_dtype = None
            self.use_amp = False

        else:

            raise ValueError(
                f"Unsupported precision: {self.precision}. "
                f"Use fp32, fp16 or bf16."
            )

        # GradScaler is needed for FP16.
        # BF16 normally does not require it.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                torch.cuda.is_available()
                and self.precision == "fp16"
            ),
        )

        # ====================================================
        # Model
        # ====================================================

        if args.backbone == "seanet":

            self.model = seanet(
                256,
                40,
                64,
                128,
                100,
                6,
            ).cuda()

        elif args.backbone == "avsep":

            self.model = avsep().cuda()

        elif args.backbone == "muse":

            self.model = muse(
                M=800
            ).cuda()

        elif args.backbone == "dprnn":

            self.model = dprnn().cuda()

        else:

            raise ValueError(
                f"Unknown backbone: {args.backbone}"
            )

        # ====================================================
        # Losses
        # ====================================================

        self.loss_se = loss_speech().cuda()

        self.speaker_loss = (
            nn.CrossEntropyLoss().cuda()
        )

        # ====================================================
        # Optimizer
        # ====================================================

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
        )

        self.scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size=args.val_step,
                gamma=args.lr_decay,
            )
        )

        # ====================================================
        # torch.compile
        # ====================================================

        if self.use_compile:

            if not hasattr(torch, "compile"):

                raise RuntimeError(
                    "torch.compile requested but "
                    "not available in this PyTorch version."
                )

            print(
                f"Compiling model with torch.compile "
                f"(mode={self.compile_mode})..."
            )

            self.model = torch.compile(
                self.model,
                mode=self.compile_mode,
            )

        # ====================================================
        # Logger
        # ====================================================

        # main.py will assign:
        #
        #   s.logger = logger

        self.logger = None

        # ====================================================
        # Fixed validation audio examples
        # ====================================================

        self.val_audio_indices = None

        self.num_val_audio_samples = getattr(
            args,
            "wandb_audio_samples",
            10,
        )

        self.val_audio_seed = 42

        # ====================================================
        # Information
        # ====================================================

        n_params = sum(
            p.numel()
            for p in self._original_model().parameters()
        )

        grad_accum_steps = getattr(
            args,
            "grad_accum_steps",
            1,
        )

        effective_batch = (
            args.batch_size
            * grad_accum_steps
        )

        print(
            "Model para number = %.2f"
            % (n_params / 1e6)
        )

        print()
        print("Training configuration")
        print("----------------------")
        print(f"Precision:              {self.precision}")
        print(f"AMP:                    {self.use_amp}")
        print(f"GradScaler:             {self.scaler.is_enabled()}")
        print(f"TF32:                   {self.use_tf32}")
        print(f"torch.compile:          {self.use_compile}")

        if self.use_compile:
            print(f"Compile mode:           {self.compile_mode}")

        print(f"Micro batch size:       {args.batch_size}")
        print(f"Gradient accumulation:  {grad_accum_steps}")
        print(f"Effective batch size:   {effective_batch}")
        print()


    # ========================================================
    # Helpers
    # ========================================================

    def _original_model(self):

        if hasattr(self.model, "_orig_mod"):
            return self.model._orig_mod

        return self.model


    def _autocast_context(self):

        if not self.use_amp:
            return nullcontext()

        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
        )


    def _to_cuda(self, tensor):

        return tensor.cuda(
            non_blocking=True
        )


    def _initialize_validation_audio_indices(
        self,
        Loader,
    ):

        if self.val_audio_indices is not None:
            return

        if self.num_val_audio_samples <= 0:

            self.val_audio_indices = set()
            return

        dataset_size = len(
            Loader.dataset
        )

        n_samples = min(
            self.num_val_audio_samples,
            dataset_size,
        )

        rng = random.Random(
            self.val_audio_seed
        )

        indices = rng.sample(
            range(dataset_size),
            n_samples,
        )

        self.val_audio_indices = set(
            indices
        )

        print(
            "Fixed validation audio indices:",
            sorted(self.val_audio_indices),
        )


    # ========================================================
    # Training
    # ========================================================

    def train_network(self, args):

        # ====================================================
        # Gradient accumulation configuration
        # ====================================================

        accum_steps = getattr(
            args,
            "grad_accum_steps",
            1,
        )

        if accum_steps < 1:

            raise ValueError(
                "--grad_accum_steps must be >= 1"
            )

        num_loader_batches = len(
            args.trainLoader
        )

        # Number of micro-batches in the final accumulation
        # group if it is incomplete.
        remainder = (
            num_loader_batches
            % accum_steps
        )

        # ====================================================
        # Statistics
        # ====================================================

        time_start = time.perf_counter()

        total_loss = 0.0
        num_samples = 0
        num_batches = 0
        num_optimizer_steps = 0

        self.train()

        # ====================================================
        # Scheduler
        # ====================================================

        self.scheduler.step(
            args.epoch - 1
        )

        lr = self.optim.param_groups[0]["lr"]

        # ====================================================
        # GPU stats
        # ====================================================

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ====================================================
        # IMPORTANT:
        #
        # Zero gradients ONCE before accumulation starts.
        #
        # Do NOT zero gradients at every micro-batch.
        # ====================================================

        self.optim.zero_grad(
            set_to_none=True
        )

        # ====================================================
        # Training loop
        # ====================================================

        train_pbar = tqdm(
            enumerate(
                args.trainLoader,
                start=1,
            ),
            total=num_loader_batches,
            desc=f"Train {args.epoch:03d}/{args.max_epoch:03d}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        for num, (
            audio,
            face,
            speech,
            noise,
            muse_label,
        ) in train_pbar:

            # =================================================
            # Move data to GPU
            # =================================================

            audio = self._to_cuda(audio)
            face = self._to_cuda(face)
            speech = self._to_cuda(speech)
            noise = self._to_cuda(noise)

            if not torch.is_tensor(muse_label):

                muse_label = torch.as_tensor(
                    muse_label,
                    dtype=torch.long,
                )

            muse_label = muse_label.to(
                device="cuda",
                dtype=torch.long,
                non_blocking=True,
            )

            current_B = audio.shape[0]

            # =================================================
            # Determine accumulation group size
            #
            # Usually:
            #
            #   current_accum_steps = accum_steps
            #
            # But the final group can contain fewer batches.
            #
            # Example:
            #
            #   102 batches
            #   accumulation = 8
            #
            # final group = 6
            #
            # We divide those final losses by 6, not by 8.
            # =================================================

            if (
                remainder != 0
                and num > (
                    num_loader_batches
                    - remainder
                )
            ):

                current_accum_steps = remainder

            else:

                current_accum_steps = accum_steps

            # =================================================
            # Forward + loss
            # =================================================

            with self._autocast_context():

                # ---------------------------------------------
                # SEANet
                # ---------------------------------------------

                if args.backbone == "seanet":

                    out_s, out_n = self.model(
                        audio,
                        face,
                        M=current_B,
                    )

                    loss_s_main = (
                        self.loss_se.forward(
                            out_s[-current_B:, :],
                            speech,
                        )
                    )

                    loss_n_main = (
                        self.loss_se.forward(
                            out_n[-current_B:, :],
                            noise,
                        )
                    )

                    loss_n_rest = (
                        self.loss_se.forward(
                            out_n[:-current_B, :],
                            noise.repeat(5, 1),
                        )
                    )

                    loss_s_rest = (
                        self.loss_se.forward(
                            out_s[:-current_B, :],
                            speech.repeat(5, 1),
                        )
                    )

                    loss = (
                        loss_s_main
                        + (
                            loss_n_main
                            + loss_n_rest
                            + loss_s_rest
                        )
                        * args.alpha
                    )

                # ---------------------------------------------
                # DPRNN
                # ---------------------------------------------

                elif args.backbone == "dprnn":

                    out_s = self.model(
                        audio,
                        face,
                        M=current_B,
                    )

                    loss_s_main = (
                        self.loss_se.forward(
                            out_s[-current_B:, :],
                            speech,
                        )
                    )

                    loss_s_rest = (
                        self.loss_se.forward(
                            out_s[:-current_B, :],
                            speech.repeat(5, 1),
                        )
                    )

                    loss = (
                        loss_s_main
                        + loss_s_rest
                        * args.alpha
                    )

                # ---------------------------------------------
                # MuSE
                # ---------------------------------------------

                elif args.backbone == "muse":

                    out_e, out_s = self.model(
                        audio,
                        face,
                    )

                    loss_s_main = (
                        self.loss_se.forward(
                            out_s,
                            speech,
                        )
                    )

                    loss_muse = 0.0

                    for i in range(4):

                        loss_muse = (
                            loss_muse
                            + self.speaker_loss(
                                out_e[i],
                                muse_label,
                            )
                        )

                    loss = (
                        loss_s_main
                        + loss_muse * 0.1
                    )

                # ---------------------------------------------
                # AVSep
                # ---------------------------------------------

                elif args.backbone == "avsep":

                    out_s = self.model(
                        audio,
                        face,
                    )

                    loss = (
                        self.loss_se.forward(
                            out_s,
                            speech,
                        )
                    )

            # =================================================
            # Logging uses ORIGINAL loss.
            #
            # Do not log loss/current_accum_steps because that
            # would make the reported loss depend on gradient
            # accumulation.
            # =================================================

            raw_loss = (
                loss
                .detach()
                .float()
                .item()
            )

            total_loss += raw_loss

            # =================================================
            # Normalize loss for gradient accumulation
            # =================================================

            loss_for_backward = (
                loss
                / current_accum_steps
            )

            # =================================================
            # Backward
            #
            # Backward happens for EVERY micro-batch.
            # Gradients accumulate in model parameters.
            # =================================================

            if self.scaler.is_enabled():

                # FP16
                self.scaler.scale(
                    loss_for_backward
                ).backward()

            else:

                # BF16 / FP32
                loss_for_backward.backward()

            # =================================================
            # Optimizer step?
            #
            # Only every accum_steps micro-batches, OR at the
            # end of the epoch for the incomplete final group.
            # =================================================

            should_step = (
                num % accum_steps == 0
                or num == num_loader_batches
            )

            if should_step:

                # ---------------------------------------------
                # FP16
                # ---------------------------------------------

                if self.scaler.is_enabled():

                    # Gradients are currently scaled.
                    # Unscale BEFORE gradient clipping.
                    self.scaler.unscale_(
                        self.optim
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=5.0,
                    )

                    self.scaler.step(
                        self.optim
                    )

                    self.scaler.update()

                # ---------------------------------------------
                # BF16 / FP32
                # ---------------------------------------------

                else:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=5.0,
                    )

                    self.optim.step()

                # ---------------------------------------------
                # Clear gradients AFTER optimizer step.
                # ---------------------------------------------

                self.optim.zero_grad(
                    set_to_none=True
                )

                num_optimizer_steps += 1

            # =================================================
            # Statistics
            # =================================================

            num_samples += current_B
            num_batches += 1

            time_used = (
                time.perf_counter()
                - time_start
            )

            mean_loss = (
                total_loss
                / num_batches
            )

            progress = (
                100.0
                * num
                / num_loader_batches
            )

            estimated_minutes = (
                time_used
                * num_loader_batches
                / num
                / 60.0
            )

            effective_batch = (
                args.batch_size
                * accum_steps
            )

            train_pbar.set_postfix(
                loss=f"{mean_loss:.3f}",
                lr=f"{lr:.2e}",
                eff_batch=effective_batch,
                opt_steps=num_optimizer_steps,
            )

        # ====================================================
        # End epoch
        # ====================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = (
            time.perf_counter()
            - time_start
        )

        mean_loss = (
            total_loss
            / max(num_batches, 1)
        )

        samples_per_sec = (
            num_samples
            / max(epoch_time, 1e-8)
        )

        batches_per_sec = (
            num_batches
            / max(epoch_time, 1e-8)
        )

        optimizer_steps_per_sec = (
            num_optimizer_steps
            / max(epoch_time, 1e-8)
        )

        effective_batch = (
            args.batch_size
            * accum_steps
        )

        sys.stdout.write("\n")

        log_line = (
            "Train: [%2d] %.2f%% "
            "(%.1f mins) "
            "Lr: %.6f, "
            "Loss: %.3f, "
            "Samples/s: %.2f, "
            "MicroBatch: %d, "
            "Accum: %d, "
            "EffectiveBatch: %d, "
            "OptimizerSteps: %d"
            % (
                args.epoch,
                100.0,
                epoch_time / 60.0,
                lr,
                mean_loss,
                samples_per_sec,
                args.batch_size,
                accum_steps,
                effective_batch,
                num_optimizer_steps,
            )
        )

        args.score_file.write(
            log_line + "\n"
        )

        args.score_file.flush()

        # ====================================================
        # W&B
        # ====================================================

        if self.logger is not None:

            self.logger.log_train(
                epoch=args.epoch,
                loss=mean_loss,
                lr=lr,
                epoch_time=epoch_time,
                samples_per_sec=samples_per_sec,
                batches_per_sec=batches_per_sec,
            )

        return


    # ========================================================
    # Validation / test
    # ========================================================

    def eval_network(
        self,
        eval_type,
        args,
    ):

        Loader = (
            args.valLoader
            if eval_type == "Val"
            else args.testLoader
        )

        self.eval()

        time_start = time.perf_counter()

        sisdr_speech = 0.0
        sdr_speech = 0.0
        sisdri_speech = 0.0
        sdri_speech = 0.0

        num_batches = 0

        # ====================================================
        # Audio examples
        # ====================================================

        collect_audio = (
            eval_type == "Val"
            and self.num_val_audio_samples > 0
        )

        if collect_audio:

            self._initialize_validation_audio_indices(
                Loader
            )

        audio_examples = []

        dataset_offset = 0

        # ====================================================
        # Evaluation
        # ====================================================
        with torch.inference_mode():

            eval_pbar = tqdm(
                enumerate(
                    Loader,
                    start=1,
                ),
                total=len(Loader),
                desc=f"{eval_type:5s} {args.epoch:03d}/{args.max_epoch:03d}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            )

            for num, (
                audio,
                face,
                speech,
                noise,
                _,
            ) in eval_pbar:


                current_B = audio.shape[0]

                batch_indices = range(
                    dataset_offset,
                    dataset_offset
                    + current_B,
                )

                dataset_offset += current_B

                # =============================================
                # GPU
                # =============================================

                audio = self._to_cuda(audio)
                face = self._to_cuda(face)
                speech = self._to_cuda(speech)

                # =============================================
                # Forward
                # =============================================

                with self._autocast_context():

                    if args.backbone == "seanet":

                        out_speech, _ = self.model(
                            audio,
                            face,
                            current_B,
                        )

                        out = out_speech[
                            -current_B:,
                            :
                        ]

                    elif args.backbone == "dprnn":

                        out_speech = self.model(
                            audio,
                            face,
                            current_B,
                        )

                        out = out_speech[
                            -current_B:,
                            :
                        ]

                    elif args.backbone == "muse":

                        _, out_speech = self.model(
                            audio,
                            face,
                        )

                        out = out_speech

                    elif args.backbone == "avsep":

                        out_speech = self.model(
                            audio,
                            face,
                        )

                        out = out_speech

                # =============================================
                # Metrics in FP32
                # =============================================

                out_fp32 = out.float()
                speech_fp32 = speech.float()
                audio_fp32 = audio.float()

                res_speech = (
                    self.loss_se.forward_eval_light(
                        out_fp32,
                        speech_fp32,
                    )
                )

                res_orig = (
                    self.loss_se.forward_eval_light(
                        audio_fp32,
                        speech_fp32,
                    )
                )

                sisdr_value = (
                    res_speech["sisdr"]
                    .detach()
                    .float()
                    .item()
                )

                sdr_value = (
                    res_speech["sdr"]
                    .detach()
                    .float()
                    .item()
                )

                orig_sisdr = (
                    res_orig["sisdr"]
                    .detach()
                    .float()
                    .item()
                )

                orig_sdr = (
                    res_orig["sdr"]
                    .detach()
                    .float()
                    .item()
                )

                sisdr_speech += sisdr_value
                sdr_speech += sdr_value

                sisdri_speech += (
                    sisdr_value
                    - orig_sisdr
                )

                sdri_speech += (
                    sdr_value
                    - orig_sdr
                )

                num_batches += 1

                # =============================================
                # Fixed W&B audio examples
                # =============================================

                if collect_audio:

                    for (
                        local_idx,
                        dataset_idx,
                    ) in enumerate(
                        batch_indices
                    ):

                        if (
                            dataset_idx
                            in self.val_audio_indices
                        ):

                            audio_examples.append({
                                "name":
                                    f"val_{dataset_idx:06d}",

                                "mixture":
                                    audio[
                                        local_idx
                                    ]
                                    .detach()
                                    .float()
                                    .cpu(),

                                "estimate":
                                    out_fp32[
                                        local_idx
                                    ]
                                    .detach()
                                    .cpu(),

                                "target":
                                    speech_fp32[
                                        local_idx
                                    ]
                                    .detach()
                                    .cpu(),
                            })

                # =============================================
                # Console
                # =============================================

                time_used = (
                    time.perf_counter()
                    - time_start
                )

                mean_sisdr = (
                    sisdr_speech
                    / num_batches
                )

                mean_sdr = (
                    sdr_speech
                    / num_batches
                )

                mean_sisdri = (
                    sisdri_speech
                    / num_batches
                )

                mean_sdri = (
                    sdri_speech
                    / num_batches
                )

                progress = (
                    100.0
                    * num
                    / len(Loader)
                )

                estimated_minutes = (
                    time_used
                    * len(Loader)
                    / num
                    / 60.0
                )

                eval_pbar.set_postfix(
                    sisdr=f"{mean_sisdr:.3f}",
                    sdr=f"{mean_sdr:.3f}",
                    sisdri=f"{mean_sisdri:.3f}",
                )
        # ====================================================
        # End evaluation
        # ====================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        eval_time = (
            time.perf_counter()
            - time_start
        )

        mean_sisdr = (
            sisdr_speech
            / max(num_batches, 1)
        )

        mean_sdr = (
            sdr_speech
            / max(num_batches, 1)
        )

        mean_sisdri = (
            sisdri_speech
            / max(num_batches, 1)
        )

        mean_sdri = (
            sdri_speech
            / max(num_batches, 1)
        )

        sys.stdout.write("\n")

        log_line = (
            "%s: [%2d] %.2f%% "
            "(%.1f mins), "
            "SISDR: %.3f, "
            "SDR: %.3f, "
            "SISDRi: %.3f, "
            "SDRi: %.3f"
            % (
                eval_type,
                args.epoch,
                100.0,
                eval_time / 60.0,
                mean_sisdr,
                mean_sdr,
                mean_sisdri,
                mean_sdri,
            )
        )

        args.score_file.write(
            log_line + "\n"
        )

        args.score_file.flush()

        # ====================================================
        # W&B
        # ====================================================

        if (
            eval_type == "Val"
            and self.logger is not None
        ):

            self.logger.log_validation(
                epoch=args.epoch,
                sisdr=mean_sisdr,
                sdr=mean_sdr,
                sisdri=mean_sisdri,
                sdri=mean_sdri,
                val_time=eval_time,
            )








            if len(audio_examples) > 0:

                audio_examples = sorted(
                    audio_examples,
                    key=lambda x: x["name"],
                )

                # ========================================================
                # W&B audio
                # ========================================================

                self.logger.log_audio_examples(
                    epoch=args.epoch,
                    examples=audio_examples,
                    sample_rate=16000,
                )

                # ========================================================
                # Save validation audio locally
                # ========================================================

                audio_dir = os.path.join(
                    args.save_path,
                    "validation_audio",
                    f"epoch_{args.epoch:03d}",
                )

                os.makedirs(
                    audio_dir,
                    exist_ok=True,
                )

                for example in audio_examples:

                    name = example["name"]

                    # ----------------------------------------------------
                    # Mixture
                    # Already normalized by the dataset loader.
                    # ----------------------------------------------------

                    save_wav(
                        os.path.join(
                            audio_dir,
                            f"{name}_mixture.wav",
                        ),
                        example["mixture"],
                        normalize=True,
                        subtype="PCM_16",
                    )

                    # ----------------------------------------------------
                    # Ground-truth target
                    # ----------------------------------------------------

                    save_wav(
                        os.path.join(
                            audio_dir,
                            f"{name}_target.wav",
                        ),
                        example["target"],
                        normalize=True,
                        subtype="PCM_16",
                    )

                    # ----------------------------------------------------
                    # Raw model output
                    #
                    # IMPORTANT:
                    # FLOAT preserves the actual network output.
                    # No clipping to [-1, 1].
                    # ----------------------------------------------------

                    save_wav(
                        os.path.join(
                            audio_dir,
                            f"{name}_estimate_float.wav",
                        ),
                        example["estimate"],
                        normalize=False,
                        subtype="FLOAT",
                    )

                    # ----------------------------------------------------
                    # Listening version
                    #
                    # Peak-normalized to 0.95 before writing PCM16.
                    # This is the one you should listen to.
                    # ----------------------------------------------------

                    save_wav(
                        os.path.join(
                            audio_dir,
                            f"{name}_estimate_listen.wav",
                        ),
                        example["estimate"],
                        normalize=True,
                        subtype="PCM_16",
                    )

                print(
                    f"Validation audio saved to: {audio_dir}"
                )
        
        
        return {
            "sisdr": mean_sisdr,
            "sdr": mean_sdr,
            "sisdri": mean_sisdri,
            "sdri": mean_sdri,
            "val_time": eval_time,
        }


    # ========================================================
    # Save
    # ========================================================

    def save_parameters(
        self,
        path,
    ):

        original_model = (
            self._original_model()
        )

        state = OrderedDict()

        for (
            name,
            param,
        ) in original_model.state_dict().items():

            state[
                "model." + name
            ] = (
                param
                .detach()
                .cpu()
            )

        torch.save(
            state,
            path,
        )


    # ========================================================
    # Load
    # ========================================================

    def load_parameters(
        self,
        path,
    ):

        print(
            f"Loading checkpoint: {path}"
        )

        loaded_state = torch.load(
            path,
            map_location="cpu",
        )

        model = self._original_model()

        model_state = (
            model.state_dict()
        )

        cleaned_state = OrderedDict()

        for (
            name,
            param,
        ) in loaded_state.items():

            original_name = name

            # =================================================
            # Remove prefixes from different checkpoint styles
            # =================================================

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

            # =================================================
            # Parameter exists?
            # =================================================

            if name not in model_state:

                print(
                    f"{original_name} "
                    f"is not in the model."
                )

                continue

            # =================================================
            # Shape matches?
            # =================================================

            if (
                model_state[name].size()
                != param.size()
            ):

                sys.stderr.write(
                    "Wrong parameter length: "
                    f"{original_name}, "
                    f"model: "
                    f"{model_state[name].size()}, "
                    f"loaded: "
                    f"{param.size()}\n"
                )

                continue

            cleaned_state[name] = param

        # ====================================================
        # Load
        # ====================================================

        missing, unexpected = (
            model.load_state_dict(
                cleaned_state,
                strict=False,
            )
        )

        if missing:

            print(
                f"Missing parameters: "
                f"{len(missing)}"
            )

            for name in missing[:20]:
                print(f"  {name}")

        if unexpected:

            print(
                f"Unexpected parameters: "
                f"{len(unexpected)}"
            )

            for name in unexpected[:20]:
                print(f"  {name}")

        print(
            f"Loaded "
            f"{len(cleaned_state)} tensors."
        )

    def save_training_checkpoint(
        self,
        path,
        epoch,
        wandb_run_id=None,
    ):

        original_model = self._original_model()

        checkpoint = {
            "epoch": epoch,

            "model": original_model.state_dict(),

            "optimizer": self.optim.state_dict(),

            "scheduler": self.scheduler.state_dict(),

            "scaler": (
                self.scaler.state_dict()
                if self.scaler is not None
                else None
            ),

            "wandb_run_id": wandb_run_id,

            # Useful metadata / sanity checks.
            "precision": self.precision,
            "compile": self.use_compile,
            "compile_mode": self.compile_mode,
        }

        # Save atomically:
        # first temporary file, then rename.
        #
        # This prevents a crash during torch.save() from destroying
        # the previous valid last.pt.
        tmp_path = path + ".tmp"

        torch.save(
            checkpoint,
            tmp_path,
        )

        os.replace(
            tmp_path,
            path,
        )

        print(
            f"Training checkpoint saved: {path}"
        )
        

    def save_rotating_checkpoint(
        self,
        checkpoint_dir,
        checkpoint_type,
        epoch,
        wandb_run_id=None,
    ):
        """
        Save exactly one checkpoint of a given type.

        Examples:
            best_epoch_012.pt
            last_epoch_013.pt

        Any previous checkpoint of the same type is removed
        after the new checkpoint has been successfully written.
        """

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        new_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_type}_epoch_{epoch:03d}.pt",
        )

        # --------------------------------------------------------
        # Save new checkpoint first.
        #
        # save_training_checkpoint() is already atomic.
        # --------------------------------------------------------

        self.save_training_checkpoint(
            path=new_path,
            epoch=epoch,
            wandb_run_id=wandb_run_id,
        )

        # --------------------------------------------------------
        # Only after successful save, remove previous versions.
        # --------------------------------------------------------

        prefix = f"{checkpoint_type}_epoch_"

        for filename in os.listdir(checkpoint_dir):

            if (
                filename.startswith(prefix)
                and filename.endswith(".pt")
                and filename != os.path.basename(new_path)
            ):

                old_path = os.path.join(
                    checkpoint_dir,
                    filename,
                )

                os.remove(old_path)

                print(
                    f"Removed old checkpoint: {old_path}"
                )

        return new_path


    def load_training_checkpoint(
        self,
        path,
    ):

        print(
            f"Loading full training checkpoint: {path}"
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        # ========================================================
        # Model
        # ========================================================

        original_model = self._original_model()

        original_model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        # ========================================================
        # Optimizer
        # ========================================================

        self.optim.load_state_dict(
            checkpoint["optimizer"]
        )

        # Optimizer tensors were loaded on CPU because of
        # map_location="cpu". Move them back to the same device
        # as the model parameters.
        device = next(
            original_model.parameters()
        ).device

        for state in self.optim.state.values():

            for key, value in state.items():

                if torch.is_tensor(value):

                    state[key] = value.to(
                        device,
                        non_blocking=True,
                    )

        # ========================================================
        # Scheduler
        # ========================================================

        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        # ========================================================
        # GradScaler
        # ========================================================

        if (
            checkpoint.get("scaler") is not None
            and self.scaler is not None
        ):

            self.scaler.load_state_dict(
                checkpoint["scaler"]
            )

        # ========================================================
        # Epoch
        # ========================================================

        completed_epoch = checkpoint["epoch"]

        next_epoch = completed_epoch + 1

        print()
        print("Resume successful")
        print("-----------------")
        print(f"Completed epoch: {completed_epoch}")
        print(f"Next epoch:      {next_epoch}")
        print(
            f"LR:              "
            f"{self.optim.param_groups[0]['lr']:.8f}"
        )
        print()

        return {
            "epoch": completed_epoch,
            "next_epoch": next_epoch,
            "wandb_run_id": checkpoint.get(
                "wandb_run_id"
            ),
        }