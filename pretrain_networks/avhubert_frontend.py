import sys
from pathlib import Path

import torch
import torch.nn as nn


class AVHubertVisualFrontend(nn.Module):
    """
    Frozen AV-HuBERT visual-only feature extractor.

    Input:
        x: [B, 1, T, H, W]

    Output:
        features: [T, B, D]

    For AV-HuBERT Large:
        D = 1024
    """

    def __init__(
        self,
        checkpoint,
        avhubert_root,
        device="cuda",
        output_layer=None,
    ):
        super().__init__()

        self.checkpoint = Path(checkpoint).resolve()
        self.avhubert_root = Path(avhubert_root).resolve()
        self.device = torch.device(device)
        self.output_layer = output_layer

        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"AV-HuBERT checkpoint not found: "
                f"{self.checkpoint}"
            )

        if not self.avhubert_root.is_dir():
            raise FileNotFoundError(
                f"AV-HuBERT repository not found: "
                f"{self.avhubert_root}"
            )

        # Needed because AV-HuBERT registers custom Fairseq modules.
        avhubert_code = self.avhubert_root / "avhubert"

        if str(avhubert_code) not in sys.path:
            sys.path.insert(0, str(avhubert_code))

        import fairseq
        import hubert
        import hubert_pretraining

        print(
            f"Loading AV-HuBERT checkpoint:\n"
            f"  {self.checkpoint}"
        )

        models, cfg, task = (
            fairseq.checkpoint_utils
            .load_model_ensemble_and_task(
                [str(self.checkpoint)]
            )
        )

        if len(models) != 1:
            raise RuntimeError(
                f"Expected one AV-HuBERT model, "
                f"got {len(models)}"
            )

        self.model = models[0]

        self.model.eval()
        self.model.to(self.device)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.cfg = cfg
        self.task = task

        print("AV-HuBERT loaded successfully.")

    @torch.inference_mode()
    def forward(self, x):

        if x.ndim != 5:
            raise RuntimeError(
                f"Expected [B,1,T,H,W], "
                f"got {tuple(x.shape)}"
            )

        x = x.to(
            self.device,
            non_blocking=True,
        )

        # AV-HuBERT visual-only inference
        features, _ = self.model.extract_finetune(
            source={
                "video": x,
                "audio": None,
            },
            padding_mask=None,
            output_layer=self.output_layer,
        )

        # Typical extract_finetune output:
        # [B,T,D]
        if features.ndim != 3:
            raise RuntimeError(
                f"Unexpected AV-HuBERT output: "
                f"{tuple(features.shape)}"
            )

        # Standardise interface used by your cache generator:
        #
        # [B,T,D] -> [T,B,D]

        features = (
            features
            .transpose(0, 1)
            .contiguous()
        )

        return features