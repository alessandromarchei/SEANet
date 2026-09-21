#!/usr/bin/env python3

import re
import argparse
import matplotlib.pyplot as plt


def parse_log(log_path):
    train_epochs = []
    train_loss = []
    train_lr = []

    val_epochs = []
    val_sisdr = []
    val_sdr = []
    val_sisdri = []
    val_sdri = []

    train_re = re.compile(
        r"Train:\s*\[\s*(\d+)\].*?"
        r"Lr:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),\s*"
        r"Loss:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    )

    val_re = re.compile(
        r"Val:\s*\[\s*(\d+)\].*?"
        r"SISDR:\s*([+-]?\d*\.?\d+),\s*"
        r"SDR:\s*([+-]?\d*\.?\d+),\s*"
        r"SISDRi:\s*([+-]?\d*\.?\d+),\s*"
        r"SDRi:\s*([+-]?\d*\.?\d+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            train_match = train_re.search(line)
            if train_match:
                train_epochs.append(int(train_match.group(1)))
                train_lr.append(float(train_match.group(2)))
                train_loss.append(float(train_match.group(3)))
                continue

            val_match = val_re.search(line)
            if val_match:
                val_epochs.append(int(val_match.group(1)))
                val_sisdr.append(float(val_match.group(2)))
                val_sdr.append(float(val_match.group(3)))
                val_sisdri.append(float(val_match.group(4)))
                val_sdri.append(float(val_match.group(5)))

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "train_lr": train_lr,
        "val_epochs": val_epochs,
        "val_sisdr": val_sisdr,
        "val_sdr": val_sdr,
        "val_sisdri": val_sisdri,
        "val_sdri": val_sdri,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log",
        help="SEANet training log .txt"
    )
    parser.add_argument(
        "--output",
        default="training_curves.png",
        help="Output plot"
    )
    args = parser.parse_args()

    data = parse_log(args.log)

    if not data["train_epochs"]:
        raise RuntimeError("No training entries found")

    if not data["val_epochs"]:
        raise RuntimeError("No validation entries found")

    # ---------------------------------------------------------
    # Print information
    # ---------------------------------------------------------

    print(f"Train epochs found: {len(data['train_epochs'])}")
    print(f"Validation points:   {len(data['val_epochs'])}")
    print(f"Last train epoch:    {data['train_epochs'][-1]}")

    best_idx = max(
        range(len(data["val_sisdr"])),
        key=lambda i: data["val_sisdr"][i]
    )

    best_epoch = data["val_epochs"][best_idx]
    best_sisdr = data["val_sisdr"][best_idx]

    print()
    print("Best validation:")
    print(f"  Epoch:   {best_epoch}")
    print(f"  SI-SDR:  {data['val_sisdr'][best_idx]:.3f} dB")
    print(f"  SDR:     {data['val_sdr'][best_idx]:.3f} dB")
    print(f"  SI-SDRi: {data['val_sisdri'][best_idx]:.3f} dB")
    print(f"  SDRi:    {data['val_sdri'][best_idx]:.3f} dB")

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    # Training loss - cold color
    ax1.plot(
        data["train_epochs"],
        data["train_loss"],
        color="#1565C0",
        linewidth=2.0,
        label="Train Loss"
    )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12, color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")

    ax1.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.3
    )

    # ---------------------------------------------------------
    # Validation metrics - warm colors
    # ---------------------------------------------------------

    ax2 = ax1.twinx()

    # SI-SDR: primary validation metric
    ax2.plot(
        data["val_epochs"],
        data["val_sisdr"],
        color="#D32F2F",
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        label="Val SI-SDR"
    )

    # SI-SDR improvement
    ax2.plot(
        data["val_epochs"],
        data["val_sisdri"],
        color="#F57C00",
        marker="s",
        markersize=4.0,
        linewidth=1.7,
        linestyle="--",
        label="Val SI-SDRi"
    )

    # SDR
    ax2.plot(
        data["val_epochs"],
        data["val_sdr"],
        color="#F9A825",
        marker="^",
        markersize=4.0,
        linewidth=1.7,
        linestyle="-.",
        label="Val SDR"
    )

    ax2.set_ylabel(
        "Validation Metric [dB]",
        fontsize=12,
        color="#B71C1C"
    )

    ax2.tick_params(
        axis="y",
        labelcolor="#B71C1C"
    )

    # ---------------------------------------------------------
    # Mark best SI-SDR
    # ---------------------------------------------------------

    ax2.scatter(
        [best_epoch],
        [best_sisdr],
        color="#8E0000",
        edgecolor="black",
        linewidth=0.7,
        s=85,
        zorder=10,
        label=f"Best SI-SDR: {best_sisdr:.2f} dB @ epoch {best_epoch}"
    )

    ax2.axvline(
        best_epoch,
        color="#8E0000",
        linestyle=":",
        linewidth=1.0,
        alpha=0.55
    )

    # ---------------------------------------------------------
    # Combined legend
    # ---------------------------------------------------------

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center right",
        frameon=True,
        fontsize=10
    )

    # ---------------------------------------------------------
    # Final formatting
    # ---------------------------------------------------------

    ax1.set_xlim(
        min(data["train_epochs"]),
        max(data["train_epochs"])
    )

    plt.title(
        "SEANet Training Convergence",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    plt.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved plot to: {args.output}")


if __name__ == "__main__":
    main()