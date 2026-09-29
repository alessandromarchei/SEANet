#!/usr/bin/env python3

"""
Detailed statistical analysis of visual speech embeddings.

Expected embedding layout:

    ROOT/
        id00001/
            utterance_001.npy   # [T, D]
            utterance_002.npy
        id00002/
            ...

Typical DeepAVSR/SEANet case:
    D = 512

Optional video layout:

    VIDEO_ROOT/
        id00001/
            utterance_001.mp4
            ...

Main analyses
--------------
1. Per-dimension global statistics
2. Within-utterance temporal variance
3. Within-speaker variance
4. Between-speaker variance
5. Speaker variance ratio / ICC-like score
6. Temporal dynamics
7. Dead / low-variance features
8. Feature correlation / redundancy
9. PCA dimensionality
10. Effective rank / participation ratio
11. Speaker centroid similarity
12. Utterance centroid similarity
13. Linear speaker decoding
14. Interactive frame + 512-D feature visualization
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
except ImportError:
    Console = None
    Table = None
    track = lambda x, **kwargs: x


# ============================================================
# Data structures
# ============================================================

@dataclass
class Utterance:
    speaker: str
    video_id: str
    clip_id: str
    name: str
    path: Path
    frames: int
    dim: int
# ============================================================
# Utility
# ============================================================

def console_print(*args):
    if Console is not None:
        Console().print(*args)
    else:
        print(*args)


def safe_mean(x, axis=None):
    return np.nanmean(x, axis=axis)


def safe_var(x, axis=None):
    return np.nanvar(x, axis=axis)


def cosine_similarity_matrix(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norm, eps)
    return Xn @ Xn.T


# ============================================================
# Dataset loading
# ============================================================

def discover_embeddings(root: Path):
    """
    Recursively discover all .npy embeddings.

    Expected VoxCeleb2-like structure:

        root/
            id00001/
                video_id_1/
                    utterance_1.npy
                    utterance_2.npy
                video_id_2/
                    ...
            id00002/
                ...

    The first directory below root is always interpreted
    as the speaker ID.
    """

    files = sorted(root.rglob("*.npy"))

    if not files:
        raise RuntimeError(
            f"No .npy files found recursively under {root}"
        )

    return files


def load_dataset(
    root: Path,
    max_speakers=None,
    max_utts_per_speaker=None,
    seed=0,
):
    rng = random.Random(seed)

    files = discover_embeddings(root)

    grouped = defaultdict(list)

    for path in files:
        relative = path.relative_to(root)

        if len(relative.parts) < 2:
            console_print(
                f"[yellow]Skipping {path}: "
                f"cannot determine speaker ID[/yellow]"
            )
            continue

        speaker = relative.parts[0]

        grouped[speaker].append(path)

    speakers = sorted(grouped.keys())

    if max_speakers is not None and len(speakers) > max_speakers:
        rng.shuffle(speakers)
        speakers = sorted(speakers[:max_speakers])

    utterances = []

    expected_dim = None

    for speaker in track(speakers, description="Loading embeddings"):
        paths = grouped[speaker]

        if max_utts_per_speaker is not None:
            paths = paths[:max_utts_per_speaker]

        for path in paths:
            x = np.load(path, mmap_mode="r")

            if x.ndim != 2:
                console_print(
                    f"[yellow]Skipping {path}: expected [T,D], got {x.shape}[/yellow]"
                )
                continue

            # x = np.asarray(x, dtype=np.float32)
            T, D = x.shape

            if not np.all(np.isfinite(x)):
                console_print(
                    f"[yellow]Skipping {path}: contains NaN/Inf[/yellow]"
                )
                continue

            if expected_dim is None:
                expected_dim = x.shape[1]

            if x.shape[1] != expected_dim:
                console_print(
                    f"[yellow]Skipping {path}: D={x.shape[1]}, "
                    f"expected {expected_dim}[/yellow]"
                )
                continue


            relative = path.relative_to(root)

            speaker = relative.parts[0]
            video_id = relative.parts[1]
            clip_id = path.stem

            utterance_name = Path(
                *relative.parts[1:]
            ).with_suffix("").as_posix()

            utterances.append(
                Utterance(
                    speaker=speaker,
                    video_id=video_id,
                    clip_id=clip_id,
                    name=utterance_name,
                    path=path,
                    frames=T,
                    dim=D,
                )
            )

    return utterances


# ============================================================
# Basic dataset summary
# ============================================================

def dataset_summary(utterances):
    speakers = sorted(
        set(u.speaker for u in utterances)
    )

    frames = sum(
        u.frames
        for u in utterances
    )

    lengths = np.array(
        [
            u.frames
            for u in utterances
        ],
        dtype=np.int64,
    )

    D = utterances[0].dim

    return {
        "num_speakers":
            len(speakers),

        "num_utterances":
            len(utterances),

        "num_frames":
            int(frames),

        "feature_dim":
            int(D),

        "mean_frames_per_utterance":
            float(lengths.mean()),

        "median_frames_per_utterance":
            float(np.median(lengths)),

        "min_frames_per_utterance":
            int(lengths.min()),

        "max_frames_per_utterance":
            int(lengths.max()),
    }



# ============================================================
# Hierarchical statistics
# ============================================================

def compute_feature_statistics(utterances):
    """
    Exact out-of-core hierarchical statistics.

    Memory complexity:
        O(num_speakers * D + num_utterances * small_metadata)

    instead of:
        O(num_frames * D)

    All first/second-order statistics are computed using
    the complete dataset.
    """

    D = utterances[0].dim

    # ========================================================
    # GLOBAL ACCUMULATORS
    # ========================================================

    global_sum = np.zeros(D, dtype=np.float64)
    global_sq_sum = np.zeros(D, dtype=np.float64)

    global_min = np.full(D, np.inf, dtype=np.float64)
    global_max = np.full(D, -np.inf, dtype=np.float64)

    total_frames = 0

    # ========================================================
    # TEMPORAL ACCUMULATORS
    # ========================================================

    utterance_var_sum = np.zeros(D, dtype=np.float64)
    temporal_diff_rms_sum = np.zeros(D, dtype=np.float64)

    num_utterances = 0

    # ========================================================
    # SPEAKER ACCUMULATORS
    #
    # Only O(num_speakers * D)
    # ========================================================

    speaker_sum = defaultdict(
        lambda: np.zeros(D, dtype=np.float64)
    )

    speaker_sq_sum = defaultdict(
        lambda: np.zeros(D, dtype=np.float64)
    )

    speaker_frames = defaultdict(int)

    speaker_utterances = defaultdict(int)

    # utterance centroid statistics per speaker
    speaker_utt_mean_sum = defaultdict(
        lambda: np.zeros(D, dtype=np.float64)
    )

    speaker_utt_mean_sq_sum = defaultdict(
        lambda: np.zeros(D, dtype=np.float64)
    )

    # ========================================================
    # OUTPUT METADATA
    # ========================================================

    utterance_rows = []

    # ========================================================
    # SINGLE STREAMING PASS
    # ========================================================

    for utt in track(
        utterances,
        description="Computing hierarchical statistics",
    ):

        # mmap: file is NOT fully loaded into RAM
        x = np.load(
            utt.path,
            mmap_mode="r",
        )

        # Keep original float32 mmap.
        # NumPy reductions below accumulate in float64.
        T = x.shape[0]

        # ----------------------------------------------------
        # Global first/second moments
        # ----------------------------------------------------

        x_sum = np.sum(
            x,
            axis=0,
            dtype=np.float64,
        )

        x_sq_sum = np.sum(
            np.square(
                x,
                dtype=np.float64,
            ),
            axis=0,
            dtype=np.float64,
        )

        global_sum += x_sum
        global_sq_sum += x_sq_sum

        global_min = np.minimum(
            global_min,
            np.min(x, axis=0),
        )

        global_max = np.maximum(
            global_max,
            np.max(x, axis=0),
        )

        total_frames += T

        # ----------------------------------------------------
        # Utterance mean / variance
        #
        # E[x²] - E[x]²
        # ----------------------------------------------------

        utt_mean = x_sum / T

        utt_var = (
            x_sq_sum / T
            - utt_mean * utt_mean
        )

        # numerical precision protection
        utt_var = np.maximum(
            utt_var,
            0.0,
        )

        utterance_var_sum += utt_var

        # ----------------------------------------------------
        # Temporal derivative RMS
        #
        # Do NOT use np.diff(x) because that allocates
        # another T x D matrix.
        # ----------------------------------------------------

        if T > 1:

            diff_sq_sum = np.zeros(
                D,
                dtype=np.float64,
            )

            # Process temporal difference in chunks
            chunk_size = 4096

            for start in range(
                0,
                T - 1,
                chunk_size,
            ):
                end = min(
                    start + chunk_size,
                    T - 1,
                )

                a = np.asarray(
                    x[start:end],
                    dtype=np.float64,
                )

                b = np.asarray(
                    x[start + 1:end + 1],
                    dtype=np.float64,
                )

                diff = b - a

                diff_sq_sum += np.sum(
                    diff * diff,
                    axis=0,
                )

            diff_rms = np.sqrt(
                diff_sq_sum / (T - 1)
            )

        else:
            diff_rms = np.zeros(
                D,
                dtype=np.float64,
            )

        temporal_diff_rms_sum += diff_rms

        num_utterances += 1

        # ----------------------------------------------------
        # Speaker accumulators
        # ----------------------------------------------------

        s = utt.speaker

        speaker_sum[s] += x_sum
        speaker_sq_sum[s] += x_sq_sum

        speaker_frames[s] += T
        speaker_utterances[s] += 1

        # utterance centroid distribution
        speaker_utt_mean_sum[s] += utt_mean
        speaker_utt_mean_sq_sum[s] += (
            utt_mean * utt_mean
        )

        # ----------------------------------------------------
        # Embedding norm statistics
        #
        # Avoid allocating norm for entire dataset globally.
        # This allocation is only T values.
        # ----------------------------------------------------

        norm_sq = np.sum(
            np.square(
                x,
                dtype=np.float64,
            ),
            axis=1,
        )

        norms = np.sqrt(norm_sq)

        utterance_rows.append({
            "speaker":
                utt.speaker,

            "utterance":
                utt.name,

            "frames":
                T,

            "mean_feature_variance":
                float(
                    utt_var.mean()
                ),

            "mean_temporal_diff_rms":
                float(
                    diff_rms.mean()
                ),

            "embedding_norm_mean":
                float(
                    norms.mean()
                ),

            "embedding_norm_std":
                float(
                    norms.std()
                ),
        })

        # mmap gets released after iteration
        del x

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    global_mean = (
        global_sum
        / total_frames
    )

    global_var = (
        global_sq_sum
        / total_frames
        - global_mean * global_mean
    )

    global_var = np.maximum(
        global_var,
        0.0,
    )

    mean_within_utterance_var = (
        utterance_var_sum
        / num_utterances
    )

    mean_temporal_diff_rms = (
        temporal_diff_rms_sum
        / num_utterances
    )

    # ========================================================
    # SPEAKER STATISTICS
    # ========================================================

    speakers = sorted(
        speaker_sum.keys()
    )

    speaker_means = []

    speaker_within_vars = []

    speaker_rows = []

    for speaker in speakers:

        N = speaker_frames[speaker]

        mu = (
            speaker_sum[speaker]
            / N
        )

        var = (
            speaker_sq_sum[speaker]
            / N
            - mu * mu
        )

        var = np.maximum(
            var,
            0.0,
        )

        speaker_means.append(mu)
        speaker_within_vars.append(var)

        # ----------------------------------------------------
        # Between-utterance centroid variance
        # ----------------------------------------------------

        U = speaker_utterances[speaker]

        utt_mu = (
            speaker_utt_mean_sum[speaker]
            / U
        )

        between_utt_var = (
            speaker_utt_mean_sq_sum[speaker]
            / U
            - utt_mu * utt_mu
        )

        between_utt_var = np.maximum(
            between_utt_var,
            0.0,
        )

        speaker_rows.append({
            "speaker":
                speaker,

            "utterances":
                U,

            "frames":
                N,

            "mean_within_speaker_variance":
                float(
                    var.mean()
                ),

            "mean_between_utterance_variance":
                float(
                    between_utt_var.mean()
                ),

            "centroid_norm":
                float(
                    np.linalg.norm(mu)
                ),
        })

    speaker_means = np.stack(
        speaker_means,
        axis=0,
    )

    speaker_within_vars = np.stack(
        speaker_within_vars,
        axis=0,
    )

    # ========================================================
    # BETWEEN-SPEAKER VARIANCE
    # ========================================================

    between_speaker_var = np.var(
        speaker_means,
        axis=0,
    )

    mean_within_speaker_var = np.mean(
        speaker_within_vars,
        axis=0,
    )

    # ========================================================
    # SPEAKER VARIANCE RATIO
    # ========================================================

    denom = (
        between_speaker_var
        + mean_within_speaker_var
        + 1e-12
    )

    speaker_ratio = (
        between_speaker_var
        / denom
    )

    # ========================================================
    # TEMPORAL VARIANCE RATIO
    # ========================================================

    temporal_ratio = (
        mean_within_utterance_var
        /
        (
            mean_within_utterance_var
            + between_speaker_var
            + 1e-12
        )
    )

    feature_range = (
        global_max
        - global_min
    )

    # ========================================================
    # FEATURE TABLE
    # ========================================================

    rows = []

    for d in range(D):

        rows.append({
            "feature":
                d,

            "global_mean":
                global_mean[d],

            "global_variance":
                global_var[d],

            "global_std":
                np.sqrt(
                    global_var[d]
                ),

            "global_min":
                global_min[d],

            "global_max":
                global_max[d],

            "global_range":
                feature_range[d],

            "within_utterance_variance":
                mean_within_utterance_var[d],

            "within_speaker_variance":
                mean_within_speaker_var[d],

            "between_speaker_variance":
                between_speaker_var[d],

            "speaker_variance_ratio":
                speaker_ratio[d],

            "temporal_variance_ratio":
                temporal_ratio[d],

            "temporal_diff_rms":
                mean_temporal_diff_rms[d],
        })

    feature_df = pd.DataFrame(
        rows
    )

    utterance_df = pd.DataFrame(
        utterance_rows
    )

    speaker_df = pd.DataFrame(
        speaker_rows
    )

    return (
        feature_df,
        utterance_df,
        speaker_df,
        speaker_means,
    )

# ============================================================
# Sampling frames
# ============================================================
def sample_frames(
    utterances,
    max_frames=100_000,
    seed=0,
):
    rng = np.random.default_rng(seed)

    total_frames = sum(
        u.frames
        for u in utterances
    )

    D = utterances[0].dim

    target = min(
        max_frames,
        total_frames,
    )

    console_print(
        f"[cyan]Sampling {target:,} / "
        f"{total_frames:,} frames "
        f"({100 * target / total_frames:.2f}%)[/cyan]"
    )

    # Preallocate final matrix
    X = np.empty(
        (target, D),
        dtype=np.float32,
    )

    written = 0

    # Probability of selecting a frame
    p = target / total_frames

    for utt in track(
        utterances,
        description="Sampling frames for PCA",
    ):

        x = np.load(
            utt.path,
            mmap_mode="r",
        )

        expected = (
            len(x) * p
        )

        n = int(
            np.floor(expected)
        )

        if rng.random() < (
            expected - n
        ):
            n += 1

        n = min(
            n,
            len(x),
            target - written,
        )

        if n <= 0:
            continue

        idx = rng.choice(
            len(x),
            size=n,
            replace=False,
        )

        X[
            written:written + n
        ] = x[idx]

        written += n

        if written >= target:
            break

    X = X[:written]

    return X

# ============================================================
# PCA / effective dimensionality
# ============================================================

def analyze_dimensionality(X):
    X = np.asarray(X, dtype=np.float64)

    Xc = X - X.mean(axis=0, keepdims=True)

    # covariance eigenvalues
    cov = np.cov(Xc, rowvar=False)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals[::-1]

    total = eigvals.sum()

    if total == 0:
        ratios = np.zeros_like(eigvals)
    else:
        ratios = eigvals / total

    cumulative = np.cumsum(ratios)

    def dims_for(threshold):
        return int(
            np.searchsorted(cumulative, threshold)
            + 1
        )

    # Participation ratio:
    #
    #   (sum lambda)^2 / sum(lambda^2)
    #
    participation_ratio = (
        total * total
        /
        (np.sum(eigvals ** 2) + 1e-12)
    )

    # Entropy effective rank
    p = ratios[ratios > 0]

    entropy = -np.sum(
        p * np.log(p + 1e-12)
    )

    effective_rank = np.exp(entropy)

    return {
        "eigenvalues": eigvals,
        "variance_ratios": ratios,
        "cumulative_variance": cumulative,

        "dims_90": dims_for(0.90),
        "dims_95": dims_for(0.95),
        "dims_99": dims_for(0.99),

        "participation_ratio":
            float(participation_ratio),

        "effective_rank":
            float(effective_rank),
    }


# ============================================================
# Feature correlation
# ============================================================

def analyze_correlation(X):
    X = np.asarray(X, dtype=np.float64)

    corr = np.corrcoef(
        X,
        rowvar=False,
    )

    corr = np.nan_to_num(corr)

    D = corr.shape[0]

    mask = ~np.eye(
        D,
        dtype=bool,
    )

    abs_corr = np.abs(corr[mask])

    stats = {
        "mean_abs_offdiag_correlation":
            float(abs_corr.mean()),

        "median_abs_offdiag_correlation":
            float(np.median(abs_corr)),

        "fraction_abs_corr_gt_0.90":
            float(np.mean(abs_corr > 0.90)),

        "fraction_abs_corr_gt_0.95":
            float(np.mean(abs_corr > 0.95)),

        "fraction_abs_corr_gt_0.99":
            float(np.mean(abs_corr > 0.99)),
    }

    return corr, stats


# ============================================================
# Speaker centroid analysis
# ============================================================

def analyze_speaker_centroids(
    speaker_means,
):
    sim = cosine_similarity_matrix(
        speaker_means
    )

    n = len(sim)

    if n > 1:
        mask = ~np.eye(
            n,
            dtype=bool,
        )

        values = sim[mask]
    else:
        values = np.array([1.0])

    return sim, {
        "mean_cross_speaker_cosine":
            float(values.mean()),

        "median_cross_speaker_cosine":
            float(np.median(values)),

        "std_cross_speaker_cosine":
            float(values.std()),

        "min_cross_speaker_cosine":
            float(values.min()),

        "max_cross_speaker_cosine":
            float(values.max()),
    }


# ============================================================
# Speaker decoding
# ============================================================

def speaker_classification(
    utterances,
    min_utts=3,
    seed=0,
):
    """
    Predict speaker identity from utterance-mean embeddings.

    Embeddings are loaded one utterance at a time using mmap,
    so the complete frame-level dataset is never kept in RAM.

    Only one 512-D centroid per utterance is retained.
    """

    grouped = defaultdict(list)

    for utt in utterances:
        grouped[utt.speaker].append(utt)

    valid_speakers = {
        speaker
        for speaker, utts in grouped.items()
        if len(utts) >= min_utts
    }

    if len(valid_speakers) < 2:
        return None

    X = []
    y = []

    # ========================================================
    # Build utterance-level centroids
    # ========================================================

    for utt in track(
        utterances,
        description="Building speaker-classification dataset",
    ):
        if utt.speaker not in valid_speakers:
            continue

        x = np.load(
            utt.path,
            mmap_mode="r",
        )

        # [T, D] -> [D]
        #
        # Accumulate in float64 for numerical stability.
        centroid = np.mean(
            x,
            axis=0,
            dtype=np.float64,
        )

        X.append(centroid)
        y.append(utt.speaker)

        del x

    X = np.stack(
        X,
        axis=0,
    )

    y = np.asarray(y)

    console_print(
        f"[cyan]Speaker classification dataset: "
        f"{X.shape[0]:,} utterances, "
        f"{X.shape[1]} features, "
        f"{len(valid_speakers):,} speakers[/cyan]"
    )

    # ========================================================
    # Per-speaker train/test split
    # ========================================================

    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []

    for speaker in sorted(valid_speakers):

        idx = np.where(
            y == speaker
        )[0]

        rng.shuffle(idx)

        n_test = max(
            1,
            int(
                round(
                    0.25 * len(idx)
                )
            ),
        )

        # Ensure at least one training sample.
        n_test = min(
            n_test,
            len(idx) - 1,
        )

        test_idx.extend(
            idx[:n_test]
        )

        train_idx.extend(
            idx[n_test:]
        )

    train_idx = np.asarray(
        train_idx,
        dtype=np.int64,
    )

    test_idx = np.asarray(
        test_idx,
        dtype=np.int64,
    )

    # ========================================================
    # Standardization
    #
    # Fit ONLY on training data to avoid leakage.
    # ========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X[train_idx]
    )

    X_test = scaler.transform(
        X[test_idx]
    )

    # ========================================================
    # Linear probe
    # ========================================================

    console_print(
        f"[cyan]Training linear speaker probe: "
        f"{len(train_idx):,} train / "
        f"{len(test_idx):,} test / "
        f"{len(valid_speakers):,} classes[/cyan]"
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=200,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
        verbose=1,
    )

    clf.fit(
        X_train,
        y[train_idx],
    )

    console_print(
        f"[green]Linear speaker probe training completed "
        f"after {clf.n_iter_} epochs[/green]"
    )

    pred = clf.predict(
        X_test
    )

    acc = accuracy_score(
        y[test_idx],
        pred,
    )

    chance = (
        1.0
        / len(valid_speakers)
    )

    return {
        "num_speakers":
            len(valid_speakers),

        "num_utterances":
            len(X),

        "train_samples":
            len(train_idx),

        "test_samples":
            len(test_idx),

        "accuracy":
            float(acc),

        "chance_accuracy":
            float(chance),

        "accuracy_over_chance":
            float(
                acc / chance
            ),
    }

# ============================================================
# Plotting
# ============================================================

def save_feature_variance_plot(
    df,
    out,
):
    plt.figure(figsize=(16, 7))

    plt.plot(
        df["feature"],
        df["within_utterance_variance"],
        label="Within utterance",
    )

    plt.plot(
        df["feature"],
        df["within_speaker_variance"],
        label="Within speaker",
    )

    plt.plot(
        df["feature"],
        df["between_speaker_variance"],
        label="Between speakers",
    )

    plt.xlabel("Feature dimension")
    plt.ylabel("Variance")
    plt.title("Variance decomposition per feature")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out,
        dpi=180,
    )

    plt.close()


def save_speaker_ratio_plot(
    df,
    out,
):
    ordered = df.sort_values(
        "speaker_variance_ratio",
        ascending=False,
    )

    plt.figure(figsize=(16, 6))

    plt.bar(
        np.arange(len(ordered)),
        ordered["speaker_variance_ratio"],
    )

    plt.xlabel("Feature rank")
    plt.ylabel("Speaker variance ratio")
    plt.title(
        "Feature dimensions ranked by speaker dependence"
    )

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=180,
    )

    plt.close()


def save_pca_plot(
    dimensionality,
    out,
):
    cumulative = dimensionality[
        "cumulative_variance"
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
    )

    plt.axhline(
        0.90,
        linestyle="--",
    )

    plt.axhline(
        0.95,
        linestyle="--",
    )

    plt.axhline(
        0.99,
        linestyle="--",
    )

    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")

    plt.title(
        "Effective dimensionality of visual embeddings"
    )

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=180,
    )

    plt.close()


def save_correlation_heatmap(
    corr,
    out,
):
    plt.figure(figsize=(12, 10))

    plt.imshow(
        corr,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar(
        label="Pearson correlation"
    )

    plt.xlabel("Feature")
    plt.ylabel("Feature")

    plt.title(
        "512-D feature correlation matrix"
    )

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=180,
    )

    plt.close()


def save_speaker_similarity_heatmap(
    sim,
    out,
):
    plt.figure(figsize=(10, 9))

    plt.imshow(
        sim,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar(
        label="Cosine similarity"
    )

    plt.xlabel("Speaker")
    plt.ylabel("Speaker")

    plt.title(
        "Speaker centroid cosine similarity"
    )

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=180,
    )

    plt.close()


# ============================================================
# Visualization
# ============================================================

def preprocess_frame(frame):
    """
    Same preprocessing previously validated for the
    DeepAVSR visual frontend.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = gray.astype(
        np.float32
    ) / 255.0

    gray = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    crop = gray[
        56:168,
        56:168,
    ]

    normalized = (
        crop - 0.4161
    ) / 0.1688

    return crop, normalized


def visualize_utterance(
    utt,
    video_root,
    fps=25.0,
):
    video_path = (
        video_root
        / utt.speaker
        / f"{utt.name}.mp4"
    )

    if not video_path.exists():
        raise FileNotFoundError(
            video_path
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    x = np.load(
        utt.path,
        mmap_mode="r",
    )

    # Speaker/utterance-normalized activity
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + 1e-8

    z = (x - mu) / sigma

    frame_idx = 0

    console_print(
        "[cyan]Controls:[/cyan] "
        "SPACE pause | A/D previous/next | Q quit"
    )

    paused = False

    while True:

        if frame_idx >= len(x):
            break

        source_idx = int(
            round(
                frame_idx
                * video_fps
                / fps
            )
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            source_idx,
        )

        ok, frame = cap.read()

        if not ok:
            break

        crop, _ = preprocess_frame(
            frame
        )

        # -----------------------------------------------
        # Build feature heatmap
        # 512 -> 16 x 32
        #
        # This grid is NOT spatial.
        # It is only a convenient display of dimensions.
        # -----------------------------------------------

        feature = x[frame_idx]
        feature_z = z[frame_idx]

        D = len(feature)

        cols = 32
        rows = math.ceil(D / cols)

        padded = np.full(
            rows * cols,
            np.nan,
            dtype=np.float32,
        )

        padded[:D] = feature_z

        heat = padded.reshape(
            rows,
            cols,
        )

        heat = np.nan_to_num(
            heat
        )

        # map roughly [-3,3] -> [0,255]
        heat_img = np.clip(
            (heat + 3.0) / 6.0,
            0,
            1,
        )

        heat_img = (
            heat_img * 255
        ).astype(np.uint8)

        heat_img = cv2.resize(
            heat_img,
            (640, 320),
            interpolation=cv2.INTER_NEAREST,
        )

        heat_img = cv2.applyColorMap(
            heat_img,
            cv2.COLORMAP_TURBO,
        )

        crop_img = (
            np.clip(crop, 0, 1)
            * 255
        ).astype(np.uint8)

        crop_img = cv2.resize(
            crop_img,
            (320, 320),
            interpolation=cv2.INTER_NEAREST,
        )

        crop_img = cv2.cvtColor(
            crop_img,
            cv2.COLOR_GRAY2BGR,
        )

        canvas = np.concatenate(
            [
                crop_img,
                heat_img,
            ],
            axis=1,
        )

        text = (
            f"{utt.speaker}/{utt.name} "
            f"| frame {frame_idx}/{len(x)-1}"
        )

        cv2.putText(
            canvas,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Visual embedding activity",
            canvas,
        )

        delay = (
            0
            if paused
            else max(
                1,
                int(1000 / fps),
            )
        )

        key = cv2.waitKey(delay) & 0xFF

        if key in (
            ord("q"),
            27,
        ):
            break

        elif key == ord(" "):
            paused = not paused

        elif key in (
            ord("a"),
            81,
        ):
            frame_idx = max(
                0,
                frame_idx - 1,
            )
            paused = True
            continue

        elif key in (
            ord("d"),
            83,
        ):
            frame_idx = min(
                len(x) - 1,
                frame_idx + 1,
            )
            paused = True
            continue

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Report
# ============================================================

def create_report(
    summary,
    feature_df,
    dimensionality,
    correlation_stats,
    speaker_stats,
    classification,
):
    lines = []

    lines.append(
        "=" * 80
    )
    lines.append(
        "VISUAL FRONTEND REPRESENTATION ANALYSIS"
    )
    lines.append(
        "=" * 80
    )

    lines.append("")
    lines.append("DATASET")
    lines.append("-" * 80)

    for key, value in summary.items():
        lines.append(
            f"{key:35s}: {value}"
        )

    lines.append("")
    lines.append("DIMENSIONALITY")
    lines.append("-" * 80)

    lines.append(
        f"Original dimensionality        : "
        f"{summary['feature_dim']}"
    )

    lines.append(
        f"Components for 90% variance    : "
        f"{dimensionality['dims_90']}"
    )

    lines.append(
        f"Components for 95% variance    : "
        f"{dimensionality['dims_95']}"
    )

    lines.append(
        f"Components for 99% variance    : "
        f"{dimensionality['dims_99']}"
    )

    lines.append(
        f"Participation-ratio dimension  : "
        f"{dimensionality['participation_ratio']:.2f}"
    )

    lines.append(
        f"Entropy effective rank         : "
        f"{dimensionality['effective_rank']:.2f}"
    )

    lines.append("")
    lines.append("FEATURE REDUNDANCY")
    lines.append("-" * 80)

    for key, value in correlation_stats.items():
        lines.append(
            f"{key:35s}: {value:.6f}"
        )

    lines.append("")
    lines.append("SPEAKER CENTROIDS")
    lines.append("-" * 80)

    for key, value in speaker_stats.items():
        lines.append(
            f"{key:35s}: {value:.6f}"
        )

    if classification is not None:
        lines.append("")
        lines.append("SPEAKER DECODING")
        lines.append("-" * 80)

        for key, value in classification.items():
            lines.append(
                f"{key:35s}: {value}"
            )

    lines.append("")
    lines.append("FEATURE CATEGORIES")
    lines.append("-" * 80)

    # dead / nearly constant
    median_var = feature_df[
        "global_variance"
    ].median()

    dead_threshold = (
        median_var * 1e-3
    )

    dead = feature_df[
        feature_df["global_variance"]
        < dead_threshold
    ]

    identity = feature_df.sort_values(
        "speaker_variance_ratio",
        ascending=False,
    ).head(20)

    dynamic = feature_df.sort_values(
        "temporal_variance_ratio",
        ascending=False,
    ).head(20)

    changing = feature_df.sort_values(
        "temporal_diff_rms",
        ascending=False,
    ).head(20)

    lines.append(
        f"Near-constant features         : "
        f"{len(dead)} / {len(feature_df)}"
    )

    lines.append("")
    lines.append(
        "Top 20 speaker-dependent dimensions:"
    )

    lines.append(
        ", ".join(
            str(int(x))
            for x in identity["feature"]
        )
    )

    lines.append("")
    lines.append(
        "Top 20 temporally-variable dimensions:"
    )

    lines.append(
        ", ".join(
            str(int(x))
            for x in dynamic["feature"]
        )
    )

    lines.append("")
    lines.append(
        "Top 20 fastest-changing dimensions:"
    )

    lines.append(
        ", ".join(
            str(int(x))
            for x in changing["feature"]
        )
    )

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Root containing <speaker>/<utterance>.npy",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "visual_embedding_analysis"
        ),
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-utts-per-speaker",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-pca-frames",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # Visualization
    parser.add_argument(
        "--viz",
        action="store_true",
    )

    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--viz-speaker",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--viz-utterance",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    utterances = load_dataset(
        root=args.embeddings,
        max_speakers=args.max_speakers,
        max_utts_per_speaker=args.max_utts_per_speaker,
        seed=args.seed,
    )

    if not utterances:
        raise RuntimeError(
            "No valid embeddings loaded."
        )

    summary = dataset_summary(
        utterances
    )

    console_print(
        f"\n[bold green]Loaded[/bold green] "
        f"{summary['num_speakers']} speakers, "
        f"{summary['num_utterances']} utterances, "
        f"{summary['num_frames']:,} frames, "
        f"D={summary['feature_dim']}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    (
        feature_df,
        utterance_df,
        speaker_df,
        speaker_means,
    ) = compute_feature_statistics(
        utterances
    )

    feature_df.to_csv(
        args.output
        / "feature_statistics.csv",
        index=False,
    )

    utterance_df.to_csv(
        args.output
        / "utterance_statistics.csv",
        index=False,
    )

    speaker_df.to_csv(
        args.output
        / "speaker_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Sample frames
    # --------------------------------------------------------

    X = sample_frames(
        utterances,
        max_frames=args.max_pca_frames,
        seed=args.seed,
    )

    console_print(
        f"[cyan]Using {len(X):,} sampled frames "
        f"for covariance/PCA analysis[/cyan]"
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    dimensionality = (
        analyze_dimensionality(X)
    )

    np.save(
        args.output
        / "covariance_eigenvalues.npy",
        dimensionality["eigenvalues"],
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    corr, correlation_stats = (
        analyze_correlation(X)
    )

    np.save(
        args.output
        / "feature_correlation.npy",
        corr,
    )

    # --------------------------------------------------------
    # Speaker centroids
    # --------------------------------------------------------

    (
        speaker_similarity,
        speaker_similarity_stats,
    ) = analyze_speaker_centroids(
        speaker_means
    )

    np.save(
        args.output
        / "speaker_centroid_similarity.npy",
        speaker_similarity,
    )

    # --------------------------------------------------------
    # Speaker classification
    # --------------------------------------------------------

    classification = (
        speaker_classification(
            utterances,
            seed=args.seed,
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_feature_variance_plot(
        feature_df,
        args.output
        / "variance_decomposition.png",
    )

    save_speaker_ratio_plot(
        feature_df,
        args.output
        / "speaker_dependence.png",
    )

    save_pca_plot(
        dimensionality,
        args.output
        / "pca_cumulative_variance.png",
    )

    save_correlation_heatmap(
        corr,
        args.output
        / "feature_correlation.png",
    )

    save_speaker_similarity_heatmap(
        speaker_similarity,
        args.output
        / "speaker_similarity.png",
    )

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    results = {
        "dataset": summary,

        "dimensionality": {
            key: value
            for key, value
            in dimensionality.items()
            if not isinstance(
                value,
                np.ndarray,
            )
        },

        "correlation":
            correlation_stats,

        "speaker_similarity":
            speaker_similarity_stats,

        "speaker_classification":
            classification,
    }

    with open(
        args.output / "summary.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    report = create_report(
        summary=summary,
        feature_df=feature_df,
        dimensionality=dimensionality,
        correlation_stats=correlation_stats,
        speaker_stats=speaker_similarity_stats,
        classification=classification,
    )

    with open(
        args.output / "report.txt",
        "w",
    ) as f:
        f.write(report)

    console_print("")
    console_print(report)

    console_print(
        f"\n[bold green]Results saved to:[/bold green] "
        f"{args.output}"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if args.viz:

        if args.video_root is None:
            raise ValueError(
                "--viz requires --video-root"
            )

        candidates = utterances

        if args.viz_speaker:
            candidates = [
                u for u in candidates
                if u.speaker
                == args.viz_speaker
            ]

        if args.viz_utterance:
            candidates = [
                u for u in candidates
                if u.name
                == args.viz_utterance
            ]

        if not candidates:
            raise RuntimeError(
                "No utterance matches visualization filters."
            )

        utt = candidates[0]

        visualize_utterance(
            utt,
            video_root=args.video_root,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()