#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two directories containing visual embeddings "
            "stored as .npy files with the same relative paths."
        )
    )

    parser.add_argument(
        "root_a",
        type=Path,
        help="First embedding directory (e.g. original/downloaded).",
    )

    parser.add_argument(
        "root_b",
        type=Path,
        help="Second embedding directory (e.g. recomputed).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_comparison.csv"),
        help="Output CSV with per-file statistics.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Compare at most N common files.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for np.isclose.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for np.isclose.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Show K worst files.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N files.",
    )

    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================

def find_npy_files(root):
    result = {}

    for path in root.rglob("*.npy"):
        if not path.is_file():
            continue

        relative = path.relative_to(root)
        result[str(relative)] = path

    return result


def safe_cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < eps:
        return float("nan")

    return float(np.dot(a, b) / denom)


def frame_cosine_similarity(a, b, eps=1e-12):
    """
    a, b: [T, D]

    Returns cosine similarity for every frame.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    dot = np.sum(a * b, axis=1)

    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)

    denom = norm_a * norm_b

    result = np.full(
        len(dot),
        np.nan,
        dtype=np.float64,
    )

    valid = denom > eps

    result[valid] = dot[valid] / denom[valid]

    return result


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if len(a) < 2:
        return float("nan")

    std_a = np.std(a)
    std_b = np.std(b)

    if std_a == 0 or std_b == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def fmt(value, digits=6):
    if value is None:
        return "N/A"

    try:
        if math.isnan(value):
            return "nan"
    except TypeError:
        pass

    return f"{value:.{digits}g}"


# ======================================================================
# Per-file comparison
# ======================================================================

def compare_file(path_a, path_b, rtol, atol):
    try:
        a = np.load(
            path_a,
            mmap_mode="r",
            allow_pickle=False,
        )

        b = np.load(
            path_b,
            mmap_mode="r",
            allow_pickle=False,
        )

    except Exception as exc:
        return {
            "status": "LOAD_ERROR",
            "error": str(exc),
        }

    result = {
        "status": "OK",
        "error": "",
        "shape_a": str(a.shape),
        "shape_b": str(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "size_a": int(a.size),
        "size_b": int(b.size),
    }

    # ------------------------------------------------------------------
    # Shape mismatch
    # ------------------------------------------------------------------

    if a.shape != b.shape:
        result["status"] = "SHAPE_MISMATCH"
        return result

    # Convert for stable metric computation.
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------

    result.update(
        {
            "mean_a": float(np.mean(a64)),
            "mean_b": float(np.mean(b64)),
            "std_a": float(np.std(a64)),
            "std_b": float(np.std(b64)),
            "min_a": float(np.min(a64)),
            "min_b": float(np.min(b64)),
            "max_a": float(np.max(a64)),
            "max_b": float(np.max(b64)),
            "l2_a": float(np.linalg.norm(a64)),
            "l2_b": float(np.linalg.norm(b64)),
            "nan_a": int(np.isnan(a64).sum()),
            "nan_b": int(np.isnan(b64).sum()),
            "inf_a": int(np.isinf(a64).sum()),
            "inf_b": int(np.isinf(b64).sum()),
        }
    )

    # ------------------------------------------------------------------
    # Non-finite values
    # ------------------------------------------------------------------

    finite = (
        np.isfinite(a64)
        & np.isfinite(b64)
    )

    if not np.any(finite):
        result["status"] = "NO_FINITE_VALUES"
        return result

    af = a64[finite]
    bf = b64[finite]

    diff = af - bf
    abs_diff = np.abs(diff)

    # ------------------------------------------------------------------
    # Difference metrics
    # ------------------------------------------------------------------

    result.update(
        {
            "mean_abs_error":
                float(np.mean(abs_diff)),

            "rmse":
                float(np.sqrt(np.mean(diff ** 2))),

            "max_abs_error":
                float(np.max(abs_diff)),

            "mean_signed_error":
                float(np.mean(diff)),

            "global_cosine":
                safe_cosine(af, bf),

            "pearson":
                safe_pearson(af, bf),

            "exact_equal":
                bool(np.array_equal(a, b)),

            "allclose":
                bool(
                    np.allclose(
                        a64,
                        b64,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=True,
                    )
                ),

            "close_fraction":
                float(
                    np.mean(
                        np.isclose(
                            af,
                            bf,
                            rtol=rtol,
                            atol=atol,
                        )
                    )
                ),
        }
    )

    # ------------------------------------------------------------------
    # Relative norm error
    # ------------------------------------------------------------------

    norm_a = np.linalg.norm(af)
    norm_diff = np.linalg.norm(diff)

    if norm_a > 0:
        result["relative_l2_error"] = float(
            norm_diff / norm_a
        )
    else:
        result["relative_l2_error"] = float("nan")

    # ------------------------------------------------------------------
    # Frame-by-frame metrics
    #
    # Expected SEANet visual embedding:
    #
    #     [T, 512]
    # ------------------------------------------------------------------

    if (
        a64.ndim == 2
        and b64.ndim == 2
        and a64.shape[0] > 0
    ):
        frame_cos = frame_cosine_similarity(
            a64,
            b64,
        )

        valid_cos = frame_cos[
            np.isfinite(frame_cos)
        ]

        if len(valid_cos) > 0:
            result.update(
                {
                    "frame_cos_mean":
                        float(np.mean(valid_cos)),

                    "frame_cos_std":
                        float(np.std(valid_cos)),

                    "frame_cos_min":
                        float(np.min(valid_cos)),

                    "frame_cos_max":
                        float(np.max(valid_cos)),

                    "frame_cos_p01":
                        float(np.percentile(valid_cos, 1)),

                    "frame_cos_p05":
                        float(np.percentile(valid_cos, 5)),

                    "frame_cos_p50":
                        float(np.percentile(valid_cos, 50)),

                    "frame_cos_p95":
                        float(np.percentile(valid_cos, 95)),
                }
            )

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    root_a = args.root_a.resolve()
    root_b = args.root_b.resolve()

    if not root_a.is_dir():
        raise NotADirectoryError(root_a)

    if not root_b.is_dir():
        raise NotADirectoryError(root_b)

    print()
    print("=" * 78)
    print("VISUAL EMBEDDING COMPARISON")
    print("=" * 78)
    print()
    print(f"Root A : {root_a}")
    print(f"Root B : {root_b}")
    print()

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    print("Scanning Root A...")
    files_a = find_npy_files(root_a)
    print(f"  {len(files_a):,} .npy files")

    print("Scanning Root B...")
    files_b = find_npy_files(root_b)
    print(f"  {len(files_b):,} .npy files")

    keys_a = set(files_a)
    keys_b = set(files_b)

    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print()
    print("File structure")
    print("-" * 78)
    print(f"Common files : {len(common):,}")
    print(f"Only A       : {len(only_a):,}")
    print(f"Only B       : {len(only_b):,}")

    if only_a:
        print()
        print("First files only in A:")

        for x in only_a[:10]:
            print(f"  {x}")

    if only_b:
        print()
        print("First files only in B:")

        for x in only_b[:10]:
            print(f"  {x}")

    if args.max_files is not None:
        common = common[:args.max_files]

    if not common:
        print()
        print("ERROR: no common .npy files.")
        sys.exit(1)

    print()
    print(f"Comparing {len(common):,} files...")
    print()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    rows = []

    for i, relative in enumerate(common, 1):
        result = compare_file(
            files_a[relative],
            files_b[relative],
            rtol=args.rtol,
            atol=args.atol,
        )

        result["relative_path"] = relative

        rows.append(result)

        if (
            i == 1
            or i % args.progress_every == 0
            or i == len(common)
        ):
            print(
                f"\r[{i:>7,}/{len(common):,}] "
                f"{relative[:55]:<55}",
                end="",
                flush=True,
            )

    print()
    print()

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    all_fields = set()

    for row in rows:
        all_fields.update(row.keys())

    preferred_fields = [
        "relative_path",
        "status",
        "shape_a",
        "shape_b",
        "dtype_a",
        "dtype_b",
        "exact_equal",
        "allclose",
        "close_fraction",
        "mean_abs_error",
        "rmse",
        "max_abs_error",
        "relative_l2_error",
        "global_cosine",
        "pearson",
        "frame_cos_mean",
        "frame_cos_std",
        "frame_cos_min",
        "frame_cos_p01",
        "frame_cos_p05",
        "frame_cos_p50",
        "frame_cos_p95",
        "mean_a",
        "mean_b",
        "std_a",
        "std_b",
        "l2_a",
        "l2_b",
        "nan_a",
        "nan_b",
        "inf_a",
        "inf_b",
        "error",
    ]

    remaining = sorted(
        all_fields - set(preferred_fields)
    )

    fields = [
        x for x in preferred_fields
        if x in all_fields
    ] + remaining

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------

    valid = [
        r for r in rows
        if r.get("status") == "OK"
    ]

    exact = sum(
        bool(r.get("exact_equal", False))
        for r in valid
    )

    allclose = sum(
        bool(r.get("allclose", False))
        for r in valid
    )

    shape_mismatch = sum(
        r.get("status") == "SHAPE_MISMATCH"
        for r in rows
    )

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()

    print(f"Compared files       : {len(rows):,}")
    print(f"Valid comparisons    : {len(valid):,}")
    print(f"Shape mismatches     : {shape_mismatch:,}")
    print(f"Exactly identical    : {exact:,}")
    print(f"Numerically allclose : {allclose:,}")

    if valid:
        # --------------------------------------------------------------
        # Dataset-level distributions
        # --------------------------------------------------------------

        def values(name):
            return np.asarray(
                [
                    r[name]
                    for r in valid
                    if name in r
                    and np.isfinite(r[name])
                ],
                dtype=np.float64,
            )

        print()
        print("Dataset-level metrics")
        print("-" * 78)

        metric_names = [
            "mean_abs_error",
            "rmse",
            "max_abs_error",
            "relative_l2_error",
            "global_cosine",
            "pearson",
            "frame_cos_mean",
            "frame_cos_min",
        ]

        print(
            f"{'Metric':<24}"
            f"{'Mean':>13}"
            f"{'Median':>13}"
            f"{'Min':>13}"
            f"{'Max':>13}"
        )

        print("-" * 76)

        for metric in metric_names:
            x = values(metric)

            if len(x) == 0:
                continue

            print(
                f"{metric:<24}"
                f"{fmt(np.mean(x)):>13}"
                f"{fmt(np.median(x)):>13}"
                f"{fmt(np.min(x)):>13}"
                f"{fmt(np.max(x)):>13}"
            )

        # --------------------------------------------------------------
        # Worst files by frame cosine
        # --------------------------------------------------------------

        sortable = [
            r for r in valid
            if (
                "frame_cos_mean" in r
                and np.isfinite(r["frame_cos_mean"])
            )
        ]

        sortable.sort(
            key=lambda r: r["frame_cos_mean"]
        )

        print()
        print(
            f"Worst {min(args.top_k, len(sortable))} files "
            f"by mean frame cosine similarity"
        )
        print("-" * 78)

        for r in sortable[:args.top_k]:
            print(
                f"{r['frame_cos_mean']:9.6f}  "
                f"RMSE={r['rmse']:.6g}  "
                f"{r['relative_path']}"
            )

    print()
    print(f"Detailed CSV: {args.output}")
    print()


if __name__ == "__main__":
    main()