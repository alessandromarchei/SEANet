#!/usr/bin/env python3

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("generated")
    args = parser.parse_args()

    ref = np.load(args.reference).astype(np.float64)
    gen = np.load(args.generated).astype(np.float64)

    print("=" * 70)
    print("REFERENCE:", args.reference)
    print("GENERATED:", args.generated)
    print("=" * 70)

    print("Shapes:", ref.shape, gen.shape)

    if ref.shape != gen.shape:
        raise RuntimeError(
            f"Shape mismatch: {ref.shape} vs {gen.shape}"
        )

    diff = gen - ref

    print("\nGLOBAL")
    print(f"Ref mean/std : {ref.mean():.6f} / {ref.std():.6f}")
    print(f"Gen mean/std : {gen.mean():.6f} / {gen.std():.6f}")
    print(f"MAE          : {np.mean(np.abs(diff)):.6f}")
    print(f"MSE          : {np.mean(diff**2):.6f}")
    print(f"RMSE         : {np.sqrt(np.mean(diff**2)):.6f}")
    print(f"Max abs err  : {np.max(np.abs(diff)):.6f}")

    # ---------------------------------------------------------
    # Global cosine
    # ---------------------------------------------------------

    global_cos = np.dot(
        ref.reshape(-1),
        gen.reshape(-1)
    ) / (
        np.linalg.norm(ref)
        * np.linalg.norm(gen)
        + 1e-12
    )

    print(f"Global cosine: {global_cos:.6f}")

    # ---------------------------------------------------------
    # Frame-wise cosine
    # ---------------------------------------------------------

    dot = np.sum(ref * gen, axis=1)

    norms = (
        np.linalg.norm(ref, axis=1)
        * np.linalg.norm(gen, axis=1)
    )

    cosine = dot / (norms + 1e-12)

    print("\nFRAME-WISE COSINE")
    print(f"Mean   : {cosine.mean():.6f}")
    print(f"Median : {np.median(cosine):.6f}")
    print(f"Min    : {cosine.min():.6f}")
    print(f"Max    : {cosine.max():.6f}")
    print(f"Std    : {cosine.std():.6f}")

    print("\nFirst 20:")
    print(
        np.array2string(
            cosine[:20],
            precision=4,
            suppress_small=True,
        )
    )

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    corr = np.corrcoef(
        ref.reshape(-1),
        gen.reshape(-1),
    )[0, 1]

    print(f"\nGlobal Pearson: {corr:.6f}")

    # ---------------------------------------------------------
    # Norm comparison
    # ---------------------------------------------------------

    ref_norm = np.linalg.norm(ref, axis=1)
    gen_norm = np.linalg.norm(gen, axis=1)

    ratio = gen_norm / (ref_norm + 1e-12)

    print("\nFRAME NORMS")
    print(f"Reference mean norm : {ref_norm.mean():.6f}")
    print(f"Generated mean norm : {gen_norm.mean():.6f}")
    print(f"Mean ratio gen/ref  : {ratio.mean():.6f}")

    # ---------------------------------------------------------
    # Check temporal shift
    # ---------------------------------------------------------

    print("\nTEMPORAL ALIGNMENT TEST")

    best_shift = None
    best_cos = -1

    for shift in range(-10, 11):

        if shift < 0:
            a = ref[-shift:]
            b = gen[:shift]

        elif shift > 0:
            a = ref[:-shift]
            b = gen[shift:]

        else:
            a = ref
            b = gen

        cos = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1)
            * np.linalg.norm(b, axis=1)
            + 1e-12
        )

        score = cos.mean()

        print(
            f"shift {shift:+3d}: "
            f"mean cosine = {score:.6f}"
        )

        if score > best_cos:
            best_cos = score
            best_shift = shift

    print()
    print(
        f"BEST TEMPORAL SHIFT: {best_shift:+d} frames "
        f"(cosine={best_cos:.6f})"
    )


if __name__ == "__main__":
    main()