import sys
from pathlib import Path
import numpy as np


def inspect_npy_file(filepath: Path):
    """Carica un file .npy e stampa un report dettagliato per l'array contenuto."""
    print("=" * 65)
    print(f"📄 FILE NPY   : {filepath.name}")
    print(f"📍 PERCORSO   : {filepath.resolve()}")

    try:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"💾 PESO DISCO : {size_mb:.2f} MB")

        # Carica il singolo array Numpy
        data = np.load(filepath, allow_pickle=True)
        print("=" * 65)

        print(f"📐 Shape: {data.shape} | Dtype: {data.dtype} | Dim: {data.ndim}D ({data.size:,} elem)")

        # Statistiche numeriche (se applicabili)
        if np.issubdtype(data.dtype, np.number) and data.size > 0:
            print(
                f"📊 Stats: Min={np.min(data):.4f} | Max={np.max(data):.4f} | Mean={np.mean(data):.4f}"
            )
            nan_cnt = np.isnan(data).sum() if np.issubdtype(data.dtype, np.inexact) else 0
            if nan_cnt > 0:
                print(f"⚠️  NaN presenti: {nan_cnt}")

        print("\n🔍 Anteprima:")
        if data.size <= 6:
            print(f"   {data}")
        else:
            preview = np.array_str(
                data, precision=3, suppress_small=True, max_line_width=75
            )
            # Formatta l'anteprima indentandola per pulizia visiva
            indented_preview = "\n".join("   " + line for line in preview.split("\n"))
            print(indented_preview)

    except Exception as e:
        print(f"❌ ERRORE durante la lettura: {e}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if target.is_file() and target.suffix == ".npy":
        inspect_npy_file(target)
    elif target.is_dir():
        npy_files = sorted(list(target.glob("*.npy")))
        if not npy_files:
            print(f"Nessun file .npy trovato in: {target.resolve()}")
        else:
            print(f"🔎 Trovati {len(npy_files)} file .npy in '{target.name}'. Analisi in corso...\n")
            for f in npy_files:
                inspect_npy_file(f)
    else:
        print(f"❌ Percorso non valido o non è un file .npy: {target}")