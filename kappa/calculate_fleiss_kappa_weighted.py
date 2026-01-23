import os
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.stats.inter_rater import fleiss_kappa as fleiss_kappa_unweighted

import matplotlib.pyplot as plt


def weight_matrix(k: int, kind: str) -> np.ndarray:
    # Categories are assumed ordinal in the order given by CATEGORIES.
    # kind: "linear" or "quadratic"
    idx = np.arange(k)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(float)
    denom = (k - 1) if k > 1 else 1.0

    if kind == "quadratic":
        return 1.0 - (dist**2) / (denom**2)
    return 1.0 - dist / denom  # linear (default)


def fleiss_weighted_kappa(counts: np.ndarray, W: np.ndarray) -> float:
    """
    Weighted multi-rater kappa using the ordered-pairs coincidence formulation:
      Po = sum_{c,d} W_cd * O_cd / (N*m*(m-1))
      Pe = sum_{c,d} W_cd * p_c * p_d
      kappa = (Po - Pe) / (1 - Pe)

    counts: N x k counts per item (each row sums to m)
    W: k x k weight matrix in [0,1], with 1 on diagonal
    """
    counts = np.asarray(counts, dtype=float)
    N, k = counts.shape
    m = counts.sum(axis=1)[0]

    # Coincidence matrix for ordered pairs of ratings within each item
    # For each item: add n_c * n_d for c != d; add n_c*(n_c-1) for c == d
    O = np.zeros((k, k), dtype=float)
    for n in counts:
        O += np.outer(n, n) - np.diag(
            n
        )  # subtract self-pairs to get ordered pairs without replacement

    total_pairs = N * m * (m - 1)
    Po = float((W * O).sum() / total_pairs)

    # Expected under independence from marginal proportions p
    p = counts.sum(axis=0) / (N * m)
    Pe = float((W * np.outer(p, p)).sum())

    return (Po - Pe) / (1.0 - Pe)


def main():
    load_dotenv()

    csvs = ["./kappa/teste1.csv", "./kappa/teste1.csv"]
    id_col = "Título do Artigo"
    label_col = "Vai para a etapa de extração?"
    categories = ["Aceito", "Em dúvida", "Rejeitado"]
    weight_type = "linear"  # linear|quadratic

    # Read and merge all raters by id
    dfs = []
    for i, path in enumerate(csvs, 1):
        df = pd.read_csv(path, usecols=[id_col, label_col]).rename(
            columns={label_col: f"label_r{i}"}
        )
        df[f"label_r{i}"] = df[f"label_r{i}"].astype(str).str.strip()
        dfs.append(df)

    base = dfs[0]
    for df in dfs[1:]:
        base = base.merge(df, on=id_col, how="inner")

    label_cols = [c for c in base.columns if c.startswith("label_r")]

    # Build N x k count matrix (assume .env and data are correct; no extra guards)
    cat_to_idx = {cat: j for j, cat in enumerate(categories)}
    N = len(base)
    k = len(categories)
    counts = np.zeros((N, k), dtype=int)

    for i, row in enumerate(base[label_cols].to_numpy()):
        for cat, n in Counter(row).items():
            counts[i, cat_to_idx[cat]] = n

    # Unweighted Fleiss (statsmodels) for reference
    kappa_unw = float(fleiss_kappa_unweighted(counts, method="fleiss"))

    # Weighted kappa (ordinal weights by category order)
    W = weight_matrix(k, weight_type)
    kappa_w = float(fleiss_weighted_kappa(counts, W))

    # Coincidence / "confusion" matrix for visualization (ordered pairs aggregated across all items)
    O = np.zeros((k, k), dtype=float)
    for n in counts.astype(float):
        O += np.outer(n, n) - np.diag(n)

    # Print summary
    m = len(label_cols)
    print(f"Avaliadores (m): {m}")
    print(f"Itens usados (N): {N}")
    print(f"Categorias (k): {k}")
    print(f"Fleiss kappa (statsmodels, não ponderado): {kappa_unw:.6f}")
    print(f"Fleiss kappa ponderado ({weight_type}): {kappa_w:.6f}")

    # Plot (heatmap) of coincidence matrix
    # Normalize by total ordered pairs to make it comparable across runs
    O_norm = O / (N * m * (m - 1))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(O_norm, interpolation="nearest")
    ax.set_title("Matriz de coincidência (pares ordenados), normalizada")
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_yticklabels(categories)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate values
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{O_norm[i, j]:.3f}", ha="center", va="center")

    plt.tight_layout()

    # Save + show
    out_path = os.environ.get("PLOT_OUT", "./kappa/confusion_coincidence.png").strip()
    plt.savefig(out_path, dpi=200)

    print(f"Plot salvo em: {out_path}")


if __name__ == "__main__":
    main()
