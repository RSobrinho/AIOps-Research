from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.stats.inter_rater import fleiss_kappa


def main():
    load_dotenv()

    csvs = [
        # "./kappa/1_iteration_csvs/X.csv",
        # "./kappa/1_iteration_csvs/Y.csv",
    ]
    label_col = "Vai para a etapa de extração?"
    id_col = "id"
    categories = ["Aceito", "Rejeitado"]

    dfs = []
    for i, path in enumerate(csvs, 1):
        raw = pd.read_csv(path)
        raw_id_col = raw.columns[0]
        raw_label_col = label_col if label_col in raw.columns else raw.columns[1]

        df = raw[[raw_id_col, raw_label_col]].rename(
            columns={raw_id_col: id_col, raw_label_col: f"label_r{i}"}
        )

        df[id_col] = df[id_col].astype("string").str.strip()
        df[f"label_r{i}"] = df[f"label_r{i}"].astype("string").str.strip()
        df[f"label_r{i}"] = df[f"label_r{i}"].replace(
            {"Em dúvida": "Aceito", "": pd.NA}
        )
        dfs.append(df)

    base = dfs[0]
    for df in dfs[1:]:
        base = base.merge(df, on=id_col, how="inner")

    label_cols = [c for c in base.columns if c.startswith("label_r")]
    base = base.dropna(subset=label_cols)
    base = base.loc[base[label_cols].isin(categories).all(axis=1)]

    cat_to_idx = {cat: j for j, cat in enumerate(categories)}
    N = len(base)
    k = len(categories)
    counts = np.zeros((N, k), dtype=int)

    for i, row in enumerate(base[label_cols].to_numpy()):
        for cat, n in Counter(row).items():
            counts[i, cat_to_idx[cat]] = n

    kappa_value = float(fleiss_kappa(counts, method="fleiss"))

    m = len(label_cols)
    print(f"Avaliadores (m): {m}")
    print(f"Itens usados (N): {N}")
    print(f"Categorias (k): {k}")
    print(f"Fleiss kappa (statsmodels): {kappa_value:.6f}")


if __name__ == "__main__":
    main()
