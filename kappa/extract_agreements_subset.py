import pandas as pd


def main() -> None:
    csvs = [
        "./kappa/2_iteration_csvs/X.csv",
        "./kappa/2_iteration_csvs/Y.csv",
    ]
    label_col = "Vai para o segundo filtro (texto completo)?"
    out_path = "./kappa/2_iteration_csvs/agreements_subset.csv"

    id_col = "id"
    dfs: list[pd.DataFrame] = []

    for i, path in enumerate(csvs, 1):
        raw = pd.read_csv(path)
        df = raw[[raw.columns[0], label_col]].rename(
            columns={raw.columns[0]: id_col, label_col: f"label_r{i}"}
        )
        df[id_col] = df[id_col].astype("string").str.strip()
        df[f"label_r{i}"] = (
            df[f"label_r{i}"]
            .astype("string")
            .str.strip()
            .replace({"Em dúvida": "Aceito"})
        )
        dfs.append(df)

    base = dfs[0]
    for df in dfs[1:]:
        base = base.merge(df, on=id_col, how="inner")

    label_cols = [c for c in base.columns if c.startswith("label_r")]
    base = base.dropna(subset=label_cols)

    agreements = base.loc[base[label_cols].nunique(axis=1) == 1]
    agreed_label = agreements.set_index(id_col)[label_cols[0]]

    template = pd.read_csv(csvs[0])
    template_id_col = template.columns[0]

    ids = agreements[id_col].astype("string").str.strip()
    subset = template.loc[
        template[template_id_col].astype("string").str.strip().isin(ids)
    ]

    subset_ids = subset[template_id_col].astype("string").str.strip()
    subset[label_col] = subset_ids.map(
        lambda x: (agreed_label.get(x, "") or "").strip()
    ).replace({"Em dúvida": "Aceito"})

    subset.to_csv(out_path, index=False)

    print(f"Itens comparados: {len(base)}")
    print(f"Itens com concordância: {len(agreements)}")
    print(f"Arquivo gerado: {out_path}")


if __name__ == "__main__":
    main()
