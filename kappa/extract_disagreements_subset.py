import pandas as pd


def main() -> None:
    csvs = [
        # "./kappa/2_iteration_csvs/X.csv",
        # "./kappa/2_iteration_csvs/Y.csv",
    ]
    label_col = "Vai para a etapa de extração?"
    reason_col = "Rejeitado/Não incluído, por quê?"
    out_path = "./kappa/2_filter/disagreements_subset.csv"

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

    disagreements = base.loc[base[label_cols].nunique(axis=1) > 1]

    template = pd.read_csv(csvs[0])
    template_id_col = template.columns[0]

    ids = disagreements[id_col].astype("string").str.strip()
    subset = template.loc[
        template[template_id_col].astype("string").str.strip().isin(ids)
    ]

    if label_col in subset.columns:
        subset[label_col] = ""
    if reason_col in subset.columns:
        subset[reason_col] = ""

    subset.to_csv(out_path, index=False)

    print(f"Itens comparados: {len(base)}")
    print(f"Itens com discordância: {len(disagreements)}")
    print(f"Arquivo gerado: {out_path}")


if __name__ == "__main__":
    main()
