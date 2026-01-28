from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a leaderboard table from experiments CSV.")
    parser.add_argument("--in", dest="in_path", default="results/experiments.csv")
    parser.add_argument("--out-csv", default="results/experiments_summary.csv")
    parser.add_argument("--out-md", default="results/experiments_summary.md")
    parser.add_argument(
        "--sort",
        default="F1_weighted",
        choices=["Accuracy", "F1_weighted", "F1_macro", "AIC", "BIC"],
    )
    parser.add_argument(
        "--with-n",
        action="store_true",
        help="Include the 'n' column (number of runs / seeds) in outputs.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    # Normalize expected columns from different versions of experiments output
    rename_map = {
        "name": "Model",
        "accuracy": "Accuracy",
        "f1_weighted": "F1_weighted",
        "f1_macro": "F1_macro",
        "aic": "AIC",
        "bic": "BIC",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Ensure optional columns exist
    for col in ["AIC", "BIC"]:
        if col not in df.columns:
            df[col] = np.nan

    if "Model" not in df.columns:
        raise SystemExit("Missing 'name'/'Model' column in input CSV")

    grouped = (
        df.groupby("Model", dropna=False)
        .agg(
            n=("Model", "size"),
            Accuracy=("Accuracy", _safe_mean),
            F1_weighted=("F1_weighted", _safe_mean),
            F1_macro=("F1_macro", _safe_mean),
            AIC=("AIC", _safe_mean),
            BIC=("BIC", _safe_mean),
        )
        .reset_index()
    )

    sort_col = args.sort
    grouped = grouped.sort_values(by=sort_col, ascending=False, na_position="last")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.with_n:
        grouped.to_csv(out_csv, index=False)
    else:
        grouped.drop(columns=["n"]).to_csv(out_csv, index=False)

    # Markdown table with formatting similar to the screenshot
    def fmt_float(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.6f}"

    def fmt_big(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.0f}"

    md = grouped.copy()
    md["Accuracy"] = md["Accuracy"].map(fmt_float)
    md["F1_weighted"] = md["F1_weighted"].map(fmt_float)
    md["F1_macro"] = md["F1_macro"].map(fmt_float)
    md["AIC"] = md["AIC"].map(fmt_big)
    md["BIC"] = md["BIC"].map(fmt_big)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    cols = ["Model", "Accuracy", "F1_weighted", "F1_macro", "AIC", "BIC"]
    if args.with_n:
        cols = ["Model", "n", *cols[1:]]

    md_table = md[cols].to_markdown(index=False)

    best_model = str(grouped.iloc[0]["Model"]) if len(grouped) else ""
    best_score = grouped.iloc[0][args.sort] if len(grouped) else float("nan")
    if best_model:
        if isinstance(best_score, float) and np.isnan(best_score):
            header = f"Najlepszy model wg {args.sort}: {best_model}\n\n"
        else:
            header = f"Najlepszy model wg {args.sort}: {best_model} ({args.sort}={float(best_score):.6f})\n\n"
    else:
        header = ""
    out_md.write_text(header + md_table + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
