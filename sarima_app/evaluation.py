"""Fonctions d'évaluation annuelle cumulée et agrégation temporelle."""
from __future__ import annotations

import pandas as pd


def aggregate_by_period(df: pd.DataFrame, months_per_period: int) -> pd.DataFrame:
    if months_per_period == 1:
        return df.copy()
    work = df.copy()
    idx = work["date"].dt.month
    start_month = ((idx - 1) // months_per_period) * months_per_period + 1
    work["period_start"] = pd.to_datetime(
        dict(year=work["date"].dt.year, month=start_month, day=1)
    )
    out = work.groupby("period_start", as_index=False)["value"].sum()
    out = out.rename(columns={"period_start": "date"})
    return out.sort_values("date")


def annual_comparison(real_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    r = real_df.copy().set_index("date")
    p = pred_df.copy().set_index("date")
    merged = r.join(p, how="inner", lsuffix="_real", rsuffix="_pred").dropna()
    if merged.empty:
        return pd.DataFrame(columns=["Année", "Σ Réel", "Σ Prévu", "Écart absolu", "Écart relatif %"])

    merged["year"] = merged.index.year
    yearly = merged.groupby("year").agg(real=("value_real", "sum"), pred=("value_pred", "sum"), n=("value_real", "size"))
    full = yearly[yearly["n"] >= 12]
    if full.empty:
        return pd.DataFrame(columns=["Année", "Σ Réel", "Σ Prévu", "Écart absolu", "Écart relatif %"])

    full["abs"] = (full["real"] - full["pred"]).abs()
    full["rel"] = (full["abs"] / full["real"].abs().replace(0, pd.NA) * 100).fillna(0)
    result = full.reset_index().rename(
        columns={"year": "Année", "real": "Σ Réel", "pred": "Σ Prévu", "abs": "Écart absolu", "rel": "Écart relatif %"}
    )
    return result[["Année", "Σ Réel", "Σ Prévu", "Écart absolu", "Écart relatif %"]]
