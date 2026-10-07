"""
SberIndex Hackathon — Time Series Forecasting Track.

Pipeline:
  1. Load config.yaml
  2. Read & preprocess consumption.parquet (+ category filter, top-3 territories)
  3. Join market_access.parquet, build lag features
  4. Structural break detection with ruptures (Pelt, RBF) -> output/structural_breaks.png
  5. Train Prophet & LightGBM per territory, evaluate MAE / R2 on horizons
  6. Export forecasts plot, metrics CSV and a pretty console table
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ruptures as rpt
from lightgbm import LGBMRegressor
from prophet import Prophet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse date column that stores month-year strings (e.g. '2023-01')."""
    for fmt in ("%Y-%m", "%m.%Y", "%Y-%m-%d", "%d.%m.%Y"):
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    return pd.to_datetime(series, errors="coerce")


def filter_category(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Keep only the aggregate category row; fall back gracefully if needed."""
    cats = df["category"].astype(str).unique()
    print(f"[INFO] Unique categories ({len(cats)}): {list(cats)[:20]}")

    mask = df["category"] == target
    if not mask.any():
        lowered = [c for c in cats if "все" in c.lower() and "катег" in c.lower()]
        if lowered:
            print(f"[WARN] '{target}' not found, using '{lowered[0]}' instead")
            mask = df["category"] == lowered[0]
    if not mask.any():
        mode = df["category"].value_counts().idxmax()
        print(f"[WARN] No aggregate category found, falling back to most frequent '{mode}'")
        mask = df["category"] == mode

    out = df[mask].copy()
    print(f"[INFO] Rows after category filter: {len(out)}")
    return out


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))



# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def load_data(cfg: dict) -> tuple:
    cons_path = ROOT / cfg["paths"]["consumption"]
    ma_path = ROOT / cfg["paths"]["market_access"]
    print(f"[INFO] Reading {cons_path}")
    df = pd.read_parquet(cons_path, engine="pyarrow")

    df["date"] = parse_dates(df["date"])
    if df["date"].isna().any():
        raise ValueError("Failed to parse `date` column")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    df = filter_category(df, cfg.get("category_filter", "Все категории"))

    top_n = int(cfg.get("top_n_territories", 3))
    top_ids = (
        df.groupby("territory_id")["value"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    )
    print(f"[INFO] Top-{top_n} territories by total consumption: {top_ids}")
    df = df[df["territory_id"].isin(top_ids)].copy()

    print(f"[INFO] Reading {ma_path}")
    ma = pd.read_parquet(ma_path, engine="pyarrow").drop_duplicates(subset=["territory_id"])
    df = df.merge(ma, on="territory_id", how="left")
    n_missing = int(df["market_access"].isna().sum())
    if n_missing:
        df["market_access"] = df["market_access"].fillna(df["market_access"].median())
        print(f"[WARN] market_access missing for {n_missing} rows -> filled with median")

    df = df.sort_values(["territory_id", "date"]).reset_index(drop=True)
    df["lag_1"] = df.groupby("territory_id")["value"].shift(1)
    df["lag_3"] = df.groupby("territory_id")["value"].shift(3)
    print(
        f"[INFO] Dataset ready: {len(df)} rows, "
        f"months {df['date'].min():%Y-%m}..{df['date'].max():%Y-%m}"
    )
    return df, top_ids


def detect_structural_breaks(df: pd.DataFrame, top_id, cfg: dict) -> None:
    rcfg = cfg.get("ruptures", {})
    model_name = rcfg.get("model", "rbf")
    algo_name = rcfg.get("algo", "Pelt")
    pen = float(rcfg.get("pen", 20.0))

    series = df[df["territory_id"] == top_id].sort_values("date")
    dates = series["date"].to_numpy()
    signal = series["value"].to_numpy(dtype=float)
    # z-normalisation so the RBF cost is scale invariant
    signal_z = ((signal - signal.mean()) / (signal.std() + 1e-9)).reshape(-1, 1)

    print(f"[INFO] ruptures: algo={algo_name}, model={model_name}, pen={pen}, n={len(signal)}")
    if algo_name.lower() == "pelt":
        algo = rpt.Pelt(model=model_name, min_size=3, jump=1)
    else:
        algo = rpt.Binseg(model=model_name, min_size=3, jump=1)
    bkps = algo.fit(signal_z).predict(pen=pen)
    # ruptures appends n as final breakpoint -> keep only real breaks
    real_bkps = [b for b in bkps if b < len(signal)]
    break_dates = [str(pd.Timestamp(d).date()) for d in dates[real_bkps]]
    print(f"[INFO] Breakpoints found at indices: {real_bkps} ({break_dates or 'none'})")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, signal, marker="o", color="black", linewidth=1.6, label="Consumption")
    for b in real_bkps:
        ax.axvline(dates[b], color="red", linestyle="--", linewidth=1.8,
                   label="Structural break" if b == real_bkps[0] else None)
    if not real_bkps:
        ax.text(0.5, 0.95, "No structural breaks detected", transform=ax.transAxes,
                ha="center", va="top", color="red")
    ax.set_title(f"Structural Breaks (Pelt, RBF) - Territory {top_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out_path = ROOT / cfg["paths"]["output_dir"] / "structural_breaks.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def train_test_split(df: pd.DataFrame, train_months: int) -> tuple:
    train, test = [], []
    for _, grp in df.groupby("territory_id"):
        grp = grp.sort_values("date")
        train.append(grp.iloc[:train_months])
        test.append(grp.iloc[train_months:])
    return pd.concat(train), pd.concat(test)


def fit_predict_prophet(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality="auto")
    m.fit(train[["date", "value"]].rename(columns={"date": "ds", "value": "y"}))
    future = test[["date"]].rename(columns={"date": "ds"})
    return m.predict(future)["yhat"].to_numpy(dtype=float)


def fit_predict_lgbm(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    features = ["lag_1", "lag_3", "market_access"]
    tr = train.dropna(subset=features)
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(tr[features], tr["value"])
    return model.predict(test[features])


def evaluate(df: pd.DataFrame, top_ids: list, cfg: dict) -> tuple:
    train, test = train_test_split(df, int(cfg["train_months"]))
    horizons = list(cfg["horizons"])
    rows = []

    for tid in top_ids:
        tr = train[train["territory_id"] == tid]
        te = test[test["territory_id"] == tid].sort_values("date").reset_index(drop=True)
        print(f"\n[INFO] Territory {tid}: train {len(tr)} months, test {len(te)} months")

        preds = {
            "Prophet": fit_predict_prophet(tr, te),
            "LightGBM": fit_predict_lgbm(tr, te, cfg["random_seed"]),
        }
        for model_name, y_pred in preds.items():
            for h in horizons:
                y_true = te["value"].to_numpy()[:h]
                y_hat = y_pred[:h]
                rows.append({
                    "territory_id": tid,
                    "model": model_name,
                    "horizon": h,
                    "mae": mae(y_true, y_hat),
                    "r2": safe_r2(y_true, y_hat),
                })

    return pd.DataFrame(rows), train, test



def plot_forecasts(train: pd.DataFrame, test: pd.DataFrame, top_id, cfg: dict) -> None:
    tr = train[train["territory_id"] == top_id].sort_values("date")
    te = test[test["territory_id"] == top_id].sort_values("date").reset_index(drop=True)
    prophet_pred = fit_predict_prophet(tr, te)
    lgbm_pred = fit_predict_lgbm(tr, te, cfg["random_seed"])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(tr["date"], tr["value"], marker="o", color="black", label="Actual (train)")
    ax.plot(te["date"], te["value"], marker="o", color="steelblue", label="Actual (test)")
    ax.plot(te["date"], prophet_pred, marker="s", color="darkorange", label="Prophet")
    ax.plot(te["date"], lgbm_pred, marker="^", color="green", label="LightGBM")
    ax.axvline(tr["date"].max(), color="gray", linestyle=":", linewidth=1.5, label="Train/Test split")
    ax.set_title(f"Actual vs Forecasts - Territory {top_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out_path = ROOT / cfg["paths"]["output_dir"] / "forecasts.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def export_metrics(metrics: pd.DataFrame, cfg: dict) -> None:
    out_path = ROOT / cfg["paths"]["output_dir"] / "metrics.csv"
    metrics.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[INFO] Saved {out_path}\n")

    print("=" * 80)
    print("METRICS (MAE / R2) BY TERRITORY, MODEL AND HORIZON")
    print("=" * 80)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    pivot = metrics.pivot_table(index=["territory_id", "model"], columns="horizon", values="mae")
    pivot.columns = [f"MAE@{c}m" for c in pivot.columns]
    pivot = pivot.reset_index()
    print("-" * 80)
    print("MAE SUMMARY")
    print("-" * 80)
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 80)


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    print("[INFO] Config loaded:")
    print(yaml.dump(cfg, allow_unicode=True, sort_keys=False))

    out_dir = ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df, top_ids = load_data(cfg)
    detect_structural_breaks(df, top_ids[0], cfg)
    metrics, train, test = evaluate(df, top_ids, cfg)
    plot_forecasts(train, test, top_ids[0], cfg)
    export_metrics(metrics, cfg)

    print("[INFO] Pipeline finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

    print(f"[INFO] Saved {out_path}")
