# scripts/optimize_portfolio.py
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.optimize import minimize

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///quant.db")
TRADING_DAYS = 252  # annualization factor

# ---------- utils ----------
def load_returns():
    """
    Load daily log returns from quant.db -> returns table with columns:
      ticker, trade_dt, log_ret
    Returns a DataFrame wide format: index=trade_dt, columns=tickers, values=log_ret
    """
    eng = create_engine(DB_URL)
    df = pd.read_sql("SELECT ticker, trade_dt, log_ret FROM returns", eng, parse_dates=["trade_dt"])
    if df.empty:
        raise RuntimeError("No data in 'returns' table. Run ingest + calc_returns first.")
    wide = df.pivot(index="trade_dt", columns="ticker", values="log_ret").dropna(how="any")
    return wide

def annualized_stats(logret_wide: pd.DataFrame):
    """
    From daily log returns:
      mu (annualized mean simple return approx) and Sigma (annualized covariance)
    """
    # Convert daily log returns to daily simple returns for mu (approx)
    # r_simple ≈ exp(log_r) - 1
    daily_simple = np.exp(logret_wide) - 1.0
    mu_daily = daily_simple.mean(axis=0).values  # shape (n,)
    mu_annual = mu_daily * TRADING_DAYS

    # Covariance of daily simple returns (close enough for planning), annualized
    Sigma_daily = daily_simple.cov().values  # shape (n,n)
    Sigma_annual = Sigma_daily * TRADING_DAYS

    tickers = list(logret_wide.columns)
    return mu_annual, Sigma_annual, tickers

def summarize(weights, mu, Sigma, rf=0.0):
    exp_ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (exp_ret - rf) / vol if vol > 0 else np.nan
    return exp_ret, vol, sharpe

def clip_weights(w):
    w = np.clip(w, 0.0, 1.0)
    s = w.sum()
    if s == 0:
        return np.ones_like(w) / len(w)
    return w / s

# ---------- optimizers ----------
def min_variance_given_return(mu, Sigma, target_ret, bounds=(0,1)):
    """
    Minimize variance subject to:
      sum(w) = 1
      w @ mu >= target_ret
      bounds on each weight (default [0,1])
    """
    n = len(mu)
    w0 = np.ones(n) / n
    bounds_list = [bounds] * n

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - target_ret},
    ]

    def objective(w):
        return w @ Sigma @ w

    res = minimize(objective, w0, method="SLSQP", bounds=bounds_list, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Min-variance optimization failed: {res.message}")
    return clip_weights(res.x)

def max_sharpe(mu, Sigma, rf=0.0, bounds=(0,1)):
    """
    Maximize Sharpe => minimize negative Sharpe:
      -( (w@mu - rf) / sqrt(w@Sigma@w) )
    with sum(w)=1 and bounds on w.
    """
    n = len(mu)
    w0 = np.ones(n) / n
    bounds_list = [bounds] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        num = w @ mu - rf
        den = np.sqrt(w @ Sigma @ w)
        if den <= 0:
            return 1e6
        return -num / den

    res = minimize(objective, w0, method="SLSQP", bounds=bounds_list, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {res.message}")
    return clip_weights(res.x)

# ---------- main flows ----------
def run_minvar(mu, Sigma, tickers, target_ret, outdir, rf=0.0):
    w = min_variance_given_return(mu, Sigma, target_ret)
    exp_ret, vol, sharpe = summarize(w, mu, Sigma, rf)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    weights_df = pd.DataFrame({"ticker": tickers, "weight": w})
    weights_path = Path(outdir) / f"weights_minvar_{target_ret:.4f}.csv"
    weights_df.to_csv(weights_path, index=False)

    metrics_df = pd.DataFrame([{"objective": "minvar", "target_return": target_ret, "exp_return": exp_ret, "vol": vol, "sharpe": sharpe}])
    metrics_path = Path(outdir) / f"metrics_minvar_{target_ret:.4f}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[minvar] saved weights -> {weights_path}")
    print(f"[minvar] saved metrics -> {metrics_path}")

def run_maxsharpe(mu, Sigma, tickers, outdir, rf):
    w = max_sharpe(mu, Sigma, rf)
    exp_ret, vol, sharpe = summarize(w, mu, Sigma, rf)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    weights_df = pd.DataFrame({"ticker": tickers, "weight": w})
    weights_path = Path(outdir) / f"weights_maxsharpe_rf{rf:.4f}.csv"
    weights_df.to_csv(weights_path, index=False)

    metrics_df = pd.DataFrame([{"objective": "maxsharpe", "rf": rf, "exp_return": exp_ret, "vol": vol, "sharpe": sharpe}])
    metrics_path = Path(outdir) / f"metrics_maxsharpe_rf{rf:.4f}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"[maxsharpe] saved weights -> {weights_path}")
    print(f"[maxsharpe] saved metrics -> {metrics_path}")

def run_sweep(mu, Sigma, tickers, outdir, points=50, rf=0.0, bounds=(0,1)):
    """
    Efficient frontier by sweeping target returns between min(mu) and max(mu).
    """
    lo, hi = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(lo, hi, points)

    rows = []
    weights_list = []

    for t in targets:
        try:
            w = min_variance_given_return(mu, Sigma, t, bounds=bounds)
            exp_ret, vol, sharpe = summarize(w, mu, Sigma, rf)
            rows.append({"target": t, "exp_return": exp_ret, "vol": vol, "sharpe": sharpe})
            weights_list.append(w)
        except Exception as e:
            # infeasible target can occur—skip
            continue

    if not rows:
        raise RuntimeError("No feasible points for frontier sweep.")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    frontier_df = pd.DataFrame(rows)
    # Save weights as JSON strings aligned with rows
    weights_json = [json.dumps(dict(zip(tickers, w))) for w in weights_list]
    frontier_df["weights_json"] = weights_json

    csv_path = Path(outdir) / "frontier_points.csv"
    frontier_df.to_csv(csv_path, index=False)

    # Plot frontier
    plt.figure(figsize=(7,5))
    plt.scatter(frontier_df["vol"], frontier_df["exp_return"], s=14)
    plt.xlabel("Volatility (σ, annualized)")
    plt.ylabel("Expected Return (annualized)")
    plt.title("Efficient Frontier")
    plt.grid(True, alpha=0.3)
    png_path = Path(outdir) / "frontier.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=160)
    plt.close()

    print(f"[sweep] saved frontier CSV -> {csv_path}")
    print(f"[sweep] saved frontier plot -> {png_path}")

def main():
    parser = argparse.ArgumentParser(description="Portfolio optimization on quant.db returns")
    parser.add_argument("--objective", choices=["minvar", "maxsharpe", "sweep"], required=True)
    parser.add_argument("--target-return", type=float, default=None, help="Annualized target return for minvar")
    parser.add_argument("--rf", type=float, default=0.02, help="Risk-free rate (annualized) for Sharpe")
    parser.add_argument("--points", type=int, default=50, help="Number of sweep points for frontier")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to write outputs")
    args = parser.parse_args()

    logret_wide = load_returns()
    mu, Sigma, tickers = annualized_stats(logret_wide)

    if args.objective == "minvar":
        if args.target_return is None:
            raise SystemExit("--target-return is required for objective=minvar")
        run_minvar(mu, Sigma, tickers, args.target_return, args.outdir, rf=args.rf)

    elif args.objective == "maxsharpe":
        run_maxsharpe(mu, Sigma, tickers, args.outdir, rf=args.rf)

    elif args.objective == "sweep":
        run_sweep(mu, Sigma, tickers, args.outdir, points=args.points, rf=args.rf)

if __name__ == "__main__":
    main()