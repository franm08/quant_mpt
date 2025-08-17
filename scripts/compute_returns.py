# scripts/compute_returns.py
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DB_URL = os.getenv("DB_URL", "sqlite:///quant.db")
engine = create_engine(DB_URL, future=True)

def compute_for_ticker(ticker: str) -> int:
    q = """
    SELECT trade_dt, adj_close
    FROM prices
    WHERE ticker = :t
    ORDER BY trade_dt
    """
    df = pd.read_sql(q, engine, params={"t": ticker}, parse_dates=["trade_dt"])
    if df.empty or len(df) < 2:
        return 0
    df["log_ret"] = np.log(df["adj_close"]).diff()
    out = df.dropna(subset=["log_ret"]).copy()
    out["ticker"] = ticker
    out = out[["ticker", "trade_dt", "log_ret"]]
    out.to_sql("returns", engine, if_exists="append", index=False, method="multi")
    return len(out)

def main():
    # Avoid duplicating: wipe returns if you’re iterating
    with engine.begin() as con:
        con.execute(text("DELETE FROM returns;"))

    tickers = pd.read_sql("SELECT DISTINCT ticker FROM prices", engine)["ticker"].tolist()
    total = 0
    for t in tickers:
        print(f"[returns] {t}")
        total += compute_for_ticker(t)
    print(f"[done] wrote {total} return rows")

if __name__ == "__main__":
    main()