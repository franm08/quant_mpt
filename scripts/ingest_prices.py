# scripts/ingest_prices.py
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ----------------------------
# env & engine
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DB_URL = os.getenv("DB_URL", "sqlite:///quant.db")
START = os.getenv("START", "2019-01-01")
END   = os.getenv("END",   datetime.now().strftime("%Y-%m-%d"))

engine = create_engine(DB_URL, future=True)

# ----------------------------
# schema init
# ----------------------------
def init_schema():
    schema_path = ROOT / "sql" / "01_schema.sql"
    sql_txt = schema_path.read_text(encoding="utf-8")

    # SQLite (via SQLAlchemy) allows one statement per execute()
    stmts = [s.strip() for s in sql_txt.split(";") if s.strip()]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))

# ----------------------------
# single-ticker loader
# ----------------------------
def load_prices(ticker: str) -> int:
    import yfinance as yf
    df = yf.download(ticker, start=START, end=END, progress=False)  # auto_adjust=True by default
    if df.empty:
        print(f"[warn] no data for {ticker}")
        return 0

    # Robustly pick a price column
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"No price column for {ticker}: {df.columns.tolist()}")

    # Make the transformation step-by-step & explicit
    # 1) Keep only the price column and rename to adj_close
    tmp = df[[price_col]].copy()
    tmp.columns = ["adj_close"]

    # 2) Turn the index into a normal column named trade_dt
    tmp = tmp.reset_index()                  # gives a 'Date' or 'index' column depending on pandas version
    # The first column after reset_index() is the date; rename it safely:
    first_col = tmp.columns[0]
    tmp = tmp.rename(columns={first_col: "trade_dt"})

    # 3) Add ticker and select exactly the three columns in the right order
    tmp["ticker"] = ticker
    out = tmp[["ticker", "trade_dt", "adj_close"]].dropna(subset=["adj_close"])

    # 4) Write to DB using the engine (let pandas manage the connection/txn)
    out.to_sql("prices", engine, if_exists="append", index=False, method="multi")
    return len(out)

# ----------------------------
# main
# ----------------------------
def main():
    # Ensure schema exists
    init_schema()

    # Read tickers.txt (one per line)
    tickers_path = ROOT / "tickers.txt"
    if not tickers_path.exists():
        raise FileNotFoundError(
            f"Missing {tickers_path}. Create it with one symbol per line, e.g.\nAAPL\nMSFT\nSPY"
        )

    tickers = [t.strip() for t in tickers_path.read_text().splitlines() if t.strip()]
    if not tickers:
        raise ValueError("tickers.txt is empty.")

    total_rows = 0
    for t in tickers:
        print(f"[load] {t}")
        try:
            total_rows += load_prices(t)
        except Exception as e:
            print(f"[error] {t}: {e}")

    print(f"[done] loaded {total_rows} price rows into {DB_URL}")

if __name__ == "__main__":
    main()