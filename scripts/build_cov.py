# scripts/build_cov.py
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DB_URL = os.getenv("DB_URL", "sqlite:///quant.db")
engine = create_engine(DB_URL, future=True)

def main():
    df = pd.read_sql("""
        SELECT ticker, trade_dt, log_ret
        FROM returns
    """, engine, parse_dates=["trade_dt"])

    # pivot to matrix: rows=dates, cols=tickers
    mat = df.pivot(index="trade_dt", columns="ticker", values="log_ret").dropna(how="any")
    cov = mat.cov()  # sample covariance
    corr = mat.corr()

    cov.to_csv("artifacts/covariance.csv")
    corr.to_csv("artifacts/correlation.csv")
    print("[done] wrote artifacts/covariance.csv and artifacts/correlation.csv")

if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    main()