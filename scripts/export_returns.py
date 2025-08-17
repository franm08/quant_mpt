import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")
EXPORT_CSV = "returns_matrix.csv"

engine = create_engine(DB_URL, future=True)

# Compute returns
with engine.begin() as con:
    con.execute(text(open("sql/02_returns.sql").read()))

# Pull into pandas
rets = pd.read_sql("SELECT * FROM returns", engine, parse_dates=["trade_dt"])
R = rets.pivot(index="trade_dt", columns="ticker", values="log_ret").dropna()
R.to_csv(EXPORT_CSV, index=True)

print(f"[done] returns matrix shape={R.shape}, saved -> {EXPORT_CSV}")