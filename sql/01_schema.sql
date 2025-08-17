CREATE TABLE IF NOT EXISTS assets (
  ticker TEXT PRIMARY KEY,
  name   TEXT
);

CREATE TABLE IF NOT EXISTS prices (
  ticker    TEXT NOT NULL,
  trade_dt  DATE NOT NULL,
  adj_close REAL NOT NULL,
  PRIMARY KEY (ticker, trade_dt)
);

CREATE TABLE IF NOT EXISTS returns (
  ticker   TEXT NOT NULL,
  trade_dt DATE NOT NULL,
  log_ret  REAL NOT NULL,
  PRIMARY KEY (ticker, trade_dt)
);