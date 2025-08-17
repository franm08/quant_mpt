DELETE FROM returns;

WITH p AS (
  SELECT
    ticker,
    trade_dt,
    adj_close,
    LAG(adj_close) OVER (PARTITION BY ticker ORDER BY trade_dt) AS prev_close
  FROM prices
)
INSERT INTO returns (ticker, trade_dt, log_ret)
SELECT
  ticker,
  trade_dt,
  CASE 
    WHEN prev_close IS NULL OR prev_close <= 0 THEN NULL
    ELSE LN(adj_close / prev_close)
  END AS log_ret
FROM p
WHERE prev_close IS NOT NULL AND adj_close > 0 AND adj_close IS NOT NULL;