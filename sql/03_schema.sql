SELECT ticker, COUNT(*) AS n_days
FROM returns
GROUP BY 1
ORDER BY n_days ASC;