# Serving self-test

- Proxy: the first 245,789 rows of `train.csv` season 2024, with the label removed.
- Returned rows/dtype: 245,789 / `float64`.
- NaN count: 0.
- Range: `[0.3304876599, 0.5829073796]`.
- Mean / population SD: `0.4602616605 / 0.0354375221`.
- `predict()` elapsed time: `1.7593s` (CSV loading excluded).
- Correlation with the forward-chained 2024 validation recipe: `0.79687654`.
- AST check on `predict.py` and `selftest.py`: no calls to `mean`, `groupby`,
  `value_counts`, `transform`, `rolling`, `expanding`, `cumsum`, or `rank`.
- Unknown pitcher IDs use the stored TrackMan global default and confidence 0.

All assertions in `selftest.py` passed on 2026-08-09.
