# Serving self-test

Overall: **PASS**

## 245,789-row cold-start proxy

- elapsed: 2.677 seconds (contract: <60 seconds)
- output rows: 245,789
- dtype: `float64`
- NaN: 0
- range: [0.2826139737, 0.6272424341]
- prediction mean: 0.4631964004
- prediction SD: 0.0427455467

The proxy is the first 245,789 rows of train season 2024 in original order, with
`control_success` removed. Timing includes cold artifact loading and feature construction.

## Full train-2024 reference

- rows: 253,507
- warm elapsed: 2.643 seconds
- `raw_mean_2024`: 0.4627953387
- `raw_std_2024`: 0.0426324779
- correlation with R2 `team_pred_2024.csv`: 0.9320958214
- row order exact match: True

The serving model is trained through 2024, whereas the validation model is trained through
2023, so exact equality is neither expected nor required.

## Static and fallback checks

- forbidden AST calls in `predict.py`: 0
- forbidden grep matches in `predict.py`: 0
- unseen game/hand/top-bottom values and missing pitcher history: True
- custom pickle objects: none; all 18 models are LightGBM text boosters

No test-batch aggregation or post-hoc test calibration is performed.

## Test-harness correction

The first self-test attempt completed both large inference calls but failed while constructing
the synthetic unseen-value fixture: pandas 3.0 rejected a string sentinel in the integer-typed
hand column. The fixture now uses an unseen integer sentinel (`999`). No serving model,
artifact, feature, or prediction logic was changed because of this harness-only correction.
