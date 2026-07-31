# Pre-fix ("before") measurements — recorded 2026-07-22

Preserved because `train_model.log` contains no metrics (that run crashed in
RandomizedSearchCV before printing the comparison table). These numbers are the
only record of the leaking model's performance.

Setup: `Data/master.csv` (9770 rows), split by `TEST_SEASONS = {"2324","2425"}`
(train 9010 / test 760), params exactly as in `train_model.py`.

| model | features | log loss | accuracy |
|---|---|---|---|
| Full (leaking) | 37 incl. `elo_xg_diff` | 0.8948 | 56.2% |
| Base (leak-free) | 36 | 1.0325 | 45.1% |
| `elo_xg_diff` alone | 1 | 0.9398 | 51.6% |
| xG component alone (`elo_xg_diff - elo_diff`) | 1 | 0.8850 | 56.1% |
| `elo_diff` alone (control) | 1 | 1.0576 | 41.1% |
| Full, xG component shuffled (mean of 3) | 37 | 1.0292 | 45.6% |

Measured inflation from the leak: **-0.1377 log loss, +11.1pp accuracy**.

Sign agreement with actual goal difference (decided matches with xG coverage, n=3410):
xG component 79.5% vs `elo_diff` 68.0%.

Both the train/test split (360 training rows from season 2025/26 dated after the
test window) and the leak are still present at the time of this recording.
