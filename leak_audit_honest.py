"""Independent leak-free audit — regenerates the honest number from scratch.

Does NOT trust the modified feature_engineering.elo_xg_diff. It keeps the 36
base features (no xG) and replaces the leaking xG feature with a strictly
backward-looking rolling xG, computed here from the raw match rows:

    for each team, the mean xG *for* and *against* over that team's PREVIOUS 6
    matches (strictly before the current fixture). Teams with no xG history yet
    get a 1.35 league-average prior (never 0.0).

Clean chronological split:
    test  = seasons 2023/24 and 2024/25  (760 matches)
    train = every match strictly before the test window
            -> naturally EXCLUDES 2025/26 (the split bug in train_model.py)
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from Features import process_matches
from feature_engineering import add_features
from train_model import BASE_FEATURES, TARGET_MAP, _add_season_match_counts

PRIOR = 1.35          # league-average xG prior for teams with no history
WINDOW = 6            # previous N matches
TEST_SEASONS = {"2023/24", "2024/25"}

ROLL_FEATURES = [
    "home_roll_xg_for", "home_roll_xg_against",
    "away_roll_xg_for", "away_roll_xg_against",
]
FEATURES = BASE_FEATURES + ROLL_FEATURES

PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)


def build_frame():
    df = pd.read_csv("Data/master.csv")
    df = process_matches(df)      # Elo (pre-match) + base engineered features
    df = add_features(df)

    df = df[df["FTR"].notna()].copy()
    df["target"] = df["FTR"].map(TARGET_MAP)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df = _add_season_match_counts(df)

    add_rolling_xg(df)   # my own backward-looking replacement
    return df


def add_rolling_xg(df):
    """Strictly backward-looking rolling mean xG for/against, per team, W=6."""
    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")

    xg_for = defaultdict(lambda: deque(maxlen=WINDOW))       # team -> last xG scored
    xg_against = defaultdict(lambda: deque(maxlen=WINDOW))   # team -> last xG conceded

    def mean_prior(dq):
        return float(np.mean(dq)) if len(dq) else PRIOR

    cols = {c: [] for c in ROLL_FEATURES}
    for row in df.itertuples(index=False):
        h, a = row.HomeTeam, row.AwayTeam
        # read state BEFORE this match (strictly previous matches only)
        cols["home_roll_xg_for"].append(mean_prior(xg_for[h]))
        cols["home_roll_xg_against"].append(mean_prior(xg_against[h]))
        cols["away_roll_xg_for"].append(mean_prior(xg_for[a]))
        cols["away_roll_xg_against"].append(mean_prior(xg_against[a]))

        # then update state with this match's xG (only if observed / non-null)
        hxg, axg = getattr(row, "home_xg"), getattr(row, "away_xg")
        if pd.notna(hxg) and pd.notna(axg):
            xg_for[h].append(float(hxg))
            xg_against[h].append(float(axg))
            xg_for[a].append(float(axg))
            xg_against[a].append(float(hxg))

    for c in ROLL_FEATURES:
        df[c] = cols[c]


def main():
    df = build_frame()

    # sanity: no NaN left in the features we use (base features get 0-filled as
    # in train_model; rolling xG is prior-filled, never NaN)
    df[BASE_FEATURES] = df[BASE_FEATURES].fillna(0.0)

    test_mask = df["Season"].astype(str).isin(TEST_SEASONS)
    earliest_test = df.loc[test_mask, "Date"].min()
    train_mask = df["Date"] < earliest_test        # clean: excludes 2025/26

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print("=== SPLIT ===")
    print(f"train rows: {len(train_df)}  ({train_df['Date'].min().date()} .. {train_df['Date'].max().date()})")
    print(f"test  rows: {len(test_df)}  ({test_df['Date'].min().date()} .. {test_df['Date'].max().date()})")
    print(f"test seasons: {sorted(test_df['Season'].astype(str).unique())}")
    print(f"2025/26 rows in train: {(train_df['Season'].astype(str)=='2025/26').sum()}  (must be 0)")
    print(f"latest train date < earliest test date: {train_df['Date'].max().date()} < {earliest_test.date()}")

    # --- rolling-xG population diagnostics (Step 2 checks, run regardless) ---
    post2014 = test_df   # test window is fully in the xG era
    at_prior = np.isclose(test_df["home_roll_xg_for"], PRIOR)
    print("\n=== ROLLING-xG DIAGNOSTICS (test set) ===")
    for c in ROLL_FEATURES:
        s = test_df[c]
        print(f"{c:22s} null={s.isna().sum():3d}  ==prior(1.35)={np.isclose(s,PRIOR).sum():3d}  "
              f"min={s.min():.3f} mean={s.mean():.3f} max={s.max():.3f}")

    x_train, y_train = train_df[FEATURES], train_df["target"]
    x_test, y_test = test_df[FEATURES], test_df["target"]

    model = XGBClassifier(**PARAMS)
    model.fit(x_train, y_train, verbose=False)
    probs = model.predict_proba(x_test)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs, labels=[0, 1, 2])

    # always-predict-home baseline (H == class 0)
    home_pred = np.zeros(len(y_test), dtype=int)
    base_acc = accuracy_score(y_test, home_pred)
    onehot = np.tile([1.0, 0.0, 0.0], (len(y_test), 1))
    base_ll = log_loss(y_test, onehot, labels=[0, 1, 2])   # degenerate one-hot (clipped)
    # constant training-prior predictor, for context
    prior_probs = np.tile(
        train_df["target"].value_counts(normalize=True).reindex([0, 1, 2]).values,
        (len(y_test), 1))
    prior_ll = log_loss(y_test, prior_probs, labels=[0, 1, 2])

    print("\n=== HONEST LEAK-FREE RESULT ===")
    print(f"test set size:                 {len(y_test)}")
    print(f"leak-free model  accuracy:     {acc*100:.2f}%")
    print(f"leak-free model  log loss:     {ll:.4f}")
    print(f"always-home      accuracy:     {base_acc*100:.2f}%")
    print(f"always-home      log loss:     {base_ll:.4f}   (degenerate one-hot [1,0,0], clipped)")
    print(f"(context) train-prior log loss:{prior_ll:.4f}   (constant class-distribution predictor)")
    print(f"test class balance H/D/A:      "
          f"{(y_test==0).mean()*100:.1f}% / {(y_test==1).mean()*100:.1f}% / {(y_test==2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
