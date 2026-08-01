import json
import os
from collections import defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from xgboost import XGBClassifier

try:
    from elo import process_matches
except ImportError:
    from Features import process_matches

from feature_engineering import add_features

# Leak-free replacement for elo_xg_diff: strictly backward-looking rolling mean
# xG for/against over each team's previous 6 matches (see feature_engineering).
XG_FEATURES = [
    "home_roll_xg_for", "home_roll_xg_against",
    "away_roll_xg_for", "away_roll_xg_against",
]

BASE_FEATURES = [
    "home_elo_before", "away_elo_before", "elo_diff",
    "home_win_rate", "away_win_rate",
    "home_goal_avg_scored", "away_goal_avg_scored",
    "home_goal_avg_conceded", "away_goal_avg_conceded",
    "home_team_home_strength", "away_team_away_strength", "home_advantage_index",
    "home_form_points_avg", "away_form_points_avg",
    "home_form_momentum", "away_form_momentum",
    "elo_form_interaction", "elo_strength_interaction",
    "home_rest_days", "away_rest_days",
    "home_matches_played", "away_matches_played",
    "home_win_streak", "home_unbeaten_streak",
    "away_win_streak", "away_unbeaten_streak",
    "home_clean_sheet_rate", "away_clean_sheet_rate",
    "home_scoring_rate", "away_scoring_rate",
    "home_draw_tendency", "away_draw_tendency",
    "h2h_home_win_rate", "h2h_away_win_rate",
    "h2h_home_goal_avg", "h2h_away_goal_avg",
]

FULL_FEATURES = BASE_FEATURES + XG_FEATURES

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_LABELS = ["H", "D", "A"]
# Inclusive test boundary.  This is the date of the first held-out fixture;
# every training row is strictly earlier.
TEST_CUTOFF = pd.Timestamp("2023-08-11")


def _normalize_season(value):
    if pd.isna(value):
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 4:
        return digits[-4:]
    return digits or None


def _evaluate_model(name, model, x_test, y_test):
    probs = model.predict_proba(x_test)
    preds = probs.argmax(axis=1)
    ll = log_loss(y_test, probs, labels=[0, 1, 2])
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {l}" for l in TARGET_LABELS],
        columns=[f"Pred {l}" for l in TARGET_LABELS],
    )
    print(f"\n=== {name} ===")
    print(f"Log Loss: {ll:.4f}")
    print(f"Accuracy: {acc * 100:.1f}%")
    print("Confusion Matrix (H/D/A):")
    print(cm_df.to_string())
    return ll, acc


def _add_season_match_counts(df):
    counts = defaultdict(int)
    home_matches_played = []
    away_matches_played = []
    for row in df.itertuples(index=False):
        season = _normalize_season(getattr(row, "Season", None))
        home_key = (getattr(row, "HomeTeam"), season)
        away_key = (getattr(row, "AwayTeam"), season)
        home_matches_played.append(float(counts[home_key]))
        away_matches_played.append(float(counts[away_key]))
        counts[home_key] += 1
        counts[away_key] += 1
    df = df.copy()
    df["home_matches_played"] = home_matches_played
    df["away_matches_played"] = away_matches_played
    return df


def _market_baseline(test_df, y_test):
    """Normalized bookmaker implied probabilities, where all three odds exist."""
    odds = test_df[["B365H", "B365D", "B365A"]].apply(pd.to_numeric, errors="coerce")
    valid = (odds > 0.0).all(axis=1)
    implied = 1.0 / odds.loc[valid].to_numpy()
    probs = implied / implied.sum(axis=1, keepdims=True)
    targets = y_test.loc[valid]
    return log_loss(targets, probs, labels=[0, 1, 2]), accuracy_score(targets, probs.argmax(axis=1)), int(valid.sum())


def main():
    df = pd.read_csv("Data/master.csv")
    df = process_matches(df)
    df = add_features(df)

    df = df[df["FTR"].notna()].copy()
    df["target"] = df["FTR"].map(TARGET_MAP)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df = _add_season_match_counts(df)

    missing = [c for c in FULL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df[FULL_FEATURES] = df[FULL_FEATURES].fillna(0.0)

    # Clean chronological split: test = seasons 2023/24 and 2024/25 (760 matches);
    # train = every match strictly before the test window.  This deliberately
    # EXCLUDES 2025/26 (which falls after the test seasons) from both sides, so
    # the model never sees the test seasons or anything later during training.
    test_mask = df["Season"].astype(str).isin({"2023/24", "2024/25"})
    earliest_test = df.loc[test_mask, "Date"].min()
    train_mask = df["Date"] < earliest_test

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Excluded (2025/26, after test window): {int((~train_mask & ~test_mask).sum())}")
    print(f"Train dates: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"Test dates:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

    x_train_base = train_df[BASE_FEATURES]
    x_test_base = test_df[BASE_FEATURES]
    x_train_full = train_df[FULL_FEATURES]
    x_test_full = test_df[FULL_FEATURES]
    y_train = train_df["target"]
    y_test = test_df["target"]

    params = dict(
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

    model_full = XGBClassifier(**params)
    model_base = XGBClassifier(**params)

    model_full.fit(x_train_full, y_train,
                   eval_set=[(x_test_full, y_test)], verbose=False)
    model_base.fit(x_train_base, y_train,
                   eval_set=[(x_test_base, y_test)], verbose=False)

    base_ll, base_acc = _evaluate_model("Base Model (no xG)", model_base, x_test_base, y_test)
    full_ll, full_acc = _evaluate_model("Full Model (with xG)", model_full, x_test_full, y_test)

    train_dist = y_train.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0).values
    naive_probs = [train_dist] * len(y_test)
    naive_ll = log_loss(y_test, naive_probs, labels=[0, 1, 2])
    naive_acc = accuracy_score(y_test, [int(train_dist.argmax())] * len(y_test))
    market_ll, market_acc, market_n = _market_baseline(test_df, y_test)

    print("\n=== FEATURE IMPORTANCES (Full Model) ===")
    importance_df = pd.DataFrame({
        "feature": FULL_FEATURES,
        "importance": model_full.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(importance_df.to_string(index=False))

    print("\n=== FINAL MODEL COMPARISON ===")
    print("                  Log Loss    Accuracy")
    print(f"Base (no xG):    {base_ll:.4f}      {base_acc * 100:.1f}%")
    print(f"Full (with xG):  {full_ll:.4f}      {full_acc * 100:.1f}%")
    print(f"Market baseline: {market_ll:.4f}      {market_acc * 100:.1f}%  (n={market_n})")
    print(f"Naive baseline:  {naive_ll:.4f}      {naive_acc * 100:.1f}%")
    print(f"xG improvement:  {base_ll - full_ll:+.4f} log loss  / {((full_acc - base_acc) * 100):+.1f}% accuracy")

    os.makedirs("model", exist_ok=True)
    model_full.get_booster().save_model("model/xgb_full.json")
    model_base.get_booster().save_model("model/xgb_base.json")

    with open("model/feature_cols_full.json", "w", encoding="utf-8") as f:
        json.dump(FULL_FEATURES, f)
    with open("model/feature_cols_base.json", "w", encoding="utf-8") as f:
        json.dump(BASE_FEATURES, f)

    print("\nSaved model artifacts to model/")


if __name__ == "__main__":
    main()
