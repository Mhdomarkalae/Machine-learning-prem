"""Target-leakage audit for the match-outcome classifier.

Read-only: this script does not modify the pipeline. It rebuilds the exact
training frame train_model.py builds, then runs three tests against the
suspected feature (elo_xg_diff) plus a chronology check on the train/test split.

Run:  python leak_audit.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from Features import process_matches
from feature_engineering import add_features
from train_model import (
    BASE_FEATURES,
    FULL_FEATURES,
    TARGET_MAP,
    TEST_CUTOFF,
    _add_season_match_counts,
)

SUSPECT = "elo_xg_diff"
# The component is now a state-derived, rolling xG differential.  It is useful
# as a control in these tests, but cannot contain the fixture's own xG.
COMPONENT = "rolling_xg_component"
LEAK_FREE = [c for c in BASE_FEATURES]  # BASE_FEATURES excludes elo_xg_diff

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
    n_jobs=1,
)


def build_frame():
    """Reproduce train_model.main()'s frame construction exactly."""
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
    df[FULL_FEATURES] = df[FULL_FEATURES].fillna(0.0)

    # Isolate the state-derived xG component.
    df[COMPONENT] = df[SUSPECT] - df["elo_diff"]
    return df


def split(df):
    test_mask = df["Date"] >= TEST_CUTOFF
    return df[~test_mask].copy(), df[test_mask].copy()


def fit_score(train_df, test_df, features, seed_shift=0):
    """Train on `features` and return (log_loss, accuracy) on the test split."""
    params = dict(PARAMS)
    params["random_state"] = PARAMS["random_state"] + seed_shift
    x_train = train_df[features]
    x_test = test_df[features]
    y_train = train_df["target"]
    y_test = test_df["target"]
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0, 2: 1.0}, y=y_train)

    model = XGBClassifier(**params)
    model.fit(x_train, y_train, sample_weight=weights, verbose=False)
    probs = model.predict_proba(x_test)
    return (
        log_loss(y_test, probs, labels=[0, 1, 2]),
        accuracy_score(y_test, probs.argmax(axis=1)),
    )


def hdr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def test_a(train_df, test_df):
    hdr("TEST A - single feature vs. the whole leak-free feature set")
    results = {}
    results["leak-free set (%d feats)" % len(LEAK_FREE)] = fit_score(train_df, test_df, LEAK_FREE)
    results["%s ALONE" % SUSPECT] = fit_score(train_df, test_df, [SUSPECT])
    results["%s ALONE (xG part only)" % COMPONENT] = fit_score(train_df, test_df, [COMPONENT])
    results["elo_diff ALONE (control)"] = fit_score(train_df, test_df, ["elo_diff"])

    print(f"{'model':<38}{'log loss':>12}{'accuracy':>12}")
    for name, (ll, acc) in results.items():
        print(f"{name:<38}{ll:>12.4f}{acc * 100:>11.1f}%")

    ref_ll, ref_acc = results["leak-free set (%d feats)" % len(LEAK_FREE)]
    sus_ll, sus_acc = results["%s ALONE" % SUSPECT]
    failed = sus_ll < ref_ll and sus_acc > ref_acc
    print(f"\n  {SUSPECT} alone vs 36-feature leak-free set: "
          f"log loss {sus_ll:.4f} vs {ref_ll:.4f}, accuracy {sus_acc * 100:.1f}% vs {ref_acc * 100:.1f}%")
    print(f"  RESULT: {'FAIL - one feature beats the entire pre-match feature set' if failed else 'PASS'}")
    return failed, results


def test_b(train_df, test_df, results_a):
    hdr("TEST B - shuffle the rolling xG component across rows")
    full_ll, full_acc = fit_score(train_df, test_df, FULL_FEATURES)
    drop_ll, drop_acc = results_a["leak-free set (%d feats)" % len(LEAK_FREE)]

    # Permute the xG component within each split, preserving its marginal
    # distribution but destroying which match each value belongs to.
    rng = np.random.default_rng(0)
    shuffled_ll, shuffled_acc = [], []
    for trial in range(3):
        tr = train_df.copy()
        te = test_df.copy()
        for frame in (tr, te):
            perm = rng.permutation(len(frame))
            frame[SUSPECT] = frame["elo_diff"].to_numpy() + frame[COMPONENT].to_numpy()[perm]
        ll, acc = fit_score(tr, te, FULL_FEATURES, seed_shift=trial)
        shuffled_ll.append(ll)
        shuffled_acc.append(acc)
    sh_ll = float(np.mean(shuffled_ll))
    sh_acc = float(np.mean(shuffled_acc))

    print(f"{'model':<38}{'log loss':>12}{'accuracy':>12}")
    print(f"{'full (real ' + SUSPECT + ')':<38}{full_ll:>12.4f}{full_acc * 100:>11.1f}%")
    print(f"{'full (component shuffled, 3 runs)':<38}{sh_ll:>12.4f}{sh_acc * 100:>11.1f}%")
    print(f"{'feature dropped entirely':<38}{drop_ll:>12.4f}{drop_acc * 100:>11.1f}%")

    # How much of the real feature's edge over "dropped" survives shuffling?
    edge = drop_ll - full_ll          # log loss gained by having the feature
    survived = drop_ll - sh_ll        # log loss still gained after shuffling
    retained = survived / edge if edge != 0 else float("nan")
    # Do not diagnose row-alignment leakage from a numerically negligible edge.
    # The pre-fix leak was 0.1377 log loss; 0.01 is deliberately conservative.
    failed = edge > 0.01 and retained < 0.10
    print(f"\n  log-loss edge over dropping it: {edge:+.4f}")
    print(f"  edge retained after shuffling:  {survived:+.4f}  ({retained * 100:.1f}% of the edge)")
    print(f"  RESULT: {'FAIL - the signal was in WHICH match the value belonged to' if failed else 'PASS'}")
    return failed


def test_c(df):
    hdr("TEST C - sign agreement with the actual goal difference")
    decided = df[df["FTHG"] != df["FTAG"]].copy()
    gd_sign = np.sign(decided["FTHG"] - decided["FTAG"])

    def agreement(series):
        s = np.sign(series)
        nz = s != 0
        return float((s[nz] == gd_sign[nz]).mean()), int(nz.sum())

    sus_rate, sus_n = agreement(decided[COMPONENT])
    elo_rate, elo_n = agreement(decided["elo_diff"])

    # Restrict to rows where a past xG differential exists; this excludes the
    # common league prior applied before any historical xG coverage.
    covered = decided[decided[COMPONENT] != 0.0]
    cov_gd = np.sign(covered["FTHG"] - covered["FTAG"])
    cov_rate = float((np.sign(covered[COMPONENT]) == cov_gd).mean())
    cov_elo = float((np.sign(covered["elo_diff"]) == cov_gd).mean())

    print(f"{'feature':<44}{'agreement':>11}{'n':>9}")
    print(f"{COMPONENT + ' (xG part of ' + SUSPECT + ')':<44}{sus_rate * 100:>10.1f}%{sus_n:>9}")
    print(f"{'elo_diff (known-legitimate pre-match)':<44}{elo_rate * 100:>10.1f}%{elo_n:>9}")
    print(f"\n  restricted to matches with xG coverage (n={len(covered)}):")
    print(f"{'  ' + COMPONENT:<44}{cov_rate * 100:>10.1f}%")
    print(f"{'  elo_diff':<44}{cov_elo * 100:>10.1f}%")

    failed = cov_rate > cov_elo + 0.05
    print(f"\n  RESULT: {'FAIL - tracks the result far better than any pre-match feature' if failed else 'PASS'}")
    return failed


def test_split_chronology(df):
    hdr("SPLIT CHECK - explicit chronological date cutoff")
    train_df, test_df = split(df)
    earliest_test = test_df["Date"].min()
    offenders = train_df[train_df["Date"] >= earliest_test]
    print(f"  cutoff / earliest test-set date: {earliest_test.date()}")
    print(f"  latest training date:   {train_df['Date'].max().date()}")
    print(f"  training rows at/after the earliest test date: {len(offenders)}")
    if len(offenders):
        print("  offending seasons:")
        for season, n in offenders["Season"].value_counts().items():
            print(f"    {season}: {n} rows")
    failed = len(offenders) > 0
    print(f"\n  RESULT: {'FAIL - the split is not chronological' if failed else 'PASS'}")
    return failed


def main():
    df = build_frame()
    train_df, test_df = split(df)
    print(f"Rows: {len(df)}   train: {len(train_df)}   test: {len(test_df)}")
    print(f"State-derived feature: {SUSPECT}   (component = {SUSPECT} - elo_diff)")

    a_failed, results_a = test_a(train_df, test_df)
    b_failed = test_b(train_df, test_df, results_a)
    c_failed = test_c(df)
    split_failed = test_split_chronology(df)

    hdr("SUMMARY")
    for name, failed in [
        ("Test A (single-feature dominance)", a_failed),
        ("Test B (shuffle)", b_failed),
        ("Test C (sign agreement)", c_failed),
        ("Split chronology", split_failed),
    ]:
        print(f"  {name:<40}{'FAIL' if failed else 'PASS'}")


if __name__ == "__main__":
    main()
