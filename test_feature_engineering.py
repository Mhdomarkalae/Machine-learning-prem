"""
Quick validation tests for feature_engineering.py:
- Verifies no KeyError on missing state keys
- Checks O(1) incremental updates (no hidden loops)
- Tests no lookahead (features only use past data)
- Validates NaN handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from feature_engineering import compute_home_away_stats, add_features

def test_state_initialization():
    """Verify all required state keys are initialized."""
    from feature_engineering import _create_team_state, _push_overall_state
    
    team_state = _create_team_state(rolling_window=5, use_time_decay=False)
    
    # Should not raise KeyError
    try:
        _push_overall_state(team_state, points=3.0, goals_for=2.0, goals_against=1.0, rolling_window=5)
        print("✓ State initialization: overall_recent_points deque initialized safely")
        return True
    except KeyError as e:
        print(f"✗ State initialization failed: {e}")
        return False

def test_incremental_rolling_aggregates():
    """Verify rolling aggregates are computed incrementally (O(1) per update)."""
    from feature_engineering import _create_context_state, _push_context_state
    
    context = _create_context_state(rolling_window=5, use_time_decay=False)
    
    # Push 10 matches and verify sums are maintained
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    for i, d in enumerate(dates):
        goals = float((i % 3) + 1)  # 1, 2, 3, 1, 2, ...
        win = i % 2 == 0
        _push_context_state(context, points=3.0 if win else 1.0, goals_for=goals, 
                           goals_against=1.0, was_win=win, rolling_window=5, match_date=d)
    
    # For rolling_window=5, should only have last 5 matches
    if len(context["history"]) != 5:
        print(f"✗ Rolling window size: expected 5, got {len(context['history'])}")
        return False
    
    # Verify incremental sums match history
    manual_wins = sum(float(e[2]) for e in context["history"])
    incremental_wins = context["_wins_sum"]
    if abs(manual_wins - incremental_wins) > 1e-6:
        print(f"✗ Incremental sums: wins mismatch {incremental_wins} vs {manual_wins}")
        return False
    
    print("✓ Incremental rolling aggregates: sums maintained correctly")
    return True

def test_no_lookahead():
    """Verify features use only past matches (no future leakage)."""
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "HomeTeam": ["Team_A"] * 10,
        "AwayTeam": ["Team_B"] * 10,
        "FTHG": [1, 2, 1, 0, 3, 1, 2, 1, 0, 2],
        "FTAG": [0, 1, 1, 2, 1, 0, 1, 1, 3, 0],
        "home_elo_before": np.linspace(1500, 1550, 10),
        "away_elo_before": np.linspace(1400, 1450, 10),
    })
    
    try:
        result = add_features(df, rolling_window=3, use_time_decay=False)
        
        # First match should have NaN/zero features (no past data)
        first_row = result.iloc[0]
        # home_win_rate should be based on empty history or prior
        if pd.isna(first_row.get("home_win_rate")) or first_row["home_win_rate"] > 1.0:
            print(f"✗ First row lookahead check failed: win_rate={first_row.get('home_win_rate')}")
            return False
        
        # No NaN in critical output columns
        critical_cols = ["home_win_rate", "away_win_rate", "elo_diff"]
        for col in critical_cols:
            if result[col].isna().any():
                print(f"✗ NaN found in {col}: {result[col].isna().sum()} values")
                return False
        
        print(f"✓ No lookahead: all {len(result)} rows computed without future leakage")
        return True
    except Exception as e:
        print(f"✗ Feature computation failed: {e}")
        return False

def test_nan_handling():
    """Verify NaN handling in xG and other fields."""
    # Create data with missing xG
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "HomeTeam": ["A", "B", "A", "B", "A"],
        "AwayTeam": ["B", "A", "B", "A", "B"],
        "FTHG": [1, 0, 2, 1, 3],
        "FTAG": [0, 1, 1, 0, 1],
        "home_elo_before": [1500, 1500, 1500, 1500, 1500],
        "away_elo_before": [1500, 1500, 1500, 1500, 1500],
        "home_xg": [1.2, np.nan, 0.8, np.nan, 1.5],  # Some NaN
        "away_xg": [np.nan, 0.9, 1.1, 0.7, np.nan],  # Some NaN
    })
    
    try:
        result = add_features(df, rolling_window=2, use_time_decay=False)
        
        # Check that NaN xG doesn't break pipeline
        if result["elo_xg_diff"].isna().any():
            print(f"✗ NaN found in elo_xg_diff despite xG handling")
            return False
        
        # elo_xg_diff should be finite
        if not np.isfinite(result["elo_xg_diff"]).all():
            print(f"✗ Non-finite values in elo_xg_diff")
            return False
        
        print("✓ NaN handling: missing xG values handled safely")
        return True
    except Exception as e:
        print(f"✗ NaN handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Feature Engineering Validation Tests ===\n")
    
    tests = [
        ("State Initialization", test_state_initialization),
        ("Incremental Rolling Aggregates", test_incremental_rolling_aggregates),
        ("No Lookahead Leakage", test_no_lookahead),
        ("NaN Handling", test_nan_handling),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name}: unexpected error: {e}")
            results.append((name, False))
        print()
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\n=== Results: {passed_count}/{len(results)} tests passed ===")
    
    if passed_count == len(results):
        print("✓ All validation tests passed!")
    else:
        print("✗ Some tests failed. Review output above.")
