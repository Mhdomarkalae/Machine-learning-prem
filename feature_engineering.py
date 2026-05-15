from collections import defaultdict, deque
import math

import pandas as pd


REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "home_elo_before", "away_elo_before"}
DEFAULT_ROLLING_WINDOW = 5
RECENCY_HALFLIFE_DAYS = 180.0
STRENGTH_TANH_SCALE = 2.0
FORM_TANH_SCALE = 3.0
XG_ELO_SCALE = 50.0
XG_DIFF_SCALE = 2.0
SMOOTHING_STRENGTH = 6.0
GOAL_FOR_PRIOR = 1.35
GOAL_AGAINST_PRIOR = 1.15


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_div(numerator, denominator, default=0.0):
    denominator = float(denominator)
    if denominator == 0.0:
        return float(default)
    return float(numerator) / denominator


def _match_points(home_goals, away_goals):
    if home_goals > away_goals:
        return 3.0, 0.0
    if home_goals < away_goals:
        return 0.0, 3.0
    return 1.0, 1.0


def _create_context_state(rolling_window, use_time_decay=False):
    # Two mutually exclusive modes: rolling window or time-decay (EWMA-like)
    if use_time_decay:
        return {
            "mode": "decay",
            "matches": 0.0,
            "wins": 0.0,
            "draws": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "clean_sheets": 0.0,
            "scored": 0.0,
            "points": 0.0,
            "last_date": None,
        }

    # Rolling mode: track history with O(1) incremental sums (avoid O(n) recomputation)
    history_maxlen = rolling_window if rolling_window and rolling_window > 0 else None
    trend_maxlen = max((rolling_window or 0) * 2, 10)
    return {
        "mode": "rolling",
        "matches": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "clean_sheets": 0.0,
        "scored": 0.0,
        "points": 0.0,
        "history": deque(maxlen=history_maxlen),
        "trend_history": deque(maxlen=trend_maxlen),
        "_wins_sum": 0.0,  # Track cumulative wins to avoid O(n) recomputation
        "_goals_for_sum": 0.0,
        "_goals_against_sum": 0.0,
        "_points_sum": 0.0,
        "_clean_sheets_sum": 0.0,
        "_scored_sum": 0.0,
        "_draws_sum": 0.0,
    }


def _create_team_state(rolling_window, use_time_decay=False):
    # Initialize overall_recent_points deque for rolling window momentum calculations
    recent_points_maxlen = max((rolling_window or 0) * 2, 10)
    return {
        "overall_matches": 0.0,
        "overall_points": 0.0,
        "overall_goals_for": 0.0,
        "overall_goals_against": 0.0,
        "overall_recent_points": deque(maxlen=recent_points_maxlen),
        "all": _create_context_state(rolling_window, use_time_decay=use_time_decay),
        "home": _create_context_state(rolling_window, use_time_decay=use_time_decay),
        "away": _create_context_state(rolling_window, use_time_decay=use_time_decay),
    }


def _push_context_state(context_state, points, goals_for, goals_against, was_win, rolling_window, match_date=None):
    # Normalize types
    points = float(points)
    goals_for = float(goals_for)
    goals_against = float(goals_against)
    was_win = 1.0 if was_win else 0.0

    if context_state.get("mode") == "decay":
        # Exponential decay of stored aggregates since last_date, then add current match
        current_date = pd.to_datetime(match_date, errors="coerce") if match_date is not None else pd.NaT
        last_date = context_state.get("last_date")
        if last_date is not None and pd.notna(last_date) and pd.notna(current_date):
            days = max((current_date - last_date).days, 0)
            factor = 0.5 ** (days / max(RECENCY_HALFLIFE_DAYS, 1e-9))
            context_state["matches"] *= factor
            context_state["wins"] *= factor
            context_state["draws"] *= factor
            context_state["goals_for"] *= factor
            context_state["goals_against"] *= factor
            context_state["clean_sheets"] *= factor
            context_state["scored"] *= factor
            context_state["points"] *= factor

        context_state["matches"] += 1.0
        context_state["wins"] += was_win
        # draw flag: 1.0 if points == 1.0 else 0.0
        draw_flag = 1.0 if points == 1.0 else 0.0
        context_state["draws"] += draw_flag
        context_state["goals_for"] += goals_for
        context_state["goals_against"] += goals_against
        # clean sheet flag and scored flag
        clean_flag = 1.0 if goals_against == 0.0 else 0.0
        scored_flag = 1.0 if goals_for >= 1.0 else 0.0
        context_state["clean_sheets"] += clean_flag
        context_state["scored"] += scored_flag
        context_state["points"] += points
        context_state["last_date"] = current_date
        return

    # Rolling mode: maintain O(1) incremental sums instead of O(n) recomputation
    entry = (match_date, points, int(was_win), goals_for, goals_against)
    
    # If history is at maxlen, the oldest entry will be discarded; update running sums first
    if "history" in context_state and len(context_state["history"]) == context_state["history"].maxlen:
        old_entry = context_state["history"][0]  # Will be popped by append
        context_state["_wins_sum"] = max(0.0, context_state["_wins_sum"] - float(old_entry[2]))
        context_state["_goals_for_sum"] = max(0.0, context_state["_goals_for_sum"] - float(old_entry[3]))
        context_state["_goals_against_sum"] = max(0.0, context_state["_goals_against_sum"] - float(old_entry[4]))
        context_state["_points_sum"] = max(0.0, context_state["_points_sum"] - float(old_entry[1]))
        # derive old flags from history tuple: old_entry = (match_date, points, was_win_int, goals_for, goals_against)
        old_points = float(old_entry[1])
        old_goals_for = float(old_entry[3])
        old_goals_against = float(old_entry[4])
        old_clean = 1.0 if old_goals_against == 0.0 else 0.0
        old_scored = 1.0 if old_goals_for >= 1.0 else 0.0
        old_draw = 1.0 if old_points == 1.0 else 0.0
        context_state["_clean_sheets_sum"] = max(0.0, context_state["_clean_sheets_sum"] - old_clean)
        context_state["_scored_sum"] = max(0.0, context_state["_scored_sum"] - old_scored)
        context_state["_draws_sum"] = max(0.0, context_state["_draws_sum"] - old_draw)
    
    # Append to history (bounded deque automatically discards oldest if full)
    context_state.setdefault("history", deque()).append(entry)
    context_state.setdefault("trend_history", deque()).append(points)
    
    # Update running sums incrementally (O(1) per match, not O(n))
    context_state["_wins_sum"] += was_win
    context_state["_goals_for_sum"] += goals_for
    context_state["_goals_against_sum"] += goals_against
    context_state["_points_sum"] += points
    # update new flags sums
    clean_flag = 1.0 if goals_against == 0.0 else 0.0
    scored_flag = 1.0 if goals_for >= 1.0 else 0.0
    draw_flag = 1.0 if points == 1.0 else 0.0
    context_state["_clean_sheets_sum"] += clean_flag
    context_state["_scored_sum"] += scored_flag
    context_state["_draws_sum"] += draw_flag
    
    # Maintain aggregate counters from running sums
    context_state["matches"] = float(len(context_state["history"]))
    context_state["wins"] = float(context_state["_wins_sum"])
    context_state["draws"] = float(context_state.get("_draws_sum", 0.0))
    context_state["goals_for"] = float(context_state["_goals_for_sum"])
    context_state["goals_against"] = float(context_state["_goals_against_sum"])
    context_state["points"] = float(context_state["_points_sum"])
    context_state["clean_sheets"] = float(context_state.get("_clean_sheets_sum", 0.0))
    context_state["scored"] = float(context_state.get("_scored_sum", 0.0))


def _push_overall_state(team_state, points, goals_for, goals_against, rolling_window):
    team_state["overall_matches"] += 1
    team_state["overall_points"] += float(points)
    team_state["overall_goals_for"] += float(goals_for)
    team_state["overall_goals_against"] += float(goals_against)

    if rolling_window and rolling_window > 0:
        team_state["overall_recent_points"].append(float(points))


def _weighted_context_snapshot(context_state, shared_state, current_date, rolling_window):
    # Replace abrupt fallback with blending of context snapshot and overall snapshot
    # use_time_decay is determined by context_state["mode"], not as a parameter
    context_snapshot = None
    if context_state.get("mode") == "decay":
        # compute decay snapshot directly from aggregated fields
        raw_matches = max(float(context_state.get("matches", 0.0)), 0.0)
        win_rate = (float(context_state.get("wins", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (raw_matches + SMOOTHING_STRENGTH)
        goal_avg_scored = (float(context_state.get("goals_for", 0.0)) + GOAL_FOR_PRIOR * SMOOTHING_STRENGTH) / (raw_matches + SMOOTHING_STRENGTH)
        goal_avg_conceded = (float(context_state.get("goals_against", 0.0)) + GOAL_AGAINST_PRIOR * SMOOTHING_STRENGTH) / (raw_matches + SMOOTHING_STRENGTH)
        clean_sheet_rate = (float(context_state.get("clean_sheets", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (raw_matches + SMOOTHING_STRENGTH)
        scoring_rate = (float(context_state.get("scored", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (raw_matches + SMOOTHING_STRENGTH)
        context_snapshot = {
            "win_rate": float(win_rate),
            "goal_avg_scored": float(goal_avg_scored),
            "goal_avg_conceded": float(goal_avg_conceded),
            "clean_sheet_rate": float(clean_sheet_rate),
            "scoring_rate": float(scoring_rate),
            "matches": float(raw_matches),
        }
    else:
        # O(1) rolling window snapshot using pre-computed sums
        matches = float(context_state.get("matches", 0.0))
        wins = float(context_state.get("wins", 0.0))
        goals_for = float(context_state.get("goals_for", 0.0))
        goals_against = float(context_state.get("goals_against", 0.0))

        win_rate = (wins + 0.5 * SMOOTHING_STRENGTH) / (matches + SMOOTHING_STRENGTH)
        goal_avg_scored = (goals_for + GOAL_FOR_PRIOR * SMOOTHING_STRENGTH) / (matches + SMOOTHING_STRENGTH)
        goal_avg_conceded = (goals_against + GOAL_AGAINST_PRIOR * SMOOTHING_STRENGTH) / (matches + SMOOTHING_STRENGTH)
        clean_sheet_rate = (float(context_state.get("clean_sheets", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (matches + SMOOTHING_STRENGTH)
        scoring_rate = (float(context_state.get("scored", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (matches + SMOOTHING_STRENGTH)

        context_snapshot = {
            "win_rate": float(win_rate),
            "goal_avg_scored": float(goal_avg_scored),
            "goal_avg_conceded": float(goal_avg_conceded),
            "clean_sheet_rate": float(clean_sheet_rate),
            "scoring_rate": float(scoring_rate),
            "matches": float(matches),
        }

    # compute overall snapshot from shared_state
    # Explicitly guard: only use rolling branch if shared_state is in rolling mode
    if shared_state.get("mode") == "decay":
        overall_matches = max(float(shared_state.get("matches", 0.0)), 0.0)
        overall_win_rate = (float(shared_state.get("wins", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (overall_matches + SMOOTHING_STRENGTH)
        overall_goal_avg_scored = (float(shared_state.get("goals_for", 0.0)) + GOAL_FOR_PRIOR * SMOOTHING_STRENGTH) / (overall_matches + SMOOTHING_STRENGTH)
        overall_goal_avg_conceded = (float(shared_state.get("goals_against", 0.0)) + GOAL_AGAINST_PRIOR * SMOOTHING_STRENGTH) / (overall_matches + SMOOTHING_STRENGTH)
    elif shared_state.get("mode") == "rolling":
        # O(1) overall snapshot using pre-computed sums from rolling state
        overall_matches = float(shared_state.get("matches", 0.0))
        overall_win_rate = (float(shared_state.get("wins", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (overall_matches + SMOOTHING_STRENGTH)
        overall_goal_avg_scored = (float(shared_state.get("goals_for", 0.0)) + GOAL_FOR_PRIOR * SMOOTHING_STRENGTH) / (overall_matches + SMOOTHING_STRENGTH)
        overall_goal_avg_conceded = (float(shared_state.get("goals_against", 0.0)) + GOAL_AGAINST_PRIOR * SMOOTHING_STRENGTH) / (overall_matches + SMOOTHING_STRENGTH)
    else:
        # Fallback if mode is not set (defensive)
        overall_matches = 0.0
        overall_win_rate = 0.5
        overall_goal_avg_scored = 0.0
        overall_goal_avg_conceded = 0.0

    overall_snapshot = {
        "win_rate": float(overall_win_rate),
        "goal_avg_scored": float(overall_goal_avg_scored),
        "goal_avg_conceded": float(overall_goal_avg_conceded),
        "matches": float(overall_matches),
    }

    # Blend using match-count driven alpha (shrinkage)
    context_n = max(0.0, float(context_snapshot.get("matches", 0.0)))
    alpha = context_n / (context_n + SMOOTHING_STRENGTH)
    blended = {
        "win_rate": float(alpha * context_snapshot["win_rate"] + (1 - alpha) * overall_snapshot["win_rate"]),
        "goal_avg_scored": float(alpha * context_snapshot["goal_avg_scored"] + (1 - alpha) * overall_snapshot["goal_avg_scored"]),
        "goal_avg_conceded": float(alpha * context_snapshot["goal_avg_conceded"] + (1 - alpha) * overall_snapshot["goal_avg_conceded"]),
        "clean_sheet_rate": float(context_snapshot.get("clean_sheet_rate", 0.0)),
        "scoring_rate": float(context_snapshot.get("scoring_rate", 0.0)),
        "matches": float(context_snapshot.get("matches", 0.0)),
    }
    return blended


def _overall_form_snapshot(team_state, current_date, rolling_window):
    # Provide a single, smoothed form snapshot. Mode (decay vs rolling) is determined by team_state["all"].get("mode")
    if team_state["all"].get("mode") == "decay":
        cs = team_state["all"]
        raw_matches = max(float(cs.get("matches", 0.0)), 0.0)
        points = float(cs.get("points", 0.0))
        form_points_avg = _safe_div(points, raw_matches, default=0.0)
        draws = float(cs.get("draws", 0.0))
        draw_tendency = (draws + 0.5 * SMOOTHING_STRENGTH) / (raw_matches + SMOOTHING_STRENGTH)
        momentum = form_points_avg - 1.0
        return {"form_points_avg": float(form_points_avg), "form_momentum": float(momentum), "draw_tendency": float(draw_tendency)}

    # Rolling mode: compute form from all available history, momentum from recent vs. previous halves
    all_context = team_state.get("all", {})
    if not all_context.get("history"):
        return {"form_points_avg": 0.0, "form_momentum": 0.0}

    form_points_avg = _safe_div(all_context.get("points", 0.0), all_context.get("matches", 0.0))
    draws = float(all_context.get("draws", 0.0))
    draw_tendency = (draws + 0.5 * SMOOTHING_STRENGTH) / (all_context.get("matches", 0.0) + SMOOTHING_STRENGTH)

    # Momentum: use last rolling_window*2 matches (or all if fewer) split into recent and previous
    all_history = list(all_context.get("history", []))
    if rolling_window and rolling_window > 0:
        momentum_window = all_history[-(rolling_window * 2):]
    else:
        momentum_window = all_history

    if len(momentum_window) < 2:
        momentum = form_points_avg - 1.0
    else:
        # Split into two halves: recent and previous
        split_idx = len(momentum_window) // 2
        recent_points = [float(e[1]) for e in momentum_window[split_idx:]]
        prev_points = [float(e[1]) for e in momentum_window[:split_idx]]
        recent_avg = sum(recent_points) / float(len(recent_points)) if recent_points else form_points_avg
        prev_avg = sum(prev_points) / float(len(prev_points)) if prev_points else form_points_avg
        momentum = recent_avg - prev_avg

    return {"form_points_avg": float(form_points_avg), "form_momentum": float(momentum), "draw_tendency": float(draw_tendency)}


def _normalize_strength(value, scale=STRENGTH_TANH_SCALE):
    value = _safe_float(value, 0.0)
    scale = max(scale, 1e-9)
    return value / (scale + abs(value))


def _stable_form_interaction(elo_diff, home_form_points_avg, away_form_points_avg):
    elo_component = math.tanh(_safe_float(elo_diff, 0.0) / 400.0)
    form_component = math.tanh((_safe_float(home_form_points_avg, 0.0) - _safe_float(away_form_points_avg, 0.0)) / FORM_TANH_SCALE)
    return elo_component * form_component


def _compute_rest_days(last_match_date, team, match_date):
    """Return days since team's last match or -1.0 if unknown."""
    last_date = last_match_date.get(team)
    if pd.notna(match_date) and last_date is not None and pd.notna(last_date):
        return float((match_date - last_date).days)
    return float(-1.0)


def _get_h2h_snapshot(h2h_states, pair_key):
    """Read H2H state for pair_key and return smoothed snapshot dict.

    Returns keys: h2h_home_win_rate, h2h_away_win_rate, h2h_home_goal_avg,
    h2h_away_goal_avg, h2h_matches
    """
    h2h = h2h_states.get(pair_key, {"matches": 0.0, "home_wins": 0.0, "away_wins": 0.0, "home_goals": 0.0, "away_goals": 0.0})
    h2h_matches = float(h2h.get("matches", 0.0))
    h2h_home_win_rate = (float(h2h.get("home_wins", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (h2h_matches + SMOOTHING_STRENGTH)
    h2h_away_win_rate = (float(h2h.get("away_wins", 0.0)) + 0.5 * SMOOTHING_STRENGTH) / (h2h_matches + SMOOTHING_STRENGTH)
    h2h_home_goal_avg = (float(h2h.get("home_goals", 0.0)) + GOAL_FOR_PRIOR * SMOOTHING_STRENGTH) / (h2h_matches + SMOOTHING_STRENGTH)
    h2h_away_goal_avg = (float(h2h.get("away_goals", 0.0)) + GOAL_FOR_PRIOR * SMOOTHING_STRENGTH) / (h2h_matches + SMOOTHING_STRENGTH)
    return {
        "h2h_home_win_rate": h2h_home_win_rate,
        "h2h_away_win_rate": h2h_away_win_rate,
        "h2h_home_goal_avg": h2h_home_goal_avg,
        "h2h_away_goal_avg": h2h_away_goal_avg,
        "h2h_matches": h2h_matches,
    }


def _get_streak_snapshot(streak_states, team):
    """Return current streak snapshot for team: win_streak and unbeaten_streak."""
    s = streak_states.get(team, {"win_streak": 0.0, "unbeaten_streak": 0.0})
    return {"win_streak": float(s.get("win_streak", 0.0)), "unbeaten_streak": float(s.get("unbeaten_streak", 0.0))}


def _push_streak_state(streak_states, team, points):
    """Update streak_states for team given points (3 win,1 draw,0 loss)."""
    st = streak_states[team]
    if points == 3.0:
        st["win_streak"] = float(st.get("win_streak", 0.0)) + 1.0
        st["unbeaten_streak"] = float(st.get("unbeaten_streak", 0.0)) + 1.0
    elif points == 1.0:
        st["win_streak"] = 0.0
        st["unbeaten_streak"] = float(st.get("unbeaten_streak", 0.0)) + 1.0
    else:
        st["win_streak"] = 0.0
        st["unbeaten_streak"] = 0.0


def _prepare_working_frame(df):
    working_df = df.copy()
    working_df["_feature_order"] = range(len(working_df))

    if "Date" in working_df.columns:
        working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")
        working_df = working_df.sort_values(["Date", "_feature_order"], kind="stable")

    return working_df


def compute_home_away_stats(
    df,
    rolling_window=DEFAULT_ROLLING_WINDOW,
    include_momentum=True,
    preserve_original_order=False,
    normalize_strengths=True,
    use_time_decay=True,
):
    if df is None or df.empty:
        return pd.DataFrame(df).copy()

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")

    working_df = _prepare_working_frame(df)

    if "home_xg" not in working_df.columns:
        working_df["home_xg"] = 0.0
    if "away_xg" not in working_df.columns:
        working_df["away_xg"] = 0.0
    if "xg_diff" not in working_df.columns:
        working_df["xg_diff"] = working_df["home_xg"].fillna(0.0) - working_df["away_xg"].fillna(0.0)

    working_df["home_xg"] = pd.to_numeric(working_df["home_xg"], errors="coerce")
    working_df["away_xg"] = pd.to_numeric(working_df["away_xg"], errors="coerce")
    working_df["xg_diff"] = pd.to_numeric(working_df["xg_diff"], errors="coerce")

    team_states = defaultdict(lambda: _create_team_state(rolling_window, use_time_decay=use_time_decay))
    # Track last played date for each team across all matches (home & away)
    last_match_date = defaultdict(lambda: None)
    # Head-to-head states keyed by frozenset({home, away})
    h2h_states = defaultdict(lambda: {
        "matches": 0.0,
        "home_wins": 0.0,
        "away_wins": 0.0,
        "home_goals": 0.0,
        "away_goals": 0.0,
    })
    # Per-team streak states (consecutive wins and unbeaten streaks)
    streak_states = defaultdict(lambda: {"win_streak": 0.0, "unbeaten_streak": 0.0})
    output_rows = []

    for row in working_df.itertuples(index=False):
        match_date = pd.to_datetime(getattr(row, "Date", pd.NaT), errors="coerce")
        home_team = row.HomeTeam
        away_team = row.AwayTeam
        home_goals = _safe_float(row.FTHG, 0.0)
        away_goals = _safe_float(row.FTAG, 0.0)
        home_elo_before = _safe_float(getattr(row, "home_elo_before", 0.0), 0.0)
        away_elo_before = _safe_float(getattr(row, "away_elo_before", 0.0), 0.0)
        home_xg = getattr(row, "home_xg", 0.0)
        away_xg = getattr(row, "away_xg", 0.0)
        xg_diff = getattr(row, "xg_diff", 0.0)

        if pd.isna(home_xg):
            home_xg = 0.0
        if pd.isna(away_xg):
            away_xg = 0.0
        if pd.isna(xg_diff):
            xg_diff = 0.0

        home_state = team_states[home_team]
        away_state = team_states[away_team]

        # Compute rest days since each team's last match (sentinel -1.0 if unknown)
        home_rest_days = _compute_rest_days(last_match_date, home_team, match_date)
        away_rest_days = _compute_rest_days(last_match_date, away_team, match_date)

        # Compute head-to-head (H2H) smoothed features for this pairing
        pair_key = frozenset({home_team, away_team})
        h2h_snapshot = _get_h2h_snapshot(h2h_states, pair_key)
        h2h_home_win_rate = h2h_snapshot["h2h_home_win_rate"]
        h2h_away_win_rate = h2h_snapshot["h2h_away_win_rate"]
        h2h_home_goal_avg = h2h_snapshot["h2h_home_goal_avg"]
        h2h_away_goal_avg = h2h_snapshot["h2h_away_goal_avg"]
        h2h_matches_out = h2h_snapshot["h2h_matches"]

        # Read current streaks for both teams (raw counts, no smoothing)
        home_snap = _get_streak_snapshot(streak_states, home_team)
        away_snap = _get_streak_snapshot(streak_states, away_team)
        home_win_streak = home_snap["win_streak"]
        home_unbeaten_streak = home_snap["unbeaten_streak"]
        away_win_streak = away_snap["win_streak"]
        away_unbeaten_streak = away_snap["unbeaten_streak"]

        home_home_snapshot = _weighted_context_snapshot(
            home_state["home"],
            home_state["all"],
            match_date,
            rolling_window,
        )
        away_away_snapshot = _weighted_context_snapshot(
            away_state["away"],
            away_state["all"],
            match_date,
            rolling_window,
        )
        home_form_snapshot = _overall_form_snapshot(home_state, match_date, rolling_window if include_momentum else None)
        away_form_snapshot = _overall_form_snapshot(away_state, match_date, rolling_window if include_momentum else None)

        home_goal_diff_profile = home_home_snapshot["goal_avg_scored"] - home_home_snapshot["goal_avg_conceded"]
        away_goal_diff_profile = away_away_snapshot["goal_avg_scored"] - away_away_snapshot["goal_avg_conceded"]

        if normalize_strengths:
            home_team_home_strength = _normalize_strength(home_goal_diff_profile, scale=STRENGTH_TANH_SCALE * 2.0)
            away_team_away_strength = _normalize_strength(away_goal_diff_profile, scale=STRENGTH_TANH_SCALE * 2.0)
            home_advantage_index = _normalize_strength(home_goal_diff_profile - away_goal_diff_profile, scale=STRENGTH_TANH_SCALE * 2.0)
        else:
            home_team_home_strength = float(home_goal_diff_profile)
            away_team_away_strength = float(away_goal_diff_profile)
            home_advantage_index = float(home_goal_diff_profile - away_goal_diff_profile)

        elo_diff = home_elo_before - away_elo_before

        home_form_points_avg = home_form_snapshot["form_points_avg"]
        away_form_points_avg = away_form_snapshot["form_points_avg"]
        home_form_momentum = home_form_snapshot["form_momentum"] if include_momentum else 0.0
        away_form_momentum = away_form_snapshot["form_momentum"] if include_momentum else 0.0
        elo_form_interaction = _stable_form_interaction(elo_diff, home_form_points_avg, away_form_points_avg)
        elo_xg_diff = elo_diff + (xg_diff * XG_ELO_SCALE)
        elo_strength_interaction = math.tanh(elo_diff / 400.0) * home_advantage_index

        row_output = row._asdict()
        row_output.update(
            {
                "home_win_rate": home_home_snapshot["win_rate"],
                "away_win_rate": away_away_snapshot["win_rate"],
                "home_goal_avg_scored": home_home_snapshot["goal_avg_scored"],
                "away_goal_avg_scored": away_away_snapshot["goal_avg_scored"],
                "home_goal_avg_conceded": home_home_snapshot["goal_avg_conceded"],
                "away_goal_avg_conceded": away_away_snapshot["goal_avg_conceded"],
                "home_team_home_strength": float(home_team_home_strength),
                "away_team_away_strength": float(away_team_away_strength),
                "home_advantage_index": float(home_advantage_index),
                "elo_diff": float(elo_diff),
                "elo_xg_diff": float(elo_xg_diff),
                "home_form_points_avg": float(home_form_points_avg),
                "away_form_points_avg": float(away_form_points_avg),
                "home_form_momentum": float(home_form_momentum),
                "away_form_momentum": float(away_form_momentum),
                "elo_form_interaction": float(elo_form_interaction),
                "elo_strength_interaction": float(elo_strength_interaction),
                "home_rest_days": float(home_rest_days),
                "away_rest_days": float(away_rest_days),
                "home_win_streak": float(home_win_streak),
                "home_unbeaten_streak": float(home_unbeaten_streak),
                "away_win_streak": float(away_win_streak),
                "away_unbeaten_streak": float(away_unbeaten_streak),
                "home_clean_sheet_rate": float(home_home_snapshot.get("clean_sheet_rate", 0.0)),
                "away_clean_sheet_rate": float(away_away_snapshot.get("clean_sheet_rate", 0.0)),
                "home_scoring_rate": float(home_home_snapshot.get("scoring_rate", 0.0)),
                "away_scoring_rate": float(away_away_snapshot.get("scoring_rate", 0.0)),
                "home_draw_tendency": float(home_form_snapshot.get("draw_tendency", 0.0)),
                "away_draw_tendency": float(away_form_snapshot.get("draw_tendency", 0.0)),
                "h2h_home_win_rate": float(h2h_home_win_rate),
                "h2h_away_win_rate": float(h2h_away_win_rate),
                "h2h_home_goal_avg": float(h2h_home_goal_avg),
                "h2h_away_goal_avg": float(h2h_away_goal_avg),
                "h2h_matches": float(h2h_matches_out),
            }
        )
        output_rows.append(row_output)

        # Compute match result and update H2H + last match dates
        home_points, away_points = _match_points(home_goals, away_goals)
        home_win = home_points == 3.0
        away_win = away_points == 3.0

        # Update last match date for both teams (track across home and away)
        last_match_date[home_team] = match_date
        last_match_date[away_team] = match_date

        # Update H2H state for this pair
        pair = h2h_states[pair_key]
        pair["matches"] += 1.0
        pair["home_wins"] += 1.0 if home_win else 0.0
        pair["away_wins"] += 1.0 if away_win else 0.0
        pair["home_goals"] += float(home_goals)
        pair["away_goals"] += float(away_goals)

        # Update streak states for both teams
        _push_streak_state(streak_states, home_team, home_points)
        _push_streak_state(streak_states, away_team, away_points)

        _push_context_state(home_state["all"], home_points, home_goals, away_goals, home_win, rolling_window, match_date)
        _push_context_state(away_state["all"], away_points, away_goals, home_goals, away_win, rolling_window, match_date)

        _push_context_state(home_state["home"], home_points, home_goals, away_goals, home_win, rolling_window, match_date)
        _push_context_state(away_state["away"], away_points, away_goals, home_goals, away_win, rolling_window, match_date)

        _push_overall_state(home_state, home_points, home_goals, away_goals, rolling_window)
        _push_overall_state(away_state, away_points, away_goals, home_goals, rolling_window)

    result = pd.DataFrame(output_rows)

    if preserve_original_order and "_feature_order" in result.columns:
        result = result.sort_values("_feature_order", kind="stable")

    if "_feature_order" in result.columns:
        result = result.drop(columns=["_feature_order"])

    return result.reset_index(drop=True)


def add_features(
    df,
    rolling_window=DEFAULT_ROLLING_WINDOW,
    include_momentum=True,
    preserve_original_order=False,
    normalize_strengths=True,
    use_time_decay=True,
):
    return compute_home_away_stats(
        df,
        rolling_window=rolling_window,
        include_momentum=include_momentum,
        preserve_original_order=preserve_original_order,
        normalize_strengths=normalize_strengths,
        use_time_decay=use_time_decay,
    )
