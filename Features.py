import math

import pandas as pd


DEFAULT_ELO = 1500.0
HOME_ADVANTAGE = 50.0
EARLY_SEASON_K = 30.0
MID_SEASON_K = 22.5
LATE_SEASON_K = 15.0
ELO_FLOOR = 800.0
ELO_CEILING = 2200.0
XG_ADJUSTMENT_CAP = 0.15
XG_SCALE = 1.5
SEASON_REVERSION_RATE = 0.25
INACTIVITY_THRESHOLD_DAYS = 30
INACTIVITY_DECAY_RATE = 0.01


def initialize_elo(teams, initial_rating=DEFAULT_ELO):
    return {team: float(initial_rating) for team in teams}


def apply_mean_reversion(ratings, anchor=DEFAULT_ELO, reversion_rate=SEASON_REVERSION_RATE):
    return {
        team: float(rating + (anchor - rating) * reversion_rate)
        for team, rating in ratings.items()
    }


def apply_inactivity_decay(rating, inactive_days, anchor=DEFAULT_ELO, threshold_days=INACTIVITY_THRESHOLD_DAYS, decay_rate=INACTIVITY_DECAY_RATE):
    if inactive_days is None or inactive_days <= threshold_days:
        return float(rating)

    decay_days = inactive_days - threshold_days
    decay_factor = 1.0 - min(decay_days * decay_rate, 0.25)
    return float(rating + (anchor - rating) * (1.0 - decay_factor))


def get_expected_score(home_rating, away_rating, home_advantage=HOME_ADVANTAGE):
    adjusted_home_rating = home_rating + home_advantage
    exponent = (away_rating - adjusted_home_rating) / 400.0
    home_expected = 1.0 / (1.0 + 10.0 ** exponent)
    away_expected = 1.0 - home_expected
    return home_expected, away_expected


def _get_match_score(home_goals, away_goals):
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def _get_dynamic_k(match_number=None, matches_in_scope=None, season_position=None, total_season_matches=None):
    if season_position is not None and total_season_matches:
        progress = season_position / max(total_season_matches, 1)
        if progress < 0.33:
            return EARLY_SEASON_K
        if progress < 0.66:
            return MID_SEASON_K
        return LATE_SEASON_K

    if match_number is not None and matches_in_scope:
        progress = match_number / max(matches_in_scope, 1)
        if progress < 0.33:
            return EARLY_SEASON_K
        if progress < 0.66:
            return MID_SEASON_K
        return LATE_SEASON_K

    return MID_SEASON_K


def _goal_margin_multiplier(goal_difference):
    if goal_difference <= 1:
        return 1.0
    return 1.0 + min(goal_difference - 1, 4) * 0.10


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_xg_bonus_zero_sum(
    home_xg,
    away_xg,
    home_goals,
    away_goals,
    max_adjustment=XG_ADJUSTMENT_CAP,
    scale=XG_SCALE,
    base_k=MID_SEASON_K,
):
    """
    Returns zero-sum xG bonus pair: (home_bonus, away_bonus).
    Ensures home_bonus + away_bonus = 0 for strict Elo conservation.
    Works for both draws and non-draws.
    """
    home_xg_value = _safe_float(home_xg, 0.0)
    away_xg_value = _safe_float(away_xg, 0.0)

    xg_diff = home_xg_value - away_xg_value
    xg_signal = math.tanh(xg_diff / max(scale, 1e-9))

    if home_goals == away_goals:
        bonus_magnitude = (max_adjustment * 0.25 * base_k) * xg_signal
    else:
        match_result = 1.0 if home_goals > away_goals else -1.0
        bonus_magnitude = (max_adjustment * 0.5 * base_k) * match_result * xg_signal

    bonus_magnitude = max(-max_adjustment * base_k, min(max_adjustment * base_k, bonus_magnitude))

    return {
        "home_xg": home_xg_value,
        "away_xg": away_xg_value,
        "xg_diff": float(xg_diff),
        "xg_bonus_home": float(bonus_magnitude),
        "xg_bonus_away": float(-bonus_magnitude),
    }


def update_elo(
    home_rating,
    away_rating,
    home_goals,
    away_goals,
    k_factor=MID_SEASON_K,
    home_advantage=HOME_ADVANTAGE,
    use_goal_margin=True,
    use_xg_adjustment=True,
    home_xg=None,
    away_xg=None,
    clip_bounds=(ELO_FLOOR, ELO_CEILING),
):
    """
    Update Elo ratings with strict zero-sum enforcement.
    home_delta + away_delta = 0 at all stages.
    """
    home_expected, away_expected = get_expected_score(
        home_rating,
        away_rating,
        home_advantage=home_advantage,
    )
    home_score, away_score = _get_match_score(home_goals, away_goals)

    goal_margin_multiplier = _goal_margin_multiplier(abs(home_goals - away_goals)) if use_goal_margin else 1.0
    effective_k = float(k_factor) * goal_margin_multiplier

    raw_home_delta = effective_k * (home_score - home_expected)
    raw_away_delta = effective_k * (away_score - away_expected)

    xg_info = {
        "home_xg": _safe_float(home_xg, 0.0),
        "away_xg": _safe_float(away_xg, 0.0),
    }
    xg_info["xg_diff"] = xg_info["home_xg"] - xg_info["away_xg"]

    if use_xg_adjustment:
        xg_bonus = get_xg_bonus_zero_sum(
            home_xg=home_xg,
            away_xg=away_xg,
            home_goals=home_goals,
            away_goals=away_goals,
            base_k=k_factor,
        )
        xg_bonus_home = xg_bonus["xg_bonus_home"]
        xg_bonus_away = xg_bonus["xg_bonus_away"]
        xg_info.update(xg_bonus)
    else:
        xg_bonus_home = 0.0
        xg_bonus_away = 0.0
        xg_info["xg_bonus_home"] = 0.0
        xg_info["xg_bonus_away"] = 0.0

    adjusted_home_delta = raw_home_delta + xg_bonus_home
    adjusted_away_delta = raw_away_delta + xg_bonus_away

    home_after = home_rating + adjusted_home_delta
    away_after = away_rating + adjusted_away_delta

    lower_bound, upper_bound = clip_bounds
    home_after = max(lower_bound, min(upper_bound, home_after))
    away_after = max(lower_bound, min(upper_bound, away_after))

    return {
        "home_elo_before": float(home_rating),
        "away_elo_before": float(away_rating),
        "home_elo_after": float(home_after),
        "away_elo_after": float(away_after),
        "elo_diff_before": float(home_rating - away_rating),
        "elo_change_raw": float(raw_home_delta),
        "elo_change_adjusted": float(adjusted_home_delta),
        "home_expected_score": float(home_expected),
        "away_expected_score": float(away_expected),
        "k_factor_used": float(effective_k),
        **xg_info,
    }


def process_matches(
    df,
    season_column="Season",
    reset_each_season=False,
    initial_rating=DEFAULT_ELO,
    home_advantage=HOME_ADVANTAGE,
    use_goal_margin=True,
    use_xg_adjustment=True,
    use_season_reversion=True,
    season_reversion_rate=SEASON_REVERSION_RATE,
    use_inactivity_decay=True,
    inactivity_threshold_days=INACTIVITY_THRESHOLD_DAYS,
    inactivity_decay_rate=INACTIVITY_DECAY_RATE,
    clip_bounds=(ELO_FLOOR, ELO_CEILING),
):
    if df is None or df.empty:
        return pd.DataFrame(df).copy()

    required_columns = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")

    working_df = df.copy()
    if "Date" in working_df.columns:
        working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")
        sort_columns = ["Date"]
        if season_column in working_df.columns:
            sort_columns.append(season_column)
        working_df = working_df.sort_values(sort_columns, kind="mergesort")
    elif season_column in working_df.columns:
        working_df = working_df.sort_values([season_column], kind="mergesort")

    if "home_xg" not in working_df.columns:
        working_df["home_xg"] = 0.0
    if "away_xg" not in working_df.columns:
        working_df["away_xg"] = 0.0

    working_df["home_xg"] = pd.to_numeric(working_df["home_xg"], errors="coerce")
    working_df["away_xg"] = pd.to_numeric(working_df["away_xg"], errors="coerce")
    working_df["xg_diff"] = working_df["home_xg"].fillna(0.0) - working_df["away_xg"].fillna(0.0)

    teams = pd.unique(pd.concat([working_df["HomeTeam"], working_df["AwayTeam"]], ignore_index=True).dropna())
    elo = initialize_elo(teams, initial_rating=initial_rating)
    last_played_date = {team: None for team in teams}

    if reset_each_season and season_column in working_df.columns and use_season_reversion:
        reset_each_season = False

    output_rows = []
    previous_season = None

    def process_match_row(row, elo_dict, match_count, total_count, is_season_reset, previous_season):
        """
        Process a single match row and return Elo update and updated previous_season.
        elo_dict is updated in-place.
        Returns (elo_update, previous_season).
        """
        match_date = pd.to_datetime(getattr(row, "Date", pd.NaT), errors="coerce")
        current_season = getattr(row, season_column, None) if season_column in working_df.columns else None

        if (
            use_season_reversion
            and season_column in working_df.columns
            and current_season is not None
            and current_season != previous_season
            and previous_season is not None
            and not is_season_reset
        ):
            for team in elo_dict:
                elo_dict[team] = elo_dict[team] + (initial_rating - elo_dict[team]) * season_reversion_rate

        home_team = row.HomeTeam
        away_team = row.AwayTeam
        home_goals = row.FTHG
        away_goals = row.FTAG
        home_xg = getattr(row, "home_xg", 0.0)
        away_xg = getattr(row, "away_xg", 0.0)

        if use_inactivity_decay and pd.notna(match_date):
            home_last_played = last_played_date.get(home_team)
            away_last_played = last_played_date.get(away_team)

            if home_last_played is not None and pd.notna(home_last_played):
                inactive_days = (match_date - home_last_played).days
                elo_dict[home_team] = apply_inactivity_decay(
                    elo_dict.get(home_team, initial_rating),
                    inactive_days,
                    anchor=initial_rating,
                    threshold_days=inactivity_threshold_days,
                    decay_rate=inactivity_decay_rate,
                )

            if away_last_played is not None and pd.notna(away_last_played):
                inactive_days = (match_date - away_last_played).days
                elo_dict[away_team] = apply_inactivity_decay(
                    elo_dict.get(away_team, initial_rating),
                    inactive_days,
                    anchor=initial_rating,
                    threshold_days=inactivity_threshold_days,
                    decay_rate=inactivity_decay_rate,
                )

        home_rating = float(elo_dict.get(home_team, initial_rating))
        away_rating = float(elo_dict.get(away_team, initial_rating))
        k_factor = _get_dynamic_k(
            match_number=match_count,
            matches_in_scope=total_count,
        )

        elo_update = update_elo(
            home_rating=home_rating,
            away_rating=away_rating,
            home_goals=home_goals,
            away_goals=away_goals,
            k_factor=k_factor,
            home_advantage=home_advantage,
            use_goal_margin=use_goal_margin,
            use_xg_adjustment=use_xg_adjustment,
            home_xg=home_xg,
            away_xg=away_xg,
            clip_bounds=clip_bounds,
        )

        elo_dict[home_team] = elo_update["home_elo_after"]
        elo_dict[away_team] = elo_update["away_elo_after"]
        if pd.notna(match_date):
            last_played_date[home_team] = match_date
            last_played_date[away_team] = match_date
        previous_season = current_season

        return elo_update, previous_season

    if reset_each_season and season_column in working_df.columns:
        season_groups = working_df.groupby(season_column, sort=False, dropna=False)
        for _, season_df in season_groups:
            season_elo = initialize_elo(teams, initial_rating=initial_rating)
            season_total_matches = len(season_df)
            season_previous = None

            for season_position, row in enumerate(season_df.itertuples(index=False), start=1):
                elo_update, season_previous = process_match_row(row, season_elo, season_position, season_total_matches, is_season_reset=True, previous_season=season_previous)
                row_output = row._asdict()
                row_output.update(elo_update)
                output_rows.append(row_output)
    else:
        total_matches = len(working_df)
        for match_number, row in enumerate(working_df.itertuples(index=False), start=1):
            elo_update, previous_season = process_match_row(row, elo, match_number, total_matches, is_season_reset=False, previous_season=previous_season)
            row_output = row._asdict()
            row_output.update(elo_update)
            output_rows.append(row_output)

    result = pd.DataFrame(output_rows)

    if not result.empty and "Date" in result.columns:
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")

    return result.reset_index(drop=True)


if __name__ == "__main__":
    source_df = pd.read_csv("Data/master.csv")
    enriched_df = process_matches(source_df)
    enriched_df.to_csv("Data/master_with_elo.csv", index=False)