import json
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Features import process_matches
from feature_engineering import add_features, build_pre_match_features
from simulator import run_simulation
from simulator_2627 import run_simulation_2627

FEATURE_COLS_PATH = "model/feature_cols_full.json"
MODEL_PATH = "model/xgb_full.json"
DATA_PATH = "Data/master.csv"

TARGET_LABELS = {0: "H", 1: "D", 2: "A"}
RESULT_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

# Serve-time ensemble: final = BLEND_WEIGHT * xgb + (1 - BLEND_WEIGHT) * elo_implied.
# w=0.6 validated out-of-sample: refitting the weight on each test season by
# min log loss gave 0.54 (fit 2024/25) and 0.62 (fit 2023/24); both beat pure
# XGBoost on the held-out season (log loss -0.016 / -0.017). 0.6 sits between.
BLEND_WEIGHT = 0.6
ELO_HOME_ADVANTAGE = 50.0   # matches Features.HOME_ADVANTAGE
ELO_DRAW_RATE = 0.24        # fixed league-average draw rate for the Elo-implied 1X2


def elo_implied_probs(home_elo, away_elo):
    """Elo-implied H/D/A from pre-match Elo and the +50 home advantage.

    Draw is fixed near the league rate; the remainder is split by the Elo
    expected score.  Exact mapping (from the calibrated sweep):
        E = 1 / (1 + 10 ** (-((home_elo + 50 - away_elo) / 400)))
        P(H), P(D), P(A) = (1 - 0.24) * E, 0.24, (1 - 0.24) * (1 - E)
    """
    expected_home = 1.0 / (1.0 + 10.0 ** (-((float(home_elo) + ELO_HOME_ADVANTAGE - float(away_elo)) / 400.0)))
    rest = 1.0 - ELO_DRAW_RATE
    return np.array([rest * expected_home, ELO_DRAW_RATE, rest * (1.0 - expected_home)])


def _iso_date(value):
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _normalize_season(value):
    if pd.isna(value):
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 4:
        return digits[-4:]
    return digits or None


def _add_season_match_counts(df):
    counts = defaultdict(int)
    home_col, away_col = [], []
    for row in df.itertuples(index=False):
        season = _normalize_season(getattr(row, "Season", None))
        hk = (getattr(row, "HomeTeam"), season)
        ak = (getattr(row, "AwayTeam"), season)
        home_col.append(float(counts[hk]))
        away_col.append(float(counts[ak]))
        counts[hk] += 1
        counts[ak] += 1
    df = df.copy()
    df["home_matches_played"] = home_col
    df["away_matches_played"] = away_col
    return df


def build_serving_feature_row(history_df: pd.DataFrame, fixture: dict, feature_cols: list) -> dict:
    """Replay the training feature state, then emit one future fixture.

    The fixture's result and xG fields are deliberately unknown.  `add_features`
    reads state and emits before it updates state, so this yields the same xG and
    H2H snapshots used by training without exposing current-match information.
    """
    return build_pre_match_features(history_df, fixture, feature_cols)


# Global state
app_state = {}


def build_team_states(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    For each team store two snapshots:
    - 'home': stats from their most recent home match
    - 'away': stats from their most recent away match
    Current Elo is tracked separately as a single value.
    """
    team_states = {}
    df_sorted = df.sort_values("Date")

    # Track current Elo from most recent appearance as either home or away
    current_elo = {}
    for _, row in df_sorted.iterrows():
        home = row.get("HomeTeam")
        away = row.get("AwayTeam")
        if pd.notna(home):
            current_elo[home] = float(row.get("home_elo_before") or 1500.0)
        if pd.notna(away):
            current_elo[away] = float(row.get("away_elo_before") or 1500.0)

    # Get last home match stats for each team
    home_latest = df_sorted.groupby("HomeTeam").last().reset_index()
    for _, row in home_latest.iterrows():
        team = row["HomeTeam"]
        if team not in team_states:
            team_states[team] = {}
        for col in feature_cols:
            if col.startswith("home_"):
                team_states[team][col] = float(row.get(col, 0.0) or 0.0)

    # Get last away match stats for each team
    away_latest = df_sorted.groupby("AwayTeam").last().reset_index()
    for _, row in away_latest.iterrows():
        team = row["AwayTeam"]
        if team not in team_states:
            team_states[team] = {}
        for col in feature_cols:
            if col.startswith("away_"):
                team_states[team][col] = float(row.get(col, 0.0) or 0.0)

    # Override Elo with current value for both contexts
    for team, elo in current_elo.items():
        if team in team_states:
            team_states[team]["home_elo_before"] = elo
            team_states[team]["away_elo_before"] = elo

    return team_states


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    app_state["booster"] = booster

    # Load feature cols
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)
    app_state["feature_cols"] = feature_cols

    # Load and process data
    df = pd.read_csv(DATA_PATH)
    df = process_matches(df)
    app_state["history_df"] = df.copy()
    df = add_features(df)
    df = _add_season_match_counts(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    # Fill nulls
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Store available teams
    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    app_state["teams"] = teams

    # Build team state snapshots
    app_state["team_states"] = build_team_states(df, feature_cols)
    app_state["df"] = df

    print(f"Loaded {len(teams)} teams")
    yield
    app_state.clear()


app = FastAPI(title="Premier League Match Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_result: str
    predicted_result_label: str
    # Data-vintage provenance: the last dated match each team appears in, plus
    # the dataset's most recent match. Lets the UI flag teams whose features are
    # stale (e.g. a club that left the Premier League years ago).
    home_last_match_date: str | None = None
    away_last_match_date: str | None = None
    data_max_date: str | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Premier League Match Predictor API"}


@app.get("/teams")
def get_teams():
    return {"teams": app_state["teams"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    home_team = request.home_team
    away_team = request.away_team

    teams = app_state["teams"]
    if home_team not in teams:
        raise HTTPException(status_code=404, detail=f"Team '{home_team}' not found")
    if away_team not in teams:
        raise HTTPException(status_code=404, detail=f"Team '{away_team}' not found")
    if home_team == away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must be different")

    feature_cols = app_state["feature_cols"]
    team_states = app_state["team_states"]

    home_state = team_states.get(home_team, {})
    away_state = team_states.get(away_team, {})
    history_df = app_state["history_df"]
    feature_row = build_serving_feature_row(
        history_df,
        {
            "Date": pd.to_datetime(history_df["Date"], errors="coerce").max() + pd.Timedelta(days=1),
            "Season": history_df.iloc[-1].get("Season"),
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "home_elo_before": home_state.get("home_elo_before", 1500.0),
            "away_elo_before": away_state.get("away_elo_before", 1500.0),
        },
        feature_cols,
    )

    x = pd.DataFrame([feature_row])[feature_cols].fillna(0.0)
    dmatrix = xgb.DMatrix(x)

    booster = app_state["booster"]
    xgb_probs = booster.predict(dmatrix)[0]

    # Serve-time ensemble with the Elo-implied 1X2 (opponent-symmetric prior).
    elo_probs = elo_implied_probs(
        home_state.get("home_elo_before", 1500.0),
        away_state.get("away_elo_before", 1500.0),
    )
    probs = BLEND_WEIGHT * np.asarray(xgb_probs, dtype=float) + (1.0 - BLEND_WEIGHT) * elo_probs
    # Both inputs are proper distributions, so the convex blend also sums to 1;
    # normalize defensively against float drift.
    probs = probs / probs.sum()

    predicted_class = int(np.argmax(probs))
    predicted_result = TARGET_LABELS[predicted_class]

    # Data-vintage: each team's most recent dated appearance, and the dataset max.
    full_df = app_state["df"]

    def _last_match_date(team):
        appears = (full_df["HomeTeam"] == team) | (full_df["AwayTeam"] == team)
        dates = full_df.loc[appears, "Date"].dropna()
        return _iso_date(dates.max()) if len(dates) else None

    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=round(float(probs[0]), 4),
        draw_prob=round(float(probs[1]), 4),
        away_win_prob=round(float(probs[2]), 4),
        predicted_result=predicted_result,
        predicted_result_label=RESULT_LABELS[predicted_result],
        home_last_match_date=_last_match_date(home_team),
        away_last_match_date=_last_match_date(away_team),
        data_max_date=_iso_date(full_df["Date"].max()),
    )


@app.get("/simulate")
def simulate():
    results = run_simulation(n=10000)
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["avg_position"])
    return {
        "table": [
            {
                "position": i + 1,
                "team": team,
                "elo": round(stats["elo"]),
                "title_prob": stats["title_prob"],
                "top5_prob": stats.get("top5_prob", stats.get("top4_prob", 0)),
                "top6_prob": stats["top6_prob"],
                "relegation_prob": stats["relegation_prob"],
                "avg_position": stats["avg_position"],
            }
            for i, (team, stats) in enumerate(sorted_teams)
        ]
    }


@app.get("/simulate2627")
def simulate_2627():
    results = run_simulation_2627(n=10000)
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["avg_position"])
    return {
        "table": [
            {
                "position": i + 1,
                "team": team,
                "elo": round(stats["elo"]),
                "title_prob": stats["title_prob"],
                "top5_prob": stats.get("top5_prob", stats.get("top4_prob", 0)),
                "top6_prob": stats["top6_prob"],
                "relegation_prob": stats["relegation_prob"],
                "avg_position": stats["avg_position"],
            }
            for i, (team, stats) in enumerate(sorted_teams)
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
