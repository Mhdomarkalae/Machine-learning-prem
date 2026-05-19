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
from feature_engineering import add_features
from simulator import run_simulation
from simulator_2627 import run_simulation_2627

FEATURE_COLS_PATH = "model/feature_cols_full.json"
MODEL_PATH = "model/xgb_full.json"
DATA_PATH = "Data/master.csv"

TARGET_LABELS = {0: "H", 1: "D", 2: "A"}
RESULT_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


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

    feature_row = {}
    home_state = team_states.get(home_team, {})
    away_state = team_states.get(away_team, {})

    # Fill home_ and away_ prefixed features from each team's state
    for col in feature_cols:
        if col.startswith("home_"):
            feature_row[col] = home_state.get(col, 0.0)
        elif col.startswith("away_"):
            feature_row[col] = away_state.get(col, 0.0)
        else:
            feature_row[col] = 0.0

    # Recompute derived features that depend on both teams
    home_elo = feature_row.get("home_elo_before", 1500.0)
    away_elo = feature_row.get("away_elo_before", 1500.0)
    home_xg = feature_row.get("home_xg", 0.0)
    away_xg = feature_row.get("away_xg", 0.0)
    xg_diff = home_xg - away_xg

    elo_diff = home_elo - away_elo
    feature_row["elo_diff"] = elo_diff
    feature_row["elo_xg_diff"] = elo_diff + (xg_diff * 50.0)

    home_form = feature_row.get("home_form_points_avg", 0.0)
    away_form = feature_row.get("away_form_points_avg", 0.0)
    import math
    elo_component = math.tanh(elo_diff / 400.0)
    form_component = math.tanh((home_form - away_form) / 3.0)
    feature_row["elo_form_interaction"] = elo_component * form_component

    home_advantage_index = feature_row.get("home_advantage_index", 0.0)
    feature_row["elo_strength_interaction"] = elo_component * home_advantage_index

    x = pd.DataFrame([feature_row])[feature_cols].fillna(0.0)
    dmatrix = xgb.DMatrix(x)

    booster = app_state["booster"]
    probs = booster.predict(dmatrix)[0]

    predicted_class = int(np.argmax(probs))
    predicted_result = TARGET_LABELS[predicted_class]

    return PredictionResponse(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=round(float(probs[0]), 4),
        draw_prob=round(float(probs[1]), 4),
        away_win_prob=round(float(probs[2]), 4),
        predicted_result=predicted_result,
        predicted_result_label=RESULT_LABELS[predicted_result],
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