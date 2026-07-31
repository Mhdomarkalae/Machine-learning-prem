# Premier League Match Predictor

End-to-end football analytics project that predicts Premier League match results and simulates season outcomes with a FastAPI backend and React frontend.

## Overview

This repository combines the full lifecycle of a production-style data project:

- data collection from historical Premier League sources
- feature engineering with rolling team form, Elo, xG, and head-to-head context
- multiclass match prediction with XGBoost
- Monte Carlo season simulation for league table forecasts
- a React UI for interactive match predictions and season views

## Highlights

- Predicts home win, draw, and away win probabilities for any supported fixture
- Runs full-season simulations to estimate title, European qualification, and relegation chances
- Uses a reproducible local setup with no API keys required to run the core app
- Packages the model, backend, and frontend in one portfolio-ready repo

## Tech Stack

- Python, Pandas, NumPy, SciPy
- XGBoost, scikit-learn, FastAPI, Uvicorn
- React, Vite
- Historical data from football-data.co.uk and Understat

## Project Structure

- `api.py` - FastAPI app that loads the model, serves predictions, and exposes simulation endpoints
- `collect_data.py` - historical data ingestion and dataset build pipeline
- `feature_engineering.py` - rolling feature generation for match inputs
- `Features.py` - match preprocessing helpers used by training and inference
- `train_model.py` - model training, evaluation, and artifact export
- `simulator.py` and `simulator_2627.py` - season simulation engines
- `frontend/` - React UI for the user-facing application
- `model/` - saved model artifacts and feature column definitions
- `Data/` - raw and processed datasets used by the project

## Local Setup

### Backend

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the API with `uvicorn api:app --reload`.

### Frontend

1. Change into `frontend/`.
2. Install dependencies with `npm install`.
3. Start the UI with `npm run dev`.

## API Endpoints

- `GET /` - health check
- `GET /teams` - returns the available team list
- `POST /predict` - returns match win/draw/loss probabilities
- `GET /simulate` - returns the standard season simulation table
- `GET /simulate2627` - returns the 2026/27 simulation table

## Why I built it

As a big soccer fan, I’ve always found myself trying to predict matches before they happen. One day, I thought, “Why not build my own machine learning model to do it?” That idea eventually turned into this project.

Looking back, if I were to rebuild it, I would focus more on incorporating player injuries and squad availability into the prediction system, since those factors can heavily impact match outcomes. Even so, I’m really happy with how the project turned out and with everything I learned while building it.

## Notes For Recruiters

- The repo demonstrates practical machine learning work from raw data to deployment.
- The backend is self-contained and can be run locally without external credentials.
- The model artifacts are already saved in `model/`, so the app can be reviewed quickly without retraining.

## Next Improvements

- Add screenshots or a short demo GIF to the README.
- Deploy the app and link it here for one-click review.
- Add a short architecture diagram showing data flow from ingestion to prediction.
