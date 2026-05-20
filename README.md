# Premier League Match Predictor

An end-to-end Premier League analytics project that combines match prediction, Elo-based season simulation, and a small React frontend.

## What It Does

- Predicts match outcomes from team form, xG, and Elo-based features
- Simulates a full Premier League season with Monte Carlo methods
- Serves both tools through a FastAPI backend
- Presents the results in a dark-themed React frontend

## Tech Stack

- Python, Pandas, NumPy, XGBoost, FastAPI
- React, Vite
- Data collection from football-data.co.uk and Understat

## Project Structure

- `api.py` - FastAPI backend for match prediction and season simulation
- `simulator.py` - Monte Carlo season simulator using Elo ratings
- `collect_data.py` - Builds the match dataset from external sources
- `feature_engineering.py` and `Features.py` - Feature preparation pipeline
- `train_model.py` - Model training workflow
- `frontend/` - React UI for match prediction and the season simulator

## Local Setup

1. Create and activate the Python virtual environment.
2. Install Python dependencies.
3. Run the FastAPI app with `uvicorn api:app --reload`.
4. Start the frontend from the `frontend/` folder with the Vite dev server.

## API Endpoints

- `GET /teams` - list available teams
- `POST /predict` - predict a single match
- `GET /simulate` - run the 10,000-simulation pre-season forecast

## Why I built it 
As a big soccer fan, I’ve always found myself trying to predict matches before they happen. One day, I thought, “Why not build my own machine learning model to do it?” That idea eventually turned into this project.

Looking back, if I were to rebuild it, I would focus more on incorporating player injuries and squad availability into the prediction system, since those factors can heavily impact match outcomes. Even so, I’m really happy with how the project turned out and with everything I learned while building it.
