import math
import random
from collections import defaultdict

import pandas as pd

# 2025/26 Premier League teams with starting Elo from end of 2024/25
TEAMS_2025_26 = {
    "Liverpool":       1656.8,
    "Arsenal":         1634.6,
    "Man City":        1625.2,
    "Aston Villa":     1583.3,
    "Newcastle":       1575.6,
    "Chelsea":         1575.5,
    "Brighton":        1535.9,
    "Nott'm Forest":   1532.4,
    "Crystal Palace":  1526.6,
    "Brentford":       1516.5,
    "Bournemouth":     1515.1,
    "Fulham":          1511.0,
    "Everton":         1500.6,
    "Man United":      1464.8,
    "West Ham":        1463.7,
    "Tottenham":       1461.3,
    "Leeds":           1418.3,
    "Burnley":         1399.9,
    "Sunderland":      1370.1,
}

HOME_ADVANTAGE = 50.0
N_SIMULATIONS = 10000


def get_win_probability(home_elo: float, away_elo: float) -> tuple:
    """Return (home_win_prob, draw_prob, away_win_prob) using Elo."""
    adjusted_home = home_elo + HOME_ADVANTAGE
    exponent = (away_elo - adjusted_home) / 400.0
    home_expected = 1.0 / (1.0 + 10.0 ** exponent)
    away_expected = 1.0 - home_expected

    # Convert to 3-way probabilities using standard draw adjustment
    draw_prob = 0.22 + 0.12 * (1.0 - abs(home_expected - 0.5) * 2)
    home_win_prob = home_expected * (1.0 - draw_prob)
    away_win_prob = away_expected * (1.0 - draw_prob)

    return home_win_prob, draw_prob, away_win_prob


def generate_fixtures(teams: list) -> list:
    """Generate full round-robin fixture list (home and away)."""
    fixtures = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i != j:
                fixtures.append((home, away))
    return fixtures


def simulate_season(teams_elo: dict, fixtures: list) -> dict:
    """Simulate one full season. Returns final points table."""
    points = defaultdict(int)
    for team in teams_elo:
        points[team] = 0

    for home, away in fixtures:
        home_elo = teams_elo[home]
        away_elo = teams_elo[away]
        home_win_prob, draw_prob, away_win_prob = get_win_probability(home_elo, away_elo)

        r = random.random()
        if r < home_win_prob:
            points[home] += 3
        elif r < home_win_prob + draw_prob:
            points[home] += 1
            points[away] += 1
        else:
            points[away] += 3

    return dict(points)


def run_simulation(n: int = N_SIMULATIONS) -> dict:
    """Run n simulations and return finish position probabilities."""
    teams = list(TEAMS_2025_26.keys())
    fixtures = generate_fixtures(teams)

    # position_counts[team][position] = number of times team finished in that position
    position_counts = defaultdict(lambda: defaultdict(int))

    for _ in range(n):
        points = simulate_season(TEAMS_2025_26, fixtures)
        # Sort by points descending
        standings = sorted(points.items(), key=lambda x: -x[1])
        for pos, (team, _) in enumerate(standings, start=1):
            position_counts[team][pos] += 1

    # Convert to probabilities
    results = {}
    for team in teams:
        results[team] = {
            "title_prob": round(position_counts[team][1] / n * 100, 1),
            "top4_prob": round(sum(position_counts[team][p] for p in range(1, 5)) / n * 100, 1),
            "top6_prob": round(sum(position_counts[team][p] for p in range(1, 7)) / n * 100, 1),
            "relegation_prob": round(sum(position_counts[team][p] for p in range(18, 21)) / n * 100, 1),
            "avg_position": round(sum(p * position_counts[team][p] for p in range(1, 21)) / n, 1),
            "elo": TEAMS_2025_26[team],
        }

    return results


if __name__ == "__main__":
    print(f"Running {N_SIMULATIONS} simulations of the 2025/26 Premier League season...\n")
    results = run_simulation()

    # Sort by average position
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["avg_position"])

    print(f"{'Pos':<4} {'Team':<20} {'Elo':<8} {'Title%':<10} {'Top4%':<10} {'Top6%':<10} {'Rel%':<10} {'AvgPos'}")
    print("-" * 80)
    for i, (team, stats) in enumerate(sorted_teams, start=1):
        print(
            f"{i:<4} {team:<20} {stats['elo']:<8.0f} "
            f"{stats['title_prob']:<10} {stats['top4_prob']:<10} "
            f"{stats['top6_prob']:<10} {stats['relegation_prob']:<10} "
            f"{stats['avg_position']}"
        )
