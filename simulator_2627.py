import math
import random
from collections import defaultdict

# 2026/27 Premier League teams with starting Elo from end of 2025/26
# Relegated: Wolves, Burnley, West Ham
# Promoted: Coventry, Ipswich, Southampton
TEAMS_2026_27 = {
    "Arsenal":         1648.4,
    "Man City":        1648.2,
    "Liverpool":       1578.3,
    "Man United":      1573.0,
    "Bournemouth":     1548.9,
    "Aston Villa":     1547.6,
    "Brighton":        1535.6,
    "Brentford":       1525.0,
    "Chelsea":         1515.2,
    "Everton":         1511.2,
    "Newcastle":       1508.1,
    "Nott'm Forest":   1507.3,
    "Fulham":          1504.0,
    "Leeds":           1497.8,
    "Crystal Palace":  1489.2,
    "Sunderland":      1485.7,
    "Tottenham":       1450.4,
    "Coventry":        1410.0,
    "Ipswich":         1394.4,
    "Southampton":     1380.0,
}

HOME_ADVANTAGE = 50.0
N_SIMULATIONS = 10000


def get_win_probability(home_elo: float, away_elo: float) -> tuple:
    adjusted_home = home_elo + HOME_ADVANTAGE
    exponent = (away_elo - adjusted_home) / 400.0
    home_expected = 1.0 / (1.0 + 10.0 ** exponent)
    away_expected = 1.0 - home_expected
    draw_prob = 0.22 + 0.12 * (1.0 - abs(home_expected - 0.5) * 2)
    home_win_prob = home_expected * (1.0 - draw_prob)
    away_win_prob = away_expected * (1.0 - draw_prob)
    return home_win_prob, draw_prob, away_win_prob


def generate_fixtures(teams: list) -> list:
    fixtures = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i != j:
                fixtures.append((home, away))
    return fixtures


def simulate_season(teams_elo: dict, fixtures: list) -> dict:
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


def run_simulation_2627(n: int = N_SIMULATIONS) -> dict:
    teams = list(TEAMS_2026_27.keys())
    fixtures = generate_fixtures(teams)
    position_counts = defaultdict(lambda: defaultdict(int))
    for _ in range(n):
        points = simulate_season(TEAMS_2026_27, fixtures)
        standings = sorted(points.items(), key=lambda x: -x[1])
        for pos, (team, _) in enumerate(standings, start=1):
            position_counts[team][pos] += 1
    results = {}
    for team in teams:
        results[team] = {
            "title_prob": round(position_counts[team][1] / n * 100, 1),
                "top5_prob": round(sum(position_counts[team][p] for p in range(1, 6)) / n * 100, 1),
            "top6_prob": round(sum(position_counts[team][p] for p in range(1, 7)) / n * 100, 1),
            "relegation_prob": round(sum(position_counts[team][p] for p in range(18, 21)) / n * 100, 1),
            "avg_position": round(sum(p * position_counts[team][p] for p in range(1, 21)) / n, 1),
            "elo": TEAMS_2026_27[team],
        }
    return results


if __name__ == "__main__":
    print(f"Running {N_SIMULATIONS} simulations of the 2026/27 Premier League season...\n")
    results = run_simulation_2627()
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["avg_position"])
    print(f"{'Pos':<4} {'Team':<20} {'Elo':<8} {'Title%':<10} {'Top5%':<10} {'Top6%':<10} {'Rel%':<10} {'AvgPos'}")
    print("-" * 80)
    for i, (team, stats) in enumerate(sorted_teams, start=1):
        print(
            f"{i:<4} {team:<20} {stats['elo']:<8.0f} "
            f"{stats['title_prob']:<10} {stats.get('top5_prob', 0):<10} "
            f"{stats['top6_prob']:<10} {stats['relegation_prob']:<10} "
            f"{stats['avg_position']}"
        )
