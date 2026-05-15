import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
import pandas as pd
import os
import time
import asyncio
import aiohttp
import understat

# ─────────────────────────────────────────────────────────────
# Phase 1: Data Collection
# ─────────────────────────────────────────────────────────────

SEASONS_DIR = "Data/Seasons"
DATA_DIR    = "Data"
os.makedirs(SEASONS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PART 1: Download season CSVs from football-data.co.uk
# ─────────────────────────────────────────────────────────────

FD_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS",   "AS",
    "HST",  "AST",
    "HF",   "AF",
    "HC",   "AC",
    "HY",   "AY",
    "HR",   "AR",
    "Referee",
    "B365H", "B365D", "B365A",
]

SEASONS = [
    (2000, "0001", "season-0001.csv"),
    (2001, "0102", "season-0102.csv"),
    (2002, "0203", "season-0203.csv"),
    (2003, "0304", "season-0304.csv"),
    (2004, "0405", "season-0405.csv"),
    (2005, "0506", "season-0506.csv"),
    (2006, "0607", "season-0607.csv"),
    (2007, "0708", "season-0708.csv"),
    (2008, "0809", "season-0809.csv"),
    (2009, "0910", "season-0910.csv"),
    (2010, "1011", "season-1011.csv"),
    (2011, "1112", "season-1112.csv"),
    (2012, "1213", "season-1213.csv"),
    (2013, "1314", "season-1314.csv"),
    (2014, "1415", "season-1415.csv"),
    (2015, "1516", "season-1516.csv"),
    (2016, "1617", "season-1617.csv"),
    (2017, "1718", "season-1718.csv"),
    (2018, "1819", "season-1819.csv"),
    (2019, "1920", "season-1920.csv"),
    (2020, "2021", "season-2021.csv"),
    (2021, "2122", "season-2122.csv"),
    (2022, "2223", "season-2223.csv"),
    (2023, "2324", "season-2324.csv"),
    (2024, "2425", "season-2425.csv"),
]


def download_seasons():
    for start_year, code, filename in SEASONS:
        save_path = os.path.join(SEASONS_DIR, filename)

        if os.path.exists(save_path):
            print(f"  - Already exists, skipping: {filename}")
            continue

        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

        try:
            df = pd.read_csv(url, encoding="latin1", on_bad_lines="skip")
            cols = [c for c in FD_COLUMNS if c in df.columns]
            df = df[cols].copy()
            df["Season"] = f"{start_year}/{str(start_year + 1)[-2:]}"
            df["Date"]   = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["FTR"])
            df.to_csv(save_path, index=False)
            print(f"  ✓ Downloaded {filename}  ({len(df)} matches)")
        except Exception as e:
            print(f"  ✗ Failed {filename}: {e}")

        time.sleep(0.5)


def combine_seasons() -> pd.DataFrame:
    dfs = []
    for _, _, filename in SEASONS:
        path = os.path.join(SEASONS_DIR, filename)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    print(f"\nTotal matches from football-data.co.uk: {len(combined)}")
    return combined


# ─────────────────────────────────────────────────────────────
# PART 2: Scrape xG data from Understat (2014/15 onward)
# ─────────────────────────────────────────────────────────────

TEAM_NAME_MAP = {
    "Manchester United":       "Man United",
    "Manchester City":         "Man City",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion":    "West Brom",
    "Nottingham Forest":       "Nott'm Forest",
    "Leicester":               "Leicester",
    "Brighton":                "Brighton",
    "West Ham":                "West Ham",
    "Sheffield United":        "Sheffield United",
    "Leeds":                   "Leeds",
    "Brentford":               "Brentford",
    "Luton":                   "Luton",
    "Burnley":                 "Burnley",
    "Ipswich":                 "Ipswich",
}


async def fetch_understat_all() -> pd.DataFrame:
    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        u = understat.Understat(session)
        dfs = []

        for year in range(2014, 2025):
            try:
                matches = await u.get_league_results("EPL", year)
                rows = []
                for m in matches:
                    if not m.get("isResult"):
                        continue
                    home = TEAM_NAME_MAP.get(m["h"]["title"], m["h"]["title"])
                    away = TEAM_NAME_MAP.get(m["a"]["title"], m["a"]["title"])
                    rows.append({
                        "Date":     m["datetime"][:10],
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "home_xg":  float(m["xG"]["h"]),
                        "away_xg":  float(m["xG"]["a"]),
                    })
                df = pd.DataFrame(rows)
                dfs.append(df)
                print(f"  ✓ Scraped Understat {year}/{year + 1}  ({len(df)} matches)")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"  ✗ Failed Understat {year}: {e}")

        return pd.concat(dfs, ignore_index=True)


def scrape_understat_all() -> pd.DataFrame:
    return asyncio.run(fetch_understat_all())


# ─────────────────────────────────────────────────────────────
# PART 3: Merge into master.csv
# ─────────────────────────────────────────────────────────────

def merge_sources(fd_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    master = fd_df.merge(
        xg_df[["Date", "HomeTeam", "AwayTeam", "home_xg", "away_xg"]],
        on=["Date", "HomeTeam", "AwayTeam"],
        how="left"
    )
    xg_coverage = master["home_xg"].notna().mean() * 100
    print(f"xG coverage: {xg_coverage:.1f}% of matches")
    return master


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: Downloading season CSVs...")
    print("=" * 50)
    download_seasons()

    print("\n" + "=" * 50)
    print("STEP 2: Combining all seasons...")
    print("=" * 50)
    fd_data = combine_seasons()
    fd_data.to_csv(f"{DATA_DIR}/football_data_raw.csv", index=False)
    print(f"Saved → Data/football_data_raw.csv")

    print("\n" + "=" * 50)
    print("STEP 3: Scraping Understat xG data...")
    print("=" * 50)
    xg_data = scrape_understat_all()
    xg_data.to_csv(f"{DATA_DIR}/understat_xg_raw.csv", index=False)
    print(f"Saved → Data/understat_xg_raw.csv")

    print("\n" + "=" * 50)
    print("STEP 4: Merging into master.csv...")
    print("=" * 50)
    master = merge_sources(fd_data, xg_data)
    master.to_csv(f"{DATA_DIR}/master.csv", index=False)
    print(f"Saved → Data/master.csv")

    print(f"\nDone. Shape: {master.shape}")
    print(f"\nColumns: {list(master.columns)}")
    print(f"\nSample:\n{master.head(3).to_string()}")