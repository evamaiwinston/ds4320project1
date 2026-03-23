import requests
import random
import pandas as pd
import time
import logging
import os
from bs4 import BeautifulSoup

# Logging
os.makedirs('logs', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
logging.basicConfig(
    filename='logs/scrape_matches.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Constants
BASE_URL = 'https://fbref.com'
MLS_ID = 22
SEASONS = list(range(2018, 2026)) # LAFC seasons
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
REQUEST_DELAY = 6


def get_soup(url: str) -> BeautifulSoup:
    """Fetches the content of a URL and returns a BeautifulSoup object."""
    try:
        logger.info(f"Fetching URL: {url}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        time.sleep(REQUEST_DELAY + random.uniform(1, 3))
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        #return None
        raise


def scrape_fixtures(year:int) -> pd.DataFrame:
    """Scrapes match fixtures for a given MLS season year and returns a DataFrame."""
    url = f"{BASE_URL}/en/comps/{MLS_ID}/{year}/schedule/{year}-Major-League-Soccer-Scores-and-Fixtures"
    try:
        soup = get_soup(url)
        table = soup.find('table', {'id': 'sched_all'})
        if table is None:
            logger.warning(f"No fixture table found for {year}")
            return pd.DataFrame()  # Return empty DataFrame if no table found
        
        df = pd.read_html(str(table))[0]
        df['season'] = year

        # Drop rows where score is NaN
        df = df[df["Score"].notna()]
        df = df[df["Score"] != "Score"]

        # Extract home and away teams
        df[["home_score", "away_score"]] = df["Score"].str.split(" - ", expand=True)
   
        # Derive match outcome from home team's perspective
        df["outcome"] = df.apply(_compute_outcome, axis=1)
 
        # Tag competition type from the "Comp" or "Round" columns if present
        if "Comp" in df.columns:
            df["competition_type"] = df["Comp"]
        else:
            df["competition_type"] = "MLS Regular Season"
 
        logger.info(f"Season {year}: {len(df)} matches scraped")
        return df
 
    except Exception as e:
        logger.error(f"Error scraping fixtures for {year}: {e}")
        return pd.DataFrame()     


def _compute_outcome(row) -> str:
    """Return 'home_win', 'away_win', or 'draw' for a match row."""
    try:
        h = int(row["home_score"])
        a = int(row["away_score"])
        if h > a:
            return "home_win"
        elif a > h:
            return "away_win"
        else:
            return "draw"
    except (ValueError, TypeError):
        return "unknown"
    

def extract_teams(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a unique teams table from the home/away columns in fixtures.
    FBref calls these 'Home' and 'Away'.
    """
    home_teams = fixtures_df[["Home"]].rename(columns={"Home": "name"})
    away_teams = fixtures_df[["Away"]].rename(columns={"Away": "name"})
    teams = pd.concat([home_teams, away_teams]).drop_duplicates().dropna()
    teams = teams.reset_index(drop=True)
    teams["team_id"] = teams.index + 1
 
    # Tag LAFC for easy filtering downstream
    teams["is_lafc"] = teams["name"].str.contains("Los Angeles FC", case=False, na=False)
 
    logger.info(f"Extracted {len(teams)} unique teams")
    return teams

def build_seasons_table() -> pd.DataFrame:
    """Build a static seasons reference table for 2018-2025."""
    seasons = []
    for year in SEASONS:
        seasons.append({
            "season_id": year - 2017,  # 1-indexed
            "year": year,
            "competition": "MLS",
            "num_teams": 24 if year < 2020 else 26 if year < 2023 else 29,
        })
    return pd.DataFrame(seasons)



def main():
    logger.info("Running scrape_matches.py....")
 
    all_fixtures = []
    for year in SEASONS:
        logger.info(f"Processing season {year}")
        df = scrape_fixtures(year)
        if not df.empty:
            all_fixtures.append(df)
 
    if not all_fixtures:
        logger.error("No fixture data collected. Exiting.")
        return
 
    fixtures_df = pd.concat(all_fixtures, ignore_index=True)
 
    # Build matches table 
    match_cols = {
        "Wk": "matchweek",
        "Date": "kickoff_date",
        "Time": "kickoff_time",
        "Home": "home_team",
        "Away": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "outcome": "outcome",
        "Attendance": "attendance",
        "Venue": "venue_name",
        "Referee": "referee_name",
        "season": "season_year",
        "competition_type": "competition_type",
    }
    available_cols = {k: v for k, v in match_cols.items() if k in fixtures_df.columns}
    matches_df = fixtures_df[list(available_cols.keys())].rename(columns=available_cols)
    matches_df["match_id"] = matches_df.index + 1
 
    # Build supporting tables 
    teams_df = extract_teams(fixtures_df)
    seasons_df = build_seasons_table()
 
    # Save outputs
    matches_df.to_parquet("data/raw/matches.parquet", index=False)
    teams_df.to_parquet("data/raw/teams.parquet", index=False)
    seasons_df.to_parquet("data/raw/seasons.parquet", index=False)

    logger.info(f"Saved {len(matches_df)} matches, {len(teams_df)} teams, {len(seasons_df)} seasons")
    logger.info("Done running scrape_matches.py")
    print(f"Done. {len(matches_df)} matches across {len(SEASONS)} seasons.")


if __name__ == "__main__":
    main()