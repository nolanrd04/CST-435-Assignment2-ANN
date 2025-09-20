import pandas as pd
import random

# ==============================
# Data Cleaning Function
# ==============================

def clean_data(
    input_file="all_seasons.csv",
    output_file="relevant_data.csv",
    start_year=None,
    num_players=100
):
    # Load dataset
    df = pd.read_csv(input_file)

    # Extract start year from season column (e.g., "1996-97" → 1996)
    df["season_start"] = df["season"].str.split("-").str[0].astype(int)

    # If using a 5-year window
    if start_year is not None:
        end_year = start_year + 4
    else:
        # Pick random valid start year if none provided
        max_year = df["season_start"].max()
        min_year = df["season_start"].min()
        start_year = random.randint(min_year, max_year - 4)
        end_year = start_year + 4
        print(f"🎲 Randomly selected start year: {start_year}")

    # Filter rows within the 5-year window
    df = df[(df["season_start"] >= start_year) & (df["season_start"] <= end_year)]

    # Keep only relevant columns
    relevant_columns = [
        "player_name",
        "team_abbreviation",
        "age",
        "player_height",
        "player_weight",
        "gp",
        "pts",
        "reb",
        "ast",
        "net_rating",
        "oreb_pct",
        "dreb_pct",
        "usg_pct",
        "ts_pct",
        "ast_pct",
        "season"
    ]
    df = df[relevant_columns]

    # Randomly sample players (default 100)
    if num_players is not None and num_players < len(df):
        df = df.sample(n=num_players, random_state=42)  # fixed seed for reproducibility

    # Save filtered dataset
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data ({start_year}-{end_year}, {len(df)} players) saved to {output_file}")


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    # Default run → random 5-year window, 100 players
    clean_data()

    # Custom run → preset start year and 200 players
    clean_data(start_year=2000, num_players=200, output_file="relevant_data_custom.csv")
