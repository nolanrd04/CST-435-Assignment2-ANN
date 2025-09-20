import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================
# Player Stat Extraction
# ==============================

def extract_top3_stats(input_file="relevant_data.csv", output_file="top3_data.csv"):
    # Load dataset
    df = pd.read_csv(input_file)

    # Stats we want to evaluate (exclude identity columns)
    stat_columns = [
        "age","player_height","player_weight","gp","pts","reb","ast",
        "net_rating","oreb_pct","dreb_pct","usg_pct","ts_pct","ast_pct"
    ]

    # Standardize stats (Z-scores)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[stat_columns]),
        columns=stat_columns,
        index=df.index
    )

    # Identify top 3 above-average stats for each player
    top_stats = []
    for idx, row in df_scaled.iterrows():
        # Keep only stats above average (positive z-scores)
        above_avg = {col: val for col, val in row.items() if val > 0}

        # Sort descending by z-score (best stats first)
        sorted_stats = sorted(above_avg.items(), key=lambda x: x[1], reverse=True)

        # Extract top 3 stat names
        top3 = [s[0] for s in sorted_stats[:3]]

        # Pad with "None" if player has fewer than 3 above-average stats
        while len(top3) < 3:
            top3.append("None")

        top_stats.append([df.loc[idx, "player_name"]] + top3)

    # Create final DataFrame
    df_top3 = pd.DataFrame(top_stats, columns=["player_name","top1","top2","top3"])

    # Save to CSV
    df_top3.to_csv(output_file, index=False)
    print(f"✅ Top 3 stats saved to {output_file}")


# ==============================
# Quadruplet Builder
# ==============================

def create_top3_quad(input_file="top3_data.csv", output_file="top3_quad.csv"):
    # Load the top 3 stats dataset
    df = pd.read_csv(input_file)

    # Create quadruplet strings for each player
    # Format: (player_name, top1, top2, top3)
    quadruplets = df.apply(
        lambda row: f"({row['player_name']}, {row['top1']}, {row['top2']}, {row['top3']})", axis=1
    )

    # Put into a single-column DataFrame
    df_quad = pd.DataFrame(quadruplets, columns=["quadruplet"])

    # Save to CSV
    df_quad.to_csv(output_file, index=False)
    print(f"✅ Quadruplets saved to {output_file}")


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    # Extract top 3 stats per player
    extract_top3_stats()

    # Create quadruplet dataset
    create_top3_quad()
