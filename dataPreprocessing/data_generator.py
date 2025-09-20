import csv
import random
import ast
import pandas as pd

# ==============================
# Team Evaluation Rules
# ==============================

def is_optimal_team(team):
    # Rule 1: Team cannot have duplicate players
    players = [t[0] for t in team]
    if len(players) != len(set(players)):
        return 0

    # Rule 2: Any player with >1 "nan" stat → suboptimal
    for t in team:
        nan_count = sum(1 for stat in t[1:] if stat == "nan")
        if nan_count > 1:
            return 0

    # Rule 3: No stat can be a "top stat" for >3 players
    stat_counts = {}
    for t in team:
        for stat in t[1:]:
            if stat == "nan":
                continue
            stat_counts[stat] = stat_counts.get(stat, 0) + 1

    if any(count > 3 for count in stat_counts.values()):
        return 0

    return 1


# ==============================
# Team Generation Functions
# ==============================

def load_quadruplets(input_file):
    quadruplets = []
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            # Remove parentheses and split by comma
            clean = row[0].strip("()")
            parts = [p.strip() for p in clean.split(",")]
            quadruplets.append(tuple(parts))
    return quadruplets



def generate_teams(quadruplets, num_teams=1000):
    # Generate random quintuplets and label them
    teams, labels = [], []
    for _ in range(num_teams):
        team = random.sample(quadruplets, 5)
        label = is_optimal_team(team)
        teams.append(team)
        labels.append(label)
    return teams, labels


def save_dataset(teams, labels, output_file):
    # Convert teams + labels to DataFrame and save
    df = pd.DataFrame({
        "team": [str(t) for t in teams],
        "label": labels
    })
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(teams)} teams to {output_file}")


# ==============================
# Modular Dataset Builder
# ==============================

def build_dataset(input_file, output_file, num_teams=1000):
    # Load player quadruplets
    quadruplets = load_quadruplets(input_file)

    # Generate labeled teams
    teams, labels = generate_teams(quadruplets, num_teams)

    # Save dataset
    save_dataset(teams, labels, output_file)


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    # Generate training dataset
    build_dataset("top3_quad.csv", "train_dataset.csv", num_teams=1000)

    # Generate testing dataset
    build_dataset("top3_quad.csv", "test_dataset.csv", num_teams=100)
