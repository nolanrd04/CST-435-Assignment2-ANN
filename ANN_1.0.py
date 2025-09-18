import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


########### IMPORT DATA ###########
# Get the folder where this script is located. 
# The csv file and script file should be in the same folder to make it easy.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to the CSV in the same folder.
file_path = os.path.join(script_dir, "all_seasons.csv")

df = pd.read_csv(file_path)
# print(df.head())

########### Feature information ###########

# 1. player_name — Name of the player
# 2. team_abbreviation — Abbreviated name of the team the player played for (at the end of the season)
# 3. age — Age of the player
# 4. player_height — Height of the player (in centimeters)
# 5. player_weight — Weight of the player (in kilograms)
# 6. college — Name of the college the player attended
# 7. country — Name of the country the player was born in (not necessarily the nationality)
# 8. draft_year — The year the player was drafted
# 9. draft_round — The draft round the player was picked
# 10. draft_number — The number at which the player was picked in his draft round
# 11. gp — Games played throughout the season
# 12. pts — Average number of points scored
# 13. reb — Average number of rebounds grabbed
# 14. ast — Average number of assists distributed
# 15. net_rating — Team's point differential per 100 possessions while the player is on the court
# 16. oreb_pct — Percentage of available offensive rebounds the player grabbed while he was on the floor
# 17. dreb_pct — Percentage of available defensive rebounds the player grabbed while he was on the floor
# 18. usg_pct — Percentage of team plays used by the player while he was on the floor ((FGA + Possession Ending FTA + TO) / POSS)
# 19. ts_pct — Measure of the player's shooting efficiency that takes into account free throws, 2 and 3 point shots (PTS / (2*(FGA + 0.44 * FTA)))
# 20. ast_pct — Percentage of teammate field goals the player assisted while he was on the floor
# 21. season — NBA season

########### CLEAN DATA ###########
# saving player names for later output:
player_names = df["player_name"]

# drop non-numerical features
X = df.drop(columns=["player_name", "team_abbreviation", "season", "pts"]) #pts is sample target

########### DROPPING and ENCODING INSIGNIFICANT FEATURES ###########
y = df["pts"]

# Identify categorical + numeric features
categorical_features = ["college", "country", "draft_year", "draft_round", "draft_number"]
numeric_features = [col for col in X.columns if col not in categorical_features]

# Use OrdinalEncoder instead of OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

########### RANDOM FOREST PIPELINE ###########
rf = RandomForestRegressor(n_estimators=20, random_state=42)

# Build pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                       ("rf", rf)])

# Fit model
model.fit(X, y)

########### FEATURE IMPORTANCE ###########
# Get feature names after encoding
ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_features = ohe.get_feature_names_out(categorical_features)
all_features = list(encoded_cat_features) + numeric_features

# Get feature importance from Random Forest
importances = model.named_steps["rf"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(importance_df.head(20))

########### PLOT ###########
plt.figure(figsize=(10,8))
plt.barh(importance_df["Feature"].head(20), importance_df["Importance"].head(20))
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances from Random Forest for Scoring POINTS")
plt.show()