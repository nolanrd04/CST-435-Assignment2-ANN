import os
import pandas as pd

########### IMPORT DATA ##########
# Get the folder where this script is located. 
# The csv file and script file should be in the same folder to make it easy.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to the CSV in the same folder.
file_path = os.path.join(script_dir, "all_seasons.csv")

df = pd.read_csv(file_path)
print(df.head())