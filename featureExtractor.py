from dataPreprocessing.cleaner import clean_data
from dataPreprocessing.preprocess import extract_top3_stats, create_top3_quad
from dataPreprocessing.data_generator import build_dataset

def preprocessData(start_year=2018, train_size=1000, test_size=100, num_players=100):
    # Step 1: Isolate relevant data
    clean_data(input_file="datasets/all_seasons.csv", output_file="datasets/relevant_data.csv", start_year=start_year, num_players=num_players)

    # Step 2: Extract top 3 stats per player
    extract_top3_stats(input_file="datasets/relevant_data.csv", output_file="datasets/top3_data.csv")

    # Step 3: Create quadruplets of player names and their top 3 stats
    create_top3_quad(input_file="datasets/top3_data.csv", output_file="datasets/top3_quad.csv")

    # Step 4: Build dataset of teams with labels
    build_dataset(input_file="datasets/top3_quad.csv", output_file="train_dataset.csv", num_teams=train_size)
    build_dataset(input_file="datasets/top3_quad.csv", output_file="test_dataset.csv", num_teams=test_size)

if __name__ == "__main__":
    preprocessData()