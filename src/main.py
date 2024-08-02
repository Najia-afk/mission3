from scripts.df_generator import get_dataset_directory, check_directory_exists, load_or_cache_dataframes
from scripts.df_metadata import display_metadata_dfs, create_metadata_dfs
from scripts.fetch_data_fields import fetch_and_compare_data_fields
import os
import pandas as pd

CACHE_DIR = 'data/cache'  # Directory to store cached DataFrames



def main():
    # Define the dataset directory
    notebook_directory = os.getcwd()
    dataset_directory = get_dataset_directory(notebook_directory)

    # Check if the dataset directory exists
    if not check_directory_exists(dataset_directory):
        print(f"Error: Directory '{dataset_directory}' does not exist.")
        return

    # Load DataFrames from cache or source files
    dfs = load_or_cache_dataframes(dataset_directory, CACHE_DIR)

    # Create metadata DataFrames
    metadata_dfs = create_metadata_dfs(dfs)

    # Optionally, show loaded DataFrames
    display_metadata_dfs(metadata_dfs)

    # Save the combined metadata DataFrame to a CSV file
    output_dir = os.path.join(notebook_directory, 'data')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    combined_metadata_path = os.path.join(output_dir, 'combined_metadata.csv')

    combined_metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys()).reset_index(level=0).rename(columns={'level_0': 'DataFrame'})
    combined_metadata.to_csv(combined_metadata_path, index=False)

    print(f"Metadata saved to {combined_metadata_path}")

    # Run the fetch and compare data fields script
    fetch_and_compare_data_fields()
    
if __name__ == "__main__":
    main()
