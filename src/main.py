from scripts.df_generator import get_dataset_directory, check_directory_exists, load_and_process_data
from scripts.df_metadata import display_metadata_dfs, create_metadata_dfs
import os
import pandas as pd

CACHE_DIR = 'data/cache'  # Directory to store cached DataFrames

def load_or_cache_dataframes(dataset_directory):
    # If cache exists, load from cache
    if os.path.exists(CACHE_DIR):
        print("Loading DataFrames from cache...")
        dfs = {file_name: pd.read_pickle(os.path.join(CACHE_DIR, file_name)) for file_name in os.listdir(CACHE_DIR)}
    else:
        # Load and preprocess files into DataFrames
        print("Loading DataFrames from source files...")
        dfs = load_and_process_data(dataset_directory)
        
        # Save DataFrames to cache for future use
        os.makedirs(CACHE_DIR, exist_ok=True)
        for name, df in dfs.items():
            df.to_pickle(os.path.join(CACHE_DIR, name))
    
    return dfs

def main():
    # Define the dataset directory
    notebook_directory = os.getcwd()
    dataset_directory = get_dataset_directory(notebook_directory)

    # Check if the dataset directory exists
    if not check_directory_exists(dataset_directory):
        print(f"Error: Directory '{dataset_directory}' does not exist.")
        return

    # Load DataFrames from cache or source files
    dfs = load_or_cache_dataframes(dataset_directory)

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

if __name__ == "__main__":
    main()
