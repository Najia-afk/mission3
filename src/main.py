from scripts.df_generator import get_dataset_directory, check_directory_exists, load_or_cache_dataframes
from scripts.df_metadata import display_metadata_dfs, create_metadata_dfs
from scripts.fetch_data_fields import fetch_and_compare_data_fields
from scripts.build_data_fields_config import build_data_fields_config
from scripts.df_metada_enriched import enrich_metadata_df
import os, json
import pandas as pd



def main():
    CACHE_DIR = 'data/cache'  # Directory to store cached DataFrames

    # Define the dataset directory
    notebook_directory = os.getcwd()
    dataset_directory = get_dataset_directory(notebook_directory)

    # Check if the dataset directory exists
    if not check_directory_exists(dataset_directory):
        print(f"Error: Directory '{dataset_directory}' does not exist.")
        return

    # Optionally, you can define a list of specific files to process
    # specific_files = ['file1.csv', 'file2.csv']
    specific_files = None  # Set to None to process all files

    # Load DataFrames from cache or source files
    dfs = load_or_cache_dataframes(dataset_directory, CACHE_DIR, file_list=specific_files, separator='\t')

    # Check if DataFrames are loaded
    if not dfs:
        print("No DataFrames were loaded. Exiting.")
        return

    print(f"Loaded DataFrames: {list(dfs.keys())}")

    # Create metadata DataFrames
    metadata_dfs = create_metadata_dfs(dfs)

    # Check if metadata DataFrames were created
    if not metadata_dfs:
        print("No metadata DataFrames were created. Exiting.")
        return

    print(f"Created Metadata DataFrames: {list(metadata_dfs.keys())}")

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

    # Build the config file
    build_data_fields_config()

    # Load the config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'data_fields_config.json')

    with open(config_path, 'r') as file:
        config = json.load(file)
        
    # Load your metadata DataFrame (e.g., combined_metadata)
    metadata_path = os.path.join(script_dir, 'data', 'combined_metadata.csv')
    metadata_df = pd.read_csv(metadata_path)

    # Enrich the metadata DataFrame
    enriched_metadata_df = enrich_metadata_df(metadata_df, config)

    # Save the enriched metadata DataFrame
    enriched_metadata_path = os.path.join(script_dir, 'data', 'combined_metadata_enriched.csv')
    enriched_metadata_df.to_csv(enriched_metadata_path, index=False)

    print(f"Enriched metadata saved to {enriched_metadata_path}")

if __name__ == "__main__":
    main()
