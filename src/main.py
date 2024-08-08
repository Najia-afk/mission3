from scripts.df_generator import get_dataset_directory, check_directory_exists, load_or_cache_dataframes, show_loaded_dfs
from scripts.df_metadata import display_metadata_dfs, create_metadata_dfs, enrich_metadata_df
from scripts.fetch_data_fields import fetch_and_compare_data_fields
from scripts.build_data_fields_config import build_data_fields_config
from scripts.df_filtering import filter_metadata_and_dataframes, process_dataframe
import os, json
import pandas as pd
import gc
import numpy as np

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
    specific_files = ['fr.openfoodfacts.org.products.csv']  # Set to None to process all files

    # Load DataFrames from cache or source files
    dfs = load_or_cache_dataframes(dataset_directory, CACHE_DIR, file_list=specific_files, separator='\t')

    # Check if DataFrames are loaded
    if not dfs:
        print("No DataFrames were loaded. Exiting.")
        return

    print(f"Loaded DataFrames: {list(dfs.keys())}")

    show_loaded_dfs(dfs, df_names=None)

    # Create metadata DataFrames
    metadata_dfs = create_metadata_dfs(dfs)

    # Check if metadata DataFrames were created
    if not metadata_dfs:
        print("No metadata DataFrames were created. Exiting.")
        return

    print(f"Created Metadata DataFrames: {list(metadata_dfs.keys())}")

    # Optionally, show loaded DataFrames
    display_metadata_dfs(metadata_dfs)

    # Combine metadata into a single DataFrame
    combined_metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys()).reset_index(level=0).rename(columns={'level_0': 'DataFrame'})

    # Run the fetch and compare data fields script
    fetch_and_compare_data_fields()

    # Build the config file
    build_data_fields_config()

    # Load the config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'data_fields_config.json')

    with open(config_path, 'r') as file:
        config = json.load(file)

    # Enrich the metadata DataFrame
    combined_metadata = enrich_metadata_df(combined_metadata, config)

    # Filter metadata and the related DataFrames
    combined_metadata, filtered_dfs = filter_metadata_and_dataframes(combined_metadata, dfs)

    # Save the combined metadata DataFrame to a CSV file
    output_dir = os.path.join(notebook_directory, 'data')
    os.makedirs(output_dir, exist_ok=True)

    combined_metadata_path = os.path.join(output_dir, 'combined_metadata.csv')
    combined_metadata.to_csv(combined_metadata_path, index=False)
    print(f"combined_metadata {combined_metadata.shape} has been saved or updated.")

    # Save the filtered DataFrames to individual CSVs or as needed
    for df_name, df in filtered_dfs.items():
        df_output_path = os.path.join(dataset_directory, f'filtered_{df_name}.csv')
        df.to_csv(df_output_path, index=False)
        print(f"Filtered DataFrame 'filtered_{df_name}' {df.shape} has been saved")

    # No need for all the dfs to be in RAM anymore
    del dfs
    gc.collect()  # Force garbage collection

    # Cache the filtered DataFrames
    specific_files = ['filtered_fr.openfoodfacts.org.products.csv']
    dfs = load_or_cache_dataframes(dataset_directory, CACHE_DIR, file_list=specific_files)


    df_name = 'filtered_fr.openfoodfacts.org.products'
    if df_name in dfs:
        processed_df = process_dataframe(dfs[df_name])
        df_output_path = os.path.join(dataset_directory, f'processed_{df_name}.csv')
        processed_df.to_csv(df_output_path, index=False)
        print(f"Filtered DataFrame 'processed_{df_name}' {processed_df.shape} has been saved")


    else:
        print(f"DataFrame '{df_name}' not found in the loaded DataFrames.")

if __name__ == "__main__":
    main()
