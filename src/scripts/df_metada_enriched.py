import os
import json
import pandas as pd

# Load the config.json
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'data_fields_config.json')

with open(config_path, 'r') as file:
    config = json.load(file)

def enrich_metadata_df(metadata_df, config):
    """
    Enrich the metadata DataFrame by comparing detected types with config file and adding a column indicating match.
    """
    def get_config_type(column_name):
        # Traverse the config to find the type for the given column
        for section, section_data in config.items():
            if "fields" in section_data and column_name in section_data["fields"]:
                return section_data["fields"][column_name].get("type", "Unknown")
        return "Unknown"

    # Add a new column to metadata_df to indicate if the type matches with config
    metadata_df['Config Type'] = metadata_df['Column Name'].apply(get_config_type)
    metadata_df['Data Fields Match'] = metadata_df.apply(
        lambda row: "Yes" if row['Type'][:3] == row['Config Type'][:3] else "No", axis=1
    )
    
    return metadata_df

def enrich_metadata_dfs(metadata_dfs, config):
    """
    Enrich all metadata DataFrames in a dictionary.
    """
    enriched_metadata_dfs = {}
    for name, df in metadata_dfs.items():
        enriched_metadata_dfs[name] = enrich_metadata_df(df, config)
    return enriched_metadata_dfs

# This section runs the script when executed directly
if __name__ == "__main__":
    # Load your metadata DataFrame (e.g., combined_metadata)
    metadata_path = os.path.join(script_dir, '..', 'data', 'combined_metadata.csv')
    metadata_df = pd.read_csv(metadata_path)

    # Enrich the metadata DataFrame
    enriched_metadata_df = enrich_metadata_df(metadata_df, config)

    # Save the enriched metadata DataFrame
    enriched_metadata_path = os.path.join(script_dir, '..', 'data', 'combined_metadata_enriched.csv')
    enriched_metadata_df.to_csv(enriched_metadata_path, index=False)

    print(f"Enriched metadata saved to {enriched_metadata_path}")
