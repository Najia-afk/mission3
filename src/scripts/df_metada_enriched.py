
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


