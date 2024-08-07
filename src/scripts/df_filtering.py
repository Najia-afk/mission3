def filter_metadata_and_dataframes(combined_metadata, dfs):
    """
    Filters combined_metadata to drop rows where 'Fill Percentage' < 50, and then filters 
    the related DataFrames to keep only the columns that remain in the filtered metadata.
    """
    # Filter out rows in combined_metadata where 'Fill Percentage' < 50
    combined_metadata = combined_metadata[combined_metadata['Fill Percentage'] >= 50].copy()
    
    # Iterate over the filtered metadata to update the DataFrames
    for df_name in combined_metadata['DataFrame'].unique():
        
        # Get the relevant DataFrame
        if df_name in dfs:
            df = dfs[df_name]
            # Get the columns to keep based on the filtered metadata
            columns_to_keep = combined_metadata[combined_metadata['DataFrame'] == df_name]['Column Name'].tolist()
            # Filter the DataFrame
            filtered_df = df[columns_to_keep]
            # Replace the original DataFrame with the filtered one
            dfs[df_name] = filtered_df
            print(f"Updated DataFrame '{df_name}' to retain only relevant columns.")
        else:
            print(f"DataFrame '{df_name}' not found in the provided DataFrames.")
    
    return combined_metadata, dfs

