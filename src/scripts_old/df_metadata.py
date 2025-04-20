import pandas as pd

# Adjust display options to show all columns
pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)

def analyze_column(col):
    """Analyze a single column for data types, missing values, and potential issues like duplicates."""
    col_data = col.dropna()
    col_dtypes = col.dtypes
    
    if col_data.empty:
        col_type = 'NaN'
        fill_percentage = 0.0
        nan_percentage = 100.0
        bad_null_percentage = 0.0
        duplicate_percentage = 0.0
    else:
        type_counts = col_data.apply(lambda x: type(x).__name__).value_counts(normalize=True) * 100
        if len(type_counts) == 1:
            if type_counts.index[0] != 'NaN':
                max_length = col_data.apply(lambda x: len(str(x))).max()
                col_type = f"{type_counts.index[0]}({max_length})"
            else:
                col_type = type_counts.index[0]
        else:
            error_type_details = ', '.join([f"{t}: {p:.2f}%" for t, p in type_counts.items()])
            col_type = f"errorType({error_type_details})"
        
        fill_percentage = col_data.size / col.size * 100
        nan_percentage = col.isna().sum() / col.size * 100
        
        # Check for other forms of null values
        bad_null_count = col.isin(['', 'None', 'NULL', 'null']).sum()
        bad_null_percentage = bad_null_count / col.size * 100

        # Calculate duplicate percentage
        duplicate_percentage = col.dropna().duplicated().mean() * 100

    # Calculate missing percentage using missingno
    missing_percentage = col.isna().mean() * 100

    return {
        'Column Name': col.name,
        'Dtype': col_dtypes,
        'Type': col_type,
        'Fill Percentage': fill_percentage,
        'NaN Percentage': nan_percentage,
        'Bad Null Percentage': bad_null_percentage,
        'Duplicate Percentage': duplicate_percentage,
        'Missing Percentage': missing_percentage
    }

def analyze_dataframe(df):
    """Analyze the DataFrame by examining each column."""
    columns_info = []
    for col_name in df.columns:
        col_info = analyze_column(df[col_name])
        columns_info.append(col_info)
    df_info = pd.DataFrame(columns_info)
    return df_info, len(df)

def create_metadata_dfs(dfs):
    """Create metadata DataFrames for a dictionary of DataFrames."""
    metadata_dfs = {}
    for df_name, df in dfs.items():
        metadata_df, _ = analyze_dataframe(df)
        metadata_dfs[f'{df_name}'] = metadata_df
    return metadata_dfs

def display_metadata_dfs(metadata_dfs):
    """Display metadata DataFrames."""
    for name, metadata_df in metadata_dfs.items():
        print(f"Metadata for {name} {metadata_df.shape}:")
        print(metadata_df)
        print("\n")

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
