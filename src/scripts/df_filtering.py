import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def filter_metadata_and_dataframes(combined_metadata, dfs, min_fill_percentage=40):
    """
    Filters combined_metadata to drop rows where 'Fill Percentage' < min_fill_percentage,
    and then filters the related DataFrames to keep only the columns that remain in the filtered metadata.
    
    Parameters:
    - combined_metadata: DataFrame containing the metadata.
    - dfs: Dictionary of DataFrames to be filtered.
    - min_fill_percentage: Minimum fill percentage to filter the metadata (default is 40).
    
    Returns:
    - Filtered combined_metadata and DataFrames.
    """
    combined_metadata = combined_metadata[combined_metadata['Fill Percentage'] >= min_fill_percentage].copy()
    
    for df_name in combined_metadata['DataFrame'].unique():
        if df_name in dfs:
            df = dfs[df_name]
            columns_to_keep = combined_metadata[combined_metadata['DataFrame'] == df_name]['Column Name'].tolist()
            dfs[df_name] = df[columns_to_keep]  # Filter the DataFrame
            logging.info(f"Updated DataFrame '{df_name}' to retain only relevant columns.")
        else:
            logging.warning(f"DataFrame '{df_name}' not found in the provided DataFrames.")
    
    return combined_metadata, dfs

def check_datetime_consistency(df, timestamp_column, datetime_column, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{datetime_column}_log.csv')
    discrepancies = []

    with open(log_path, 'w') as log_file:
        log_file.write(f'Row,{timestamp_column},{datetime_column}\n')
        for idx, row in df.iterrows():
            timestamp_value = pd.to_numeric(row[timestamp_column], errors='coerce')
            datetime_value = pd.to_datetime(row[datetime_column], errors='coerce')

            if pd.notna(timestamp_value):
                timestamp_date = pd.to_datetime(timestamp_value, unit='s', errors='coerce').date()

            if pd.notna(datetime_value):
                datetime_date = datetime_value.date()

            if pd.isnull(datetime_value) and pd.notnull(timestamp_value):
                df.at[idx, datetime_column] = pd.to_datetime(timestamp_date)
            elif pd.notnull(timestamp_value) and pd.notnull(datetime_value) and timestamp_date != datetime_date:
                log_file.write(f"{idx},{timestamp_date},{datetime_date}\n")
                discrepancies.append((idx, timestamp_date, datetime_date))

    if discrepancies:
        logging.warning(f"Found {len(discrepancies)} discrepancies between '{timestamp_column}' and '{datetime_column}'. Please review the log before proceeding.")
    else:
        logging.info(f"No significant discrepancies found, '{timestamp_column}' can been dropped.")

def check_field_frequency(df, fields, temp_dir, generic_name):
    os.makedirs(temp_dir, exist_ok=True)

    df[f'{generic_name}_combination'] = df.apply(lambda x: tuple(x[field] for field in fields), axis=1)
    combination_counts = df[f'{generic_name}_combination'].value_counts()

    combination_log_path = os.path.join(temp_dir, f'{generic_name}_combination_log.csv')
    combination_counts.to_csv(combination_log_path, header=['Frequency'])
    
    logging.info(f"Check the {generic_name} combination file for more details about fields frequency.")

def process_dataframe(df, log_dir='logs', temp_dir='temp', datetime_checks=None, field_checks=None):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    if datetime_checks:
        for timestamp_column, datetime_column in datetime_checks:
            check_datetime_consistency(df, timestamp_column, datetime_column, log_dir=log_dir)
    
    if field_checks:
        for fields, generic_name in field_checks:
            check_field_frequency(df, fields, temp_dir, generic_name)

    return df