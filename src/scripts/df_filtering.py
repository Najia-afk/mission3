import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import seaborn as sns
import logging
from sklearn.cluster import DBSCAN
from matplotlib.patches import Ellipse
from matplotlib.collections import PathCollection

# Set up logging
logging.basicConfig(level=logging.INFO)



def remove_url_column(df):
    if 'url' in df.columns:
        df.drop(columns=['url'], inplace=True)
        logging.info("Column 'url' has been removed")

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
        df.drop(columns=[timestamp_column], inplace=True)
        logging.info(f"No significant discrepancies found, '{timestamp_column}' has been dropped.")

def check_field_frequency(df, fields, temp_dir, graph_dir, generic_name):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    df[f'{generic_name}_combination'] = df.apply(lambda x: tuple(x[field] for field in fields), axis=1)
    combination_counts = df[f'{generic_name}_combination'].value_counts()

    combination_log_path = os.path.join(temp_dir, f'{generic_name}_combination_log.csv')
    combination_counts.to_csv(combination_log_path, header=['Frequency'])

    first_field = fields[0]
    field_groupings = df.groupby(first_field)[fields[1:]].nunique()
    frequency_entries = field_groupings[(field_groupings > 1).any(axis=1)]
    
    frequency_temp_path = os.path.join(temp_dir, f'frequency_{generic_name}_mappings.csv')
    frequency_entries.to_csv(frequency_temp_path)

    draw_histogram_for_field_combinations(combination_counts, graph_dir, generic_name)
    
    logging.info(f"Check the {generic_name} combination file for more details about fields frequency.")

def draw_histogram_for_field_combinations(combination_counts, graph_dir, generic_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(combination_counts, kde=True, bins=20, color='blue', alpha=0.6)
    
    mean_val = combination_counts.mean()
    median_val = combination_counts.median()
    std_val = combination_counts.std()
    outlier_threshold = mean_val + 3 * std_val
    num_outliers = (combination_counts > outlier_threshold).sum()

    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='-', label=f'Median: {median_val:.2f}')
    plt.axvline(mean_val + std_val, color='b', linestyle=':', label=f'Std Dev (+): {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='b', linestyle=':', label=f'Std Dev (-): {mean_val - std_val:.2f}')

    if num_outliers > 0:
        plt.text(outlier_threshold, plt.ylim()[1] * 0.9, f'{num_outliers} outliers detected', color='black')

    plt.title(f'Histogram of {generic_name} Combination Frequencies (All Data)')
    plt.xlabel('Frequency')
    plt.ylabel('Count of Combinations')
    plt.legend()
    plt.tight_layout()

    file_name = f"{generic_name}_combination_histogram_all_data.png"
    plt.savefig(os.path.join(graph_dir, file_name))
    plt.close()

    clipped_combination_counts = combination_counts[combination_counts <= outlier_threshold]
    plt.figure(figsize=(10, 6))
    sns.histplot(clipped_combination_counts, kde=True, bins=20, color='green', alpha=0.6)
    
    clipped_mean_val = clipped_combination_counts.mean()
    clipped_median_val = clipped_combination_counts.median()
    clipped_std_val = clipped_combination_counts.std()

    plt.axvline(clipped_mean_val, color='r', linestyle='--', label=f'Mean: {clipped_mean_val:.2f}')
    plt.axvline(clipped_median_val, color='g', linestyle='-', label=f'Median: {clipped_median_val:.2f}')
    plt.axvline(clipped_mean_val + clipped_std_val, color='b', linestyle=':', label=f'Std Dev (+): {clipped_mean_val + clipped_std_val:.2f}')
    plt.axvline(clipped_mean_val - clipped_std_val, color='b', linestyle=':', label=f'Std Dev (-): {clipped_mean_val - clipped_std_val:.2f}')

    plt.title(f'Histogram of {generic_name} Combination Frequencies (Without Outliers)')
    plt.xlabel('Frequency')
    plt.ylabel('Count of Combinations')
    plt.legend()
    plt.tight_layout()

    clipped_file_name = f"{generic_name}_combination_histogram_without_outliers.png"
    plt.savefig(os.path.join(graph_dir, clipped_file_name))
    plt.close()

    logging.info(f"{num_outliers} outliers were excluded from the second plot.")



def process_dataframe(df, log_dir='logs', temp_dir='temp', graph_dir='graph'):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    #remove_url_column(df)

    # Optimize date columns
    #check_datetime_consistency(df, 'created_t', 'created_datetime', log_dir='logs')
    #check_datetime_consistency(df, 'last_modified_t', 'last_modified_datetime', log_dir='logs')
    
    # Field frequency checks
    checks = [
        (['countries', 'countries_tags', 'countries_fr'], 'countries'),
        (['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n'], 'ingredients_palm_oil'),
        (['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'], 'nutrition'),
        (['brands_tags', 'brands'], 'brands'),
        (['additives_n', 'additives'], 'additives'),
        (['states', 'states_tags', 'states_fr'], 'states')
    ]
    for fields, generic_name in checks:
        check_field_frequency(df, fields, temp_dir, graph_dir, generic_name)

    return df