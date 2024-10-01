import os
import pandas as pd
import logging
import numpy as np

def setup_logging(log_dir):
    """Set up logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'integrity_check.log')
    
    # Configure logging to write to the file
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')  # 'w' mode overwrites the log file on each run
    
    # Disable console logging
    logging.getLogger().handlers = [logging.FileHandler(log_file)]

# Set up logging
setup_logging('logs')

# Define the max limits for each metric
MAX_LIMITS = {
    'energy_100g': 4300,
    'fat_100g': 95,
    'saturated-fat_100g': 55,
    'carbohydrates_100g': 95,
    'sugars_100g': 95,
    'sodium_100g': 3.0,
    'salt_100g': 6.0,
    'trans-fat_100g': 5,
    'cholesterol_100g': 500,
    'fiber_100g': 50,
    'proteins_100g': 90,
    'vitamin-a_100g': 30,
    'vitamin-c_100g': 50,
    'calcium_100g': 30,
    'iron_100g': 40,
}

# Define any multi-field dependency rules
DEPENDENCY_RULES = [
    ('saturated-fat_100g', 'fat_100g'),
    ('sodium_100g', 'salt_100g')
]

def log_outliers(df, metric, max_limit):
    """Log outlier information for a specific metric and replace outliers with NaN."""
    outliers = df[df[metric] > max_limit]
    for index, row in outliers.iterrows():
        logging.warning(f"Outlier detected in '{metric}': {row[metric]} exceeds the maximum limit of {max_limit}. "
                        f"Row details: {row.to_dict()}")
        # Set the outlier value to NaN instead of dropping the row
        df.at[index, metric] = np.nan
    return df

def log_dependency_violations(df, field_1, field_2):
    """Log dependency violations and set field_1 values to NaN if they exceed field_2."""
    violations = df[df[field_1] > df[field_2]]
    for index, row in violations.iterrows():
        logging.warning(f"Dependency violation: '{field_1}' ({row[field_1]}) > '{field_2}' ({row[field_2]}). "
                        f"Row details: {row.to_dict()}")
        # Set the violating value in field_1 to NaN instead of dropping the row
        df.at[index, field_1] = np.nan
    return df

def apply_integrity_checks(df, max_limits, dependency_rules):
    """Apply integrity checks on the DataFrame and replace outliers with NaN."""
    total_rows_before = len(df)

    # Single-field integrity checks
    for metric, max_limit in max_limits.items():
        if metric in df.columns:
            logging.info(f"Checking metric '{metric}' against max limit of {max_limit}...")
            df = log_outliers(df, metric, max_limit)

    # Multi-field dependency checks
    for field_1, field_2 in dependency_rules:
        if field_1 in df.columns and field_2 in df.columns:
            logging.info(f"Checking dependency: '{field_1}' should not exceed '{field_2}'...")
            df = log_dependency_violations(df, field_1, field_2)

    total_rows_after = len(df)
    rows_removed = total_rows_before - total_rows_after
    logging.info(f"Integrity check complete. Rows retained: {total_rows_after}. "
                 f"Values replaced with NaN due to violations or outliers.")

    return df

def run_integrity_check(df, log_dir='logs'):
    """Run the integrity check and update the DataFrame."""
    setup_logging(log_dir)
    df_cleaned = apply_integrity_checks(df.copy(), MAX_LIMITS, DEPENDENCY_RULES)
    return df_cleaned
