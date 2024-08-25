import os
import pandas as pd
import logging

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
    'energy_100g': 900,
    'fat_100g': 100,
    'saturated-fat_100g': 50,
    'carbohydrates_100g': 100,
    'sugars_100g': 100,
    'sodium_100g': 2.3,
    'salt_100g': 5.75,
    'trans-fat_100g': 55.33,
    'cholesterol_100g': 55.08,
    'fiber_100g': 99.49,
    'proteins_100g': 99.04,
    'vitamin-a_100g': 57.12,
    'vitamin-c_100g': 56.09,
    'calcium_100g': 56.03,
    'iron_100g': 56.21,
    # Add other limits if needed
}

# Define any multi-field dependency rules
DEPENDENCY_RULES = [
    ('saturated-fat_100g', 'fat_100g'),
    ('sodium_100g', 'salt_100g')
    # Add more dependencies if required
]

def log_outliers(df, metric, max_limit):
    """Log outlier information for a specific metric."""
    outliers = df[df[metric] > max_limit]
    for index, row in outliers.iterrows():
        logging.warning(f"Outlier detected in '{metric}': {row[metric]} exceeds the maximum limit of {max_limit}. "
                        f"Row details: {row.to_dict()}")
    return outliers

def log_dependency_violations(df, field_1, field_2):
    """Log and remove rows where field_1 exceeds field_2."""
    violations = df[df[field_1] > df[field_2]]
    for index, row in violations.iterrows():
        logging.warning(f"Dependency violation: '{field_1}' ({row[field_1]}) > '{field_2}' ({row[field_2]}). "
                        f"Row details: {row.to_dict()}")
    return violations

def apply_integrity_checks(df, max_limits, dependency_rules):
    """Apply integrity checks on the DataFrame and remove rows with outliers."""
    total_rows_before = len(df)

    # Single-field integrity checks
    for metric, max_limit in max_limits.items():
        if metric in df.columns:
            logging.info(f"Checking metric '{metric}' against max limit of {max_limit}...")
            outliers = log_outliers(df, metric, max_limit)
            if not outliers.empty:
                logging.info(f"Removing {len(outliers)} rows with '{metric}' values exceeding {max_limit}.")
                df = df[df[metric] <= max_limit]
            else:
                logging.info(f"No outliers found for '{metric}'.")

    # Multi-field dependency checks
    for field_1, field_2 in dependency_rules:
        if field_1 in df.columns and field_2 in df.columns:
            logging.info(f"Checking dependency: '{field_1}' should not exceed '{field_2}'...")
            violations = log_dependency_violations(df, field_1, field_2)
            if not violations.empty:
                logging.info(f"Removing {len(violations)} rows due to '{field_1}' > '{field_2}'.")
                df = df[df[field_1] <= df[field_2]]
            else:
                logging.info(f"No dependency violations found between '{field_1}' and '{field_2}'.")

    total_rows_after = len(df)
    rows_removed = total_rows_before - total_rows_after
    logging.info(f"Integrity check complete. Total rows removed: {rows_removed}. "
                 f"DataFrame now contains {total_rows_after} rows.")

    return df

def run_integrity_check(df, log_dir='logs'):
    """Run the integrity check and update the DataFrame."""
    setup_logging(log_dir)
    df_cleaned = apply_integrity_checks(df.copy(), MAX_LIMITS, DEPENDENCY_RULES)
    return df_cleaned
