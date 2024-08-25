import os
import json
import logging
import pandas as pd
from fuzzywuzzy import fuzz, process
import ast

def safe_eval(x):
    try:
        x = x.replace('nan', 'None')  # Replace string 'nan' with 'None'
        return ast.literal_eval(x)
    except Exception as e:
        logging.error(f"Error evaluating string: {x}, error: {e}")
        return None

def group_combinations_with_fuzzy(log_file, threshold=85, generic_name=""):
    combination_log = pd.read_csv(log_file, index_col=0)
    combination_log.index = combination_log.index.map(safe_eval)

    total_frequency = combination_log['Frequency'].sum()
    grouped_results = {}

    logging.info(f"[{generic_name}] Total combinations to process: {len(combination_log)}")

    for i, (combination, row) in enumerate(combination_log.iterrows()):
        logging.info(f"[{generic_name}] Processing combination {i + 1}/{len(combination_log)}")
        
        frequency = row['Frequency']
        percentage = (frequency / total_frequency) * 100
        combination_key = tuple(combination)

        if len(combination_key) < 2:
            logging.warning(f"[{generic_name}] Skipping combination with insufficient elements: {combination_key}")
            continue

        key_for_fuzzy = f"{combination_key[1]} {combination_key[2] if len(combination_key) > 2 else ''}"

        matched_group = None
        if grouped_results:
            match_info = process.extractOne(key_for_fuzzy, grouped_results.keys(), scorer=fuzz.ratio)
            if match_info and match_info[1] >= threshold:
                matched_group = match_info[0]

        if matched_group:
            grouped_results[matched_group]["combinations"].append({
                "combination": combination_key,
                "combination_percentage": percentage,
                "frequency": frequency,
                "fuzzy_score": fuzz.ratio(" ".join(map(str, combination_key)), matched_group)
            })
        else:
            grouped_results[key_for_fuzzy] = {
                "combinations": [{
                    "combination": combination_key,
                    "combination_percentage": percentage,
                    "frequency": frequency,
                    "fuzzy_score": 100
                }],
                "total_percentage": percentage,
                "total_frequency": frequency
            }

    logging.info(f"[{generic_name}] Finished processing all combinations.")

    for group in grouped_results.values():
        group["total_percentage"] = sum([item["combination_percentage"] for item in group["combinations"]])
        group["total_frequency"] = sum([item["frequency"] for item in group["combinations"]])

    return grouped_results


def save_grouped_results_to_json(grouped_results, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(grouped_results, json_file, indent=4, default=str)

def fuzzy_dataframe(temp_dir='temp', config_dir='config', checks=None, threshold=85):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    for fields, generic_name in checks:
        log_file = os.path.join(temp_dir, f'{generic_name}_combination_log.csv')
        
        if not os.path.exists(log_file):
            logging.warning(f"[{generic_name}] Log file {log_file} does not exist. Skipping.")
            continue
        
        # Group combinations and calculate statistics from the log
        grouped_results = group_combinations_with_fuzzy(log_file, threshold=threshold, generic_name=generic_name)

        # Save the grouped structure to a JSON file
        json_file_path = os.path.join(config_dir, f'{generic_name}_grouped_results.json')
        save_grouped_results_to_json(grouped_results, json_file_path)
        
        logging.info(f"Grouped results for {generic_name} have been generated")

    logging.info("Fuzzy matching and grouping completed for all checks.")

# Example of calling the fuzzy_dataframe function from a Jupyter notebook
if __name__ == "__main__":
    fuzzy_dataframe()
