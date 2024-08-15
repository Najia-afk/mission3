import os, json, logging
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

def group_combinations_with_fuzzy(log_file, threshold=85):
    combination_log = pd.read_csv(log_file, index_col=0)
    combination_log.index = combination_log.index.map(safe_eval)

    total_frequency = combination_log['Frequency'].sum()
    grouped_results = {}

    for combination, row in combination_log.iterrows():
        frequency = row['Frequency']
        percentage = (frequency / total_frequency) * 100
        combination_key = tuple(combination)

        # Ensure that the combination_key has at least three elements
        if len(combination_key) < 3:
            logging.warning(f"Skipping combination with insufficient elements: {combination_key}")
            continue

        key_for_fuzzy = f"{combination_key[1]} {combination_key[2]}"

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

    for group in grouped_results.values():
        group["total_percentage"] = sum([item["combination_percentage"] for item in group["combinations"]])
        group["total_frequency"] = sum([item["frequency"] for item in group["combinations"]])

    return grouped_results

def save_grouped_results_to_json(grouped_results, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(grouped_results, json_file, indent=4, default=str)

def fuzzy_dataframe(temp_dir='temp', config_dir='config'):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Define the checks for different fields
    checks = [
        (['countries', 'countries_tags', 'countries_fr'], 'countries'),
        (['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n'], 'ingredients_palm_oil'),
        #(['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'], 'nutrition'),
        (['brands_tags', 'brands'], 'brands'),
        #(['additives_n', 'additives'], 'additives'),
        (['states', 'states_tags', 'states_fr'], 'states')
    ]
    
    for fields, generic_name in checks:
        log_file = os.path.join(temp_dir, f'{generic_name}_combination_log.csv')
        
        if not os.path.exists(log_file):
            logging.warning(f"Log file {log_file} does not exist. Skipping.")
            continue
        
        # Group combinations and calculate statistics from the log
        grouped_results = group_combinations_with_fuzzy(log_file, threshold=85)

        # Save the grouped structure to a JSON file
        json_file_path = os.path.join(config_dir, f'{generic_name}_grouped_results.json')
        save_grouped_results_to_json(grouped_results, json_file_path)
        
        logging.info(f"Grouped results for {generic_name} has been generated")

    logging.info("Fuzzy matching and grouping completed for all checks.")

# Example of calling the fuzzy_dataframe function
if __name__ == "__main__":
    fuzzy_dataframe()
