# src/scripts/df_pnns_group.py

import json

def standardize_pnns_groups(json_file_path):
    """
    Standardizes the pnns_groups_1 and pnns_groups_2 combinations based on the most popular combination.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Process the combinations to standardize them
    for key, value in data.items():
        combinations = value["combinations"]
        most_popular_combination = max(combinations, key=lambda x: float(x['frequency']))
        standard_pnns_groups_1, standard_pnns_groups_2 = most_popular_combination["combination"]

        for combination in combinations:
            combination["combination"][0] = standard_pnns_groups_1
            combination["combination"][1] = standard_pnns_groups_2

        data[key]["combinations"] = combinations

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"pnns_groups_1 and pnns_groups_2 combinations have been standardized.")
