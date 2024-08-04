import json
import pandas as pd
import re

# Load the config.json
config_path = 'src/config/config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

# Function to get the expected type from the config
def get_expected_type(column_name, config):
    # Check for specific fields first
    for group, group_data in config.items():
        if 'fields' in group_data and column_name in group_data['fields']:
            return group_data['fields'][column_name]['type']
    
    # Check patterns if no specific field is found
    for pattern, type_name in config['field_patterns']['patterns'].items():
        if re.search(pattern, column_name):
            return type_name
    
    return None

# Function to check and enrich the metadata DataFrame
def enrich_metadata(metadata_df, base_df, config):
    enriched_data = []
    error_details = []

    for index, row in metadata_df.iterrows():
        column_name = row['Column Name']
        dtype = row['Dtype']
        
        # Get the expected type from config
        expected_type = get_expected_type(column_name, config)
        
        # Check if the type matches
        if expected_type:
            type_match = dtype == expected_type
            if not type_match:
                # Identify rows with mismatched data types
                for i, value in base_df[column_name].iteritems():
                    if not isinstance(value, eval(expected_type.title())):
                        error_details.append({
                            'DataFrame': row['DataFrame'],
                            'Column Name': column_name,
                            'Row Index': i,
                            'Value': value,
                            'Expected Type': expected_type,
                            'Actual Type': type(value).__name__
                        })
        else:
            expected_type = 'Unknown'
            type_match = False

        enriched_data.append({
            **row,
            'Expected Type': expected_type,
            'Type Match': type_match
        })

    enriched_df = pd.DataFrame(enriched_data)
    error_df = pd.DataFrame(error_details)
    
    return enriched_df, error_df

# Load the combined_metadata DataFrame
combined_metadata = pd.read_csv('data/combined_metadata.csv')

# Load the base DataFrame
base_df_path = 'data/fr.openfoodfacts.org.products.csv'
base_df = pd.read_csv(base_df_path, delimiter='\t')

# Enrich the metadata
combined_metadata_enriched, error_df = enrich_metadata(combined_metadata, base_df, config)

# Save the enriched metadata to a new CSV file
combined_metadata_enriched_path = 'data/combined_metadata_enriched.csv'
combined_metadata_enriched.to_csv(combined_metadata_enriched_path, index=False)

# Save the error details to a new CSV file
error_details_path = 'data/data_quality_issues.csv'
error_df.to_csv(error_details_path, index=False)

print(f"Enriched metadata saved to '{combined_metadata_enriched_path}'")
print(f"Data quality issues saved to '{error_details_path}'")
