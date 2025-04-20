import os
import json
import re


# Initialize patterns for generalities with both type and sub-type
generalities = {
    "_t$": ("unix_timestamp", None),
    "_datetime$": ("iso8601_datetime", None),
    "_tags$": ("string", "tags"),
    "_100g$": ("float", None),
    "_serving$": ("float", None),
    "_fr$": ("string", "tags")
}

# Function to parse fields from the text
def parse_fields(description_text):
    field_pattern = re.compile(r"(?P<field_name>\S+)\s*(?:\:\s*(?P<description>.+))?")
    parsed_fields = {}
    start_parsing = False 

    current_section = None
    for line in description_text.splitlines():
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        if line.lower().startswith("list of fields:"):
            start_parsing = True
            continue

        if start_parsing:
            if line.startswith('#'):
                current_section = line.strip('# ').lower().replace(' ','').replace(':','')
                if current_section not in parsed_fields:
                    parsed_fields[current_section] = {}
            else:
                match = field_pattern.match(line)
                if match:
                    field_name = match.group("field_name")
                    description = match.group("description")
                    if current_section:
                        parsed_fields[current_section][field_name] = description.strip() if description else None
                    else:
                        print(f"Warning: Field '{field_name}' is not associated with any section.")

    return parsed_fields

# Function to update config with parsed fields
def update_config_with_fields(config, parsed_fields):
    for section, fields in parsed_fields.items():
        if section not in config:
            config[section] = {"description": "", "fields": {}}

        for field_name, field_description in fields.items():
            if field_name not in config[section]["fields"]:
                config[section]["fields"][field_name] = {"type": "string"}  # Default type
            if field_description:
                config[section]["fields"][field_name]["description"] = field_description

    return config

# Function to apply generalities to the config
def apply_generalities(config, generalities):
    for section, section_data in config.items():
        if "fields" in section_data:
            for field_name, field_data in section_data["fields"].items():
                for pattern, (inferred_type, inferred_sub_type) in generalities.items():
                    if re.search(pattern, field_name):
                        field_data["type"] = inferred_type
                        if inferred_sub_type:
                            field_data["sub-type"] = inferred_sub_type
    return config

# Main function to build the data fields config
def build_data_fields_config(input_file=None, output_file=None):

    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Default input file and output file names if not provided
    if input_file is None:
        input_file = os.path.join(script_dir, '..', 'data', 'data_fields.txt')
    if output_file is None:
        output_file_name = os.path.splitext(os.path.basename(input_file))[0] + '_config.json'
        output_file = os.path.join(script_dir, '..', 'config', output_file_name)

    # Load data_fields.txt
    with open(input_file, 'r', encoding='utf-8') as file:
        data_fields_content = file.read()

    # Parse fields from the data_fields.txt content
    parsed_fields = parse_fields(data_fields_content)

    # Initialize an empty config
    config = {}

    # Update the config with parsed fields
    config = update_config_with_fields(config, parsed_fields)

    # Apply generalities to infer field types
    config = apply_generalities(config, generalities)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the updated config back to the file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    print(f"Config file '{output_file_name}' has been updated and saved.")

if __name__ == "__main__":
    # Call the function with default values if no arguments are provided
    build_data_fields_config()
