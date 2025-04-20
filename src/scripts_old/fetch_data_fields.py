import os
import requests
from datetime import datetime
import pandas as pd
import hashlib

DATA_DIR = os.path.join('data')
HISTORY_DIR = os.path.join(DATA_DIR, 'history')
DIFF_DIR = os.path.join(DATA_DIR, 'diffs')
LATEST_FILE_PATH = os.path.join(DATA_DIR, 'data_fields.txt')
DATA_FIELDS_URL = 'https://world.openfoodfacts.org/data/data-fields.txt'


def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def save_data(content, path):
    with open(path, 'w') as file:
        file.write(content)

def load_data(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return file.read()
    return None

def hash_data(data):
    return hashlib.md5(data.encode('utf-8')).hexdigest()

def compare_data(old_data, new_data):
    if old_data is None:
        print("No previous version found. This is the first fetch.")
        return True, new_data.splitlines(), []

    old_hash = hash_data(old_data)
    new_hash = hash_data(new_data)

    if old_hash == new_hash:
        print(f"No changes detected on {LATEST_FILE_PATH}.")
        return False, [], []

    old_lines = set(old_data.splitlines())
    new_lines = set(new_data.splitlines())
    
    added_lines = new_lines - old_lines
    removed_lines = old_lines - new_lines

    return True, added_lines, removed_lines

def save_diff(added_lines, removed_lines, version1, version2, format='csv'):
    diff_data = []
    for line in added_lines:
        diff_data.append({"Change": "Added", "Line": line.strip()})
    
    for line in removed_lines:
        diff_data.append({"Change": "Removed", "Line": line.strip()})
    
    df = pd.DataFrame(diff_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    diff_filename = f'diff_{version1}_vs_{version2}_{timestamp}.{format}'
    diff_filepath = os.path.join(DIFF_DIR, diff_filename)
    
    if format == 'csv':
        df.to_csv(diff_filepath, index=False)
    elif format == 'json':
        df.to_json(diff_filepath, orient='records', lines=True)
    
    print(f"Diff saved to {diff_filepath}")

def fetch_and_compare_data_fields(data_directory):
    DATA_DIR = data_directory
    
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(DIFF_DIR, exist_ok=True)

    # Debugging: Print to confirm directory creation
    print(f"Created HISTORY_DIR: {os.path.exists(HISTORY_DIR)}")
    print(f"Created DIFF_DIR: {os.path.exists(DIFF_DIR)}")
    
    latest_data = fetch_data(DATA_FIELDS_URL)
    previous_data = load_data(LATEST_FILE_PATH)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    latest_file_with_timestamp = os.path.join(HISTORY_DIR, f'data_fields_{timestamp}.txt')
    save_data(latest_data, latest_file_with_timestamp)
    
    has_changes, added_lines, removed_lines = compare_data(previous_data, latest_data)
    
    if has_changes:
        version1 = os.path.basename(LATEST_FILE_PATH)
        version2 = os.path.basename(latest_file_with_timestamp)
        save_diff(added_lines, removed_lines, version1, version2, format='csv')

    # Update the latest file with the new data
    save_data(latest_data, LATEST_FILE_PATH)

if __name__ == "__main__":
    fetch_and_compare_data_fields(data_directory=DATA_DIR)