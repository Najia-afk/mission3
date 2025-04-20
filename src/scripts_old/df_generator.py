import os, json
import hashlib
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt



# Set pandas display options
pd.set_option('display.max_columns', 5)
pd.set_option('display.expand_frame_repr', False)

def get_dataset_directory(notebook_directory, dataset_directory_name='Dataset'):
    """Return the full path to the dataset directory."""
    return os.path.join(notebook_directory, dataset_directory_name)

def check_directory_exists(directory):
    """Check if the directory exists."""
    return os.path.exists(directory)

def list_files_in_directory(directory, file_list=None):
    """
    List all files in the directory.
    If file_list is provided, only return those files.
    """
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if file_list:
        # Ensure file_list contains full file names with extensions
        return [f for f in all_files if f in file_list]
    return all_files

def show_loaded_dfs(dfs, df_names=None):
    """Display the head of loaded DataFrames."""
    print("Currently loaded DataFrames:")
    if df_names is None:
        for name, df in dfs.items():
            print(f"DataFrame for file '{name} {df.shape}':")
            print(df.head())
            print("\n")
    else:
        for name in df_names:
            if name in dfs:
                print(f"DataFrame for file '{name} {df.shape}':")
                print(dfs[name].head())
                print("\n")
            else:
                print(f"DataFrame '{name}' not found in the loaded DataFrames.\n")

def preprocess_df(df):
    """Preprocess DataFrames by dropping empty columns."""
    return df.dropna(axis=1, how='all')

def handle_bad_line(line):
    """Handle and print bad lines encountered during file reading."""
    print(f"Bad line encountered: {line}")
    return None  # Skip the bad line

def generate_file_hash(file_path):
    """Generate a SHA-256 hash of the file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


import numpy as np

def save_nullity_matrix(df, file_name, output_dir='graph'):
    """Save a nullity matrix of the DataFrame as a PNG file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{file_name}_nullity_matrix.png')
    plt.figure(figsize=(15, 9))
    msno.matrix(df, sparkline=False)
    plt.title(f'Nullity Matrix for {file_name}')
    plt.savefig(output_path)
    plt.close()
    print(f"Nullity matrix for '{file_name}' has been generated.")

def load_and_process_data(directory, file_list=None, separator=','):
    """
    Load and preprocess DataFrames from the specified directory.
    If file_list is provided, only process those files.
    The default separator is a comma; this can be overridden by the separator argument.
    """
    dfs = {}
    files = list_files_in_directory(directory, file_list)
    
    for file in files:
        file_name = os.path.splitext(file)[0]  # Keep full file name including extension for caching
        file_path = os.path.join(directory, file)
        try:
            # Read the file with the specified separator and handle bad lines using the python engine
            df = pd.read_csv(file_path, delimiter=separator, on_bad_lines=handle_bad_line, engine='python')
            if df is not None:
                df = preprocess_df(df)  # Preprocess the DataFrame
                dfs[file] = df  # Keep the full file name with extension
        except pd.errors.ParserError as e:
            print(f"ParserError: {e} occurred while processing file '{file}'. Skipping this file.")
        except Exception as e:
            print(f"An error occurred while processing file '{file}': {e}")
    
    return dfs

def load_or_cache_dataframes(dataset_directory, cache_directory='data/cache', file_list=None, separator=','):
    os.makedirs(cache_directory, exist_ok=True)
    """
    Load DataFrames from cache if available; otherwise, process and cache them.
    If file_list is provided, only process and cache those files.
    The default separator is a comma; this can be overridden by the separator argument.
    """
    dfs = {}
    hash_file_path = os.path.join(cache_directory, 'file_hashes.json')
    
    # Load existing hashes
    if os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as f:
            cached_hashes = json.load(f)
    else:
        cached_hashes = {}

    files_to_process = list_files_in_directory(dataset_directory, file_list)
    new_hashes = cached_hashes.copy()  # Start with the existing hashes

    for file_name_with_extension in files_to_process:
        # Remove the file extension
        file_name, _ = os.path.splitext(file_name_with_extension)
        file_path = os.path.join(dataset_directory, file_name_with_extension)
        file_hash = generate_file_hash(file_path)
        new_hashes[file_name] = file_hash

        # Check if the file has been changed since the last cache
        if file_name in cached_hashes and cached_hashes[file_name] == file_hash:
            # Load from cache
            cache_file_path = os.path.join(cache_directory, f'{file_name}.pkl')
            if os.path.exists(cache_file_path):
                dfs[file_name] = pd.read_pickle(cache_file_path)
                print(f"Loaded '{file_name}' from cache.")
            else:
                print(f"Cache file for '{file_name}' not found, processing file.")
                loaded_dfs = load_and_process_data(dataset_directory, [file_name_with_extension], separator)
                dfs[file_name] = loaded_dfs[file_name_with_extension]
                dfs[file_name].to_pickle(cache_file_path)
        else:
            # Process and cache new or changed files
            loaded_dfs = load_and_process_data(dataset_directory, [file_name_with_extension], separator)
            dfs[file_name] = loaded_dfs[file_name_with_extension]
            cache_file_path = os.path.join(cache_directory, f'{file_name}.pkl')
            dfs[file_name].to_pickle(cache_file_path)
            print(f"Processed and cached '{file_name}'.")

        # Save a nullity matrix for the DataFrame
        save_nullity_matrix(dfs[file_name], file_name)

    # Update the cache hashes
    with open(hash_file_path, 'w') as f:
        json.dump(new_hashes, f)

    return dfs
