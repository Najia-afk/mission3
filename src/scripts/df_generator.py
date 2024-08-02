import os
import pandas as pd

# Set pandas display options
pd.set_option('display.max_columns', 5)
pd.set_option('display.expand_frame_repr', False)

# Function to set up the dataset directory path
def get_dataset_directory(notebook_directory, dataset_directory_name='Dataset'):
    return os.path.join(notebook_directory, dataset_directory_name)

# Function to check if a directory exists
def check_directory_exists(directory):
    return os.path.exists(directory)

# Function to list all files in the directory, ignoring specified files
def list_files_in_directory(directory, ignore_files=None):
    if ignore_files is None:
        ignore_files = ['.gitignore']
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f not in ignore_files]

# Function to display loaded DataFrames
def show_loaded_dfs(dfs, df_names=None):
    print("Currently loaded DataFrames:")
    if df_names is None:
        for name, df in dfs.items():
            print(f"DataFrame for file '{name}':")
            print(df.head())
            print("\n")
    else:
        for name in df_names:
            if name in dfs:
                print(f"DataFrame for file '{name}':")
                print(dfs[name].head())
                print("\n")
            else:
                print(f"DataFrame '{name}' not found in the loaded DataFrames.\n")

# Function to preprocess DataFrames by dropping empty columns
def preprocess_df(df):
    return df.dropna(axis=1, how='all')

# Function to handle and print bad lines
def handle_bad_line(line):
    print(f"Bad line encountered: {line}")
    return None  # Skip the bad line

# Main function to load and preprocess DataFrames
def load_and_process_data(directory):
    dfs = {}
    files = list_files_in_directory(directory)
    
    for file in files:
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join(directory, file)
        try:
            # Read the file with a tab delimiter and handle bad lines using the python engine
            df = pd.read_csv(file_path, delimiter='\t', on_bad_lines=handle_bad_line, engine='python')
            if df is not None:
                df = preprocess_df(df)  # Preprocess the DataFrame
                dfs[file_name] = df
        except pd.errors.ParserError as e:
            print(f"ParserError: {e} occurred while processing file '{file}'. Skipping this file.")
        except Exception as e:
            print(f"An error occurred while processing file '{file}': {e}")
    
    return dfs
