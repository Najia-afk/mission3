import os
import pandas as pd

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
    if file_list:
        return [f for f in file_list if os.path.isfile(os.path.join(directory, f))]
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def show_loaded_dfs(dfs, df_names=None):
    """Display the head of loaded DataFrames."""
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

def preprocess_df(df):
    """Preprocess DataFrames by dropping empty columns."""
    return df.dropna(axis=1, how='all')

def handle_bad_line(line):
    """Handle and print bad lines encountered during file reading."""
    print(f"Bad line encountered: {line}")
    return None  # Skip the bad line

def load_and_process_data(directory, file_list=None, separator=','):
    """
    Load and preprocess DataFrames from the specified directory.
    If file_list is provided, only process those files.
    The default separator is a comma; this can be overridden by the separator argument.
    """
    dfs = {}
    files = list_files_in_directory(directory, file_list)
    
    for file in files:
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join(directory, file)
        try:
            # Read the file with the specified separator and handle bad lines using the python engine
            df = pd.read_csv(file_path, delimiter=separator, on_bad_lines=handle_bad_line, engine='python')
            if df is not None:
                df = preprocess_df(df)  # Preprocess the DataFrame
                dfs[file_name] = df
        except pd.errors.ParserError as e:
            print(f"ParserError: {e} occurred while processing file '{file}'. Skipping this file.")
        except Exception as e:
            print(f"An error occurred while processing file '{file}': {e}")
    
    return dfs

def load_or_cache_dataframes(dataset_directory, cache_directory='data/cache', file_list=None, separator=','):
    """
    Load DataFrames from cache if available; otherwise, process and cache them.
    If file_list is provided, only process and cache those files.
    The default separator is a comma; this can be overridden by the separator argument.
    """
    dfs = {}
    
    if os.path.exists(cache_directory):
        # Load DataFrames from cache
        print("Loading DataFrames from cache...")
        for file_name in os.listdir(cache_directory):
            df_name = os.path.splitext(file_name)[0]
            if not file_list or df_name in file_list:
                cache_file_path = os.path.join(cache_directory, file_name)
                dfs[df_name] = pd.read_pickle(cache_file_path)
    else:
        # Load and preprocess DataFrames from the source directory
        print("Loading DataFrames from source files...")
        dfs = load_and_process_data(dataset_directory, file_list, separator)
        
        # Save the loaded DataFrames to cache for future use
        os.makedirs(cache_directory, exist_ok=True)
        for name, df in dfs.items():
            cache_file_path = os.path.join(cache_directory, f'{name}.pkl')
            df.to_pickle(cache_file_path)
    
    return dfs



