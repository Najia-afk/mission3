import os
import pandas as pd
import time
from typing import Dict, List, Optional, Union


class DataFrameLoader:
    """
    A class for loading and caching pandas DataFrames from CSV files.
    
    This class provides functionality to:
    - Load dataframes from source files or cache
    - Cache dataframes for faster future loading
    - Display information about loaded dataframes
    """
    
    def __init__(self, dataset_dir: str, cache_dir: str):
        """
        Initialize the DataFrameLoader with directories for dataset and cache.
        
        Parameters:
            dataset_dir (str): Directory containing the source files
            cache_dir (str): Directory to store cached dataframes
        """
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_dataframes(self, file_list: Optional[List[str]] = None, 
                       separator: str = '\t', 
                       force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load dataframes from source files or cache.
        
        Parameters:
            file_list (list, optional): List of specific files to process, or None for all files
            separator (str): Separator used in the CSV files
            force_reload (bool): If True, bypass cache and reload from source files
            
        Returns:
            dict: Dictionary of dataframes with filenames as keys
        """
        # Get list of files to process
        if file_list is None:
            files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.csv')]
        else:
            files = [f for f in file_list if f in os.listdir(self.dataset_dir) and f.endswith('.csv')]
        
        if not files:
            print(f"No CSV files found in {self.dataset_dir}")
            return {}
        
        dfs = {}
        
        for file in files:
            start_time = time.time()
            # Define cache file path
            cache_file = os.path.join(self.cache_dir, f"{file.replace('.', '_')}_cache.pkl")
            
            # Try to load from cache first if not forcing reload
            if os.path.exists(cache_file) and not force_reload:
                print(f"Loading {file} from cache...")
                try:
                    df = pd.read_pickle(cache_file)
                    elapsed = time.time() - start_time
                    print(f"Loaded {file} from cache successfully in {elapsed:.2f} seconds.")
                    # Get name without extension
                    df_name = os.path.splitext(file)[0]
                    dfs[df_name] = df
                    continue
                except Exception as e:
                    print(f"Error loading from cache: {e}. Will load from source instead.")
            
            # If cache doesn't exist, loading failed, or force_reload is True, load from source
            print(f"Loading {file} from source...")
            file_path = os.path.join(self.dataset_dir, file)
            
            try:
                # Try with specific separator
                df = pd.read_csv(file_path, sep=separator, low_memory=False)
                elapsed = time.time() - start_time
                print(f"Loaded {file} successfully with shape: {df.shape} in {elapsed:.2f} seconds.")
                
                # Save to cache for future use
                try:
                    df.to_pickle(cache_file)
                    print(f"Saved {file} to cache.")
                except Exception as e:
                    print(f"Error saving to cache: {e}")
                
                # Get name without extension
                df_name = os.path.splitext(file)[0]
                dfs[df_name] = df
                
            except Exception as e:
                print(f"Error loading {file} with specified separator: {e}")
                # Try with automatic separator detection if the specified one fails
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    elapsed = time.time() - start_time
                    print(f"Loaded {file} with automatic separator detection in {elapsed:.2f} seconds.")
                    
                    # Save to cache
                    try:
                        df.to_pickle(cache_file)
                        print(f"Saved {file} to cache.")
                    except Exception as e:
                        print(f"Error saving to cache: {e}")
                    
                    # Get name without extension
                    df_name = os.path.splitext(file)[0]
                    dfs[df_name] = df
                    
                except Exception as e2:
                    print(f"Failed to load {file} with all methods: {e2}")
        
        return dfs
    
    def display_dataframes_info(self, dfs: Dict[str, pd.DataFrame], 
                              num_samples: int = 2,
                              memory_unit: str = 'MB') -> None:
        """
        Display basic information about loaded dataframes.
        
        Parameters:
            dfs (dict): Dictionary of dataframes
            num_samples (int): Number of sample rows to display
            memory_unit (str): Unit for memory usage ('KB', 'MB', 'GB')
        """
        if not dfs:
            print("No DataFrames to display.")
            return
        
        unit_multiplier = {
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }
        
        multiplier = unit_multiplier.get(memory_unit, unit_multiplier['MB'])
        
        for name, df in dfs.items():
            print(f"\n{'='*50}")
            print(f"DataFrame: {name}")
            print(f"{'='*50}")
            print(f"Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
            print(f"Memory usage: {df.memory_usage().sum() / multiplier:.2f} {memory_unit}")
            
            # Count missing values
            missing_values = df.isna().sum().sum()
            missing_percentage = missing_values / (df.shape[0] * df.shape[1]) * 100
            print(f"Missing values: {missing_values} ({missing_percentage:.2f}% of all cells)")
            
            # Data types summary
            print("\nData Types:")
            dtype_counts = df.dtypes.value_counts().to_dict()
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} columns")
            
            #print("\nSample data:")
            #print(df.head(num_samples))
            
            print("\nColumn names preview:")
            if len(df.columns) > 10:
                print(", ".join(df.columns[:10]) + "... and " + str(len(df.columns) - 10) + " more")
            else:
                print(", ".join(df.columns))
    
    def clear_cache(self, file_pattern: Optional[str] = None) -> None:
        """
        Clear cache files matching the given pattern or all cache files.
        
        Parameters:
            file_pattern (str, optional): Pattern to match cache files, or None for all
        """
        if not os.path.exists(self.cache_dir):
            print(f"Cache directory {self.cache_dir} does not exist.")
            return
            
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('_cache.pkl')]
        
        if file_pattern:
            cache_files = [f for f in cache_files if file_pattern in f]
            
        if not cache_files:
            print("No cache files found to clear.")
            return
            
        for file in cache_files:
            try:
                os.remove(os.path.join(self.cache_dir, file))
                print(f"Removed {file} from cache.")
            except Exception as e:
                print(f"Error removing {file}: {e}")
                
        print(f"Cleared {len(cache_files)} cache files.")


# For backward compatibility, maintain the original function that uses the new class
def load_or_cache_dataframes(dataset_dir, cache_dir, file_list=None, separator='\t'):
    """
    Load dataframes from source files or cache using DataFrameLoader class.
    
    Parameters:
        dataset_dir (str): Directory containing the source files
        cache_dir (str): Directory to store cached dataframes
        file_list (list): List of specific files to process, or None for all files
        separator (str): Separator used in the files
        
    Returns:
        dict: Dictionary of dataframes with filenames as keys
    """
    loader = DataFrameLoader(dataset_dir, cache_dir)
    dfs = loader.load_dataframes(file_list, separator)
    loader.display_dataframes_info(dfs)
    return dfs