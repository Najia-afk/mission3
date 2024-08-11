import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import seaborn as sns

def filter_metadata_and_dataframes(combined_metadata, dfs):
    """
    Filters combined_metadata to drop rows where 'Fill Percentage' < 50, and then filters 
    the related DataFrames to keep only the columns that remain in the filtered metadata.
    """
    # Filter out rows in combined_metadata where 'Fill Percentage' < 50
    combined_metadata = combined_metadata[combined_metadata['Fill Percentage'] >= 50].copy()
    
    # Iterate over the filtered metadata to update the DataFrames
    for df_name in combined_metadata['DataFrame'].unique():
        
        # Get the relevant DataFrame
        if df_name in dfs:
            df = dfs[df_name]
            # Get the columns to keep based on the filtered metadata
            columns_to_keep = combined_metadata[combined_metadata['DataFrame'] == df_name]['Column Name'].tolist()
            # Filter the DataFrame
            filtered_df = df[columns_to_keep]
            # Replace the original DataFrame with the filtered one
            dfs[df_name] = filtered_df
            print(f"Updated DataFrame '{df_name}' to retain only relevant columns.")
        else:
            print(f"DataFrame '{df_name}' not found in the provided DataFrames.")
    
    # Generate scatter plots for each DataFrame in the metadata
    plot_scatter_for_metadata(combined_metadata)
    
    return combined_metadata, dfs

def plot_scatter_for_metadata(combined_metadata, output_dir='graph'):
    """
    Generate scatter plots for each DataFrame in the combined metadata.
    Plots Duplicate Percentage vs. Fill Percentage for each Column Name,
    grouping points with similar values and separating them in the legend.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for df_name in combined_metadata['DataFrame'].unique():
        df_metadata = combined_metadata[combined_metadata['DataFrame'] == df_name]

        # Drop rows with NaN in 'Fill Percentage' or 'Duplicate Percentage'
        df_metadata = df_metadata.dropna(subset=['Fill Percentage', 'Duplicate Percentage'])

        if df_metadata.empty:
            print(f"No valid data for plotting in '{df_name}'. Skipping.")
            continue

        # Ensure Fill Percentage is greater than or equal to Duplicate Percentage
        #df_metadata = df_metadata[df_metadata['Fill Percentage'] >= df_metadata['Duplicate Percentage']]

        # Sort the metadata by Duplicate Percentage and Fill Percentage
        df_metadata = df_metadata.sort_values(by=['Duplicate Percentage', 'Fill Percentage'], ascending=[False, False])

        # Plotting
        plt.figure(figsize=(21, 15))
        num_points = len(df_metadata)
        colors = cm.rainbow(np.linspace(0, 1, num_points))

        plt.scatter(df_metadata['Duplicate Percentage'], df_metadata['Fill Percentage'], s=100, c=colors, alpha=0.6)

        # Create grouped legend items
        legend_items = []
        previous_group = None
        group_threshold = 4  # Define threshold for grouping (1% difference)

        for i, (index, row) in enumerate(df_metadata.iterrows()):
            current_group = (row['Duplicate Percentage'] // group_threshold, row['Fill Percentage'] // group_threshold)
            
            if previous_group is not None and current_group != previous_group:
                legend_items.append(plt.Line2D([0], [0], color='none', label=" "))  # Add empty space between groups
            
            legend_items.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           label=f"{row['Column Name']} (D: {row['Duplicate Percentage']:.2f}%, F: {row['Fill Percentage']:.2f}%)",
                                           markerfacecolor=colors[min(i, len(colors)-1)], markersize=10))
            previous_group = current_group

        
        plt.xlabel('Duplicate Percentage')
        plt.ylabel('Fill Percentage')
        plt.grid(True)
        
        # Adjust layout to make room for the legend
        plt.subplots_adjust(left=0.10, right=0.75, top=0.9, bottom=0.1)

        # Position legend outside the plot area
        legend = plt.legend(handles=legend_items, loc='center left', bbox_to_anchor=(1.01, 0.5), 
                            borderaxespad=0., title=f'Columns by Group ({group_threshold}%)', handlelength=1.5, handletextpad=0.8)

        # Remove legend border
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_facecolor('none')

        plt.title(f'Scatter Plot for {df_name} - Grouped by Duplicate & Fill Percentage', pad=20, loc='center')

        output_path = os.path.join(output_dir, f'scatter_{df_name}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot for '{df_name}' has been generated and saved as '{output_path}'.")

def remove_url_column(df):
    print("Removal of column 'url'")
    df.drop(columns=['url'], inplace=True)
    print("Column 'url' has been removed")


def check_datetime_consistency(df, timestamp_column, datetime_column, log_dir='logs'):
    print(f"Check for differences between '{timestamp_column}' and '{datetime_column}'")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{datetime_column}_log.csv')
    discrepancies = []

    with open(log_path, 'w') as log_file:
        log_file.write(f'Row,{timestamp_column},{datetime_column}\n')
        for idx, row in df.iterrows():
            timestamp_value = pd.to_numeric(row[timestamp_column], errors='coerce')
            datetime_value = pd.to_datetime(row[datetime_column], errors='coerce')

            if pd.notna(timestamp_value):
                timestamp_date = pd.to_datetime(timestamp_value, unit='s', errors='coerce').date()

            if pd.notna(datetime_value):
                datetime_date = datetime_value.date()

            if pd.isnull(datetime_value) and pd.notnull(timestamp_value):
                df.at[idx, datetime_column] = pd.to_datetime(timestamp_date)
            elif pd.notnull(timestamp_value) and pd.notnull(datetime_value) and timestamp_date != datetime_date:
                log_file.write(f"{idx},{timestamp_date},{datetime_date}\n")
                discrepancies.append((idx, timestamp_date, datetime_date))

    if discrepancies:
        print(f"Found {len(discrepancies)} discrepancies between '{timestamp_column}' and '{datetime_column}'. Please review the log before proceeding.")
    else:
        df.drop(columns=[timestamp_column], inplace=True)
        print(f"No significant discrepancies found, '{timestamp_column}' has been dropped.")


def check_field_consistency(df, fields, log_dir, output_dir, generic_name):
    """
    Analyzes the consistency across specified fields, logs discrepancies, and generates histograms.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    fields (list): List of field names to analyze.
    log_dir (str): Directory to save logs.
    output_dir (str): Directory to save plots.
    generic_name (str): Generic name for naming logs and plots.
    """
    # Create necessary directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Create unique combinations and count occurrences
    df[f'{generic_name}_combination'] = df.apply(lambda x: tuple(x[field] for field in fields), axis=1)
    combination_counts = df[f'{generic_name}_combination'].value_counts()

    # Step 2: Log the combinations and their frequencies
    combination_log_path = os.path.join(log_dir, f'{generic_name}_combination_log.csv')
    combination_counts.to_csv(combination_log_path, header=['Frequency'])

    # Step 3: Analyze consistency for each first field value
    first_field = fields[0]
    field_groupings = df.groupby(first_field)[fields[1:]].nunique()
    inconsistent_entries = field_groupings[(field_groupings > 1).any(axis=1)]
    
    # Step 4: Log inconsistent mappings
    inconsistent_log_path = os.path.join(log_dir, f'inconsistent_{generic_name}_mappings.csv')
    inconsistent_entries.to_csv(inconsistent_log_path)

    # Step 5: Plot the histogram for the combination frequencies
    draw_histogram_for_field_combinations(combination_counts, output_dir, generic_name)
    
    # Step 6: Analyze if we can drop other fields
    if inconsistent_entries.empty:
        df.drop(columns=fields[1:], inplace=True)
        print(f"Columns {fields[1:]} have been dropped as they are identical to '{first_field}'")
    else:
        print(f"Inconsistencies found in {generic_name}. Check the log for more details. Keeping all columns.")

def draw_histogram_for_field_combinations(combination_counts, output_dir, generic_name):
    # Plot the full histogram with all data
    plt.figure(figsize=(10, 6))
    sns.histplot(combination_counts, kde=True, bins=20, color='blue', alpha=0.6)
    
    mean_val = combination_counts.mean()
    median_val = combination_counts.median()
    std_val = combination_counts.std()

    # Use a more conventional outlier threshold, e.g., mean + 3 * std
    outlier_threshold = mean_val + 3 * std_val
    num_outliers = (combination_counts > outlier_threshold).sum()

    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='-', label=f'Median: {median_val:.2f}')
    plt.axvline(mean_val + std_val, color='b', linestyle=':', label=f'Std Dev (+): {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='b', linestyle=':', label=f'Std Dev (-): {mean_val - std_val:.2f}')

    # Annotate outliers
    if num_outliers > 0:
        plt.text(outlier_threshold, plt.ylim()[1] * 0.9, f'{num_outliers} outliers detected', color='black')

    plt.title(f'Histogram of {generic_name} Combination Frequencies (All Data)')
    plt.xlabel('Frequency')
    plt.ylabel('Count of Combinations')
    plt.legend()
    plt.tight_layout()

    file_name = f"{generic_name}_combination_histogram_all_data.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

    # Plot the histogram without outliers
    clipped_combination_counts = combination_counts[combination_counts <= outlier_threshold]

    plt.figure(figsize=(10, 6))
    sns.histplot(clipped_combination_counts, kde=True, bins=20, color='green', alpha=0.6)
    
    clipped_mean_val = clipped_combination_counts.mean()
    clipped_median_val = clipped_combination_counts.median()
    clipped_std_val = clipped_combination_counts.std()

    plt.axvline(clipped_mean_val, color='r', linestyle='--', label=f'Mean: {clipped_mean_val:.2f}')
    plt.axvline(clipped_median_val, color='g', linestyle='-', label=f'Median: {clipped_median_val:.2f}')
    plt.axvline(clipped_mean_val + clipped_std_val, color='b', linestyle=':', label=f'Std Dev (+): {clipped_mean_val + clipped_std_val:.2f}')
    plt.axvline(clipped_mean_val - clipped_std_val, color='b', linestyle=':', label=f'Std Dev (-): {clipped_mean_val - clipped_std_val:.2f}')

    plt.title(f'Histogram of {generic_name} Combination Frequencies (Without Outliers)')
    plt.xlabel('Frequency')
    plt.ylabel('Count of Combinations')
    plt.legend()
    plt.tight_layout()

    clipped_file_name = f"{generic_name}_combination_histogram_without_outliers.png"
    plt.savefig(os.path.join(output_dir, clipped_file_name))
    plt.close()

    print(f"{num_outliers} outliers were excluded from the second plot.")

def process_dataframe(df, log_dir='logs', output_dir='graph'):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    #remove_url_column(df)

    # Optimize date columns
    check_datetime_consistency(df, 'created_t', 'created_datetime', log_dir='logs')
    check_datetime_consistency(df, 'last_modified_t', 'last_modified_datetime', log_dir='logs')
    
    # For countries columns
    check_field_consistency(df, ['countries', 'countries_tags', 'countries_fr'], log_dir='logs', output_dir='graph', generic_name='countries')
    # For ingredients with palm oil
    check_field_consistency(df, ['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n'], log_dir='logs', output_dir='graph', generic_name='ingredients_palm_oil')
    # For nutrition columns
    check_field_consistency(df, ['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'], log_dir='logs', output_dir='graph', generic_name='nutrition')
    check_field_consistency(df, ['brands_tags', 'brands'], log_dir='logs', output_dir='graph', generic_name='brands')
    check_field_consistency(df, ['additives_n', 'additives'], log_dir='logs', output_dir='graph', generic_name='additives')
    check_field_consistency(df, ['states', 'states_tags','states_fr'], log_dir='logs', output_dir='graph', generic_name='states')
    

    return df

