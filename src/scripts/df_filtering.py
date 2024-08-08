import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd

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



def process_dataframe(df, log_file='metadata_df.log', output_dir='graph'):
    # Create or clear the log file
    if os.path.exists(log_file):
        os.remove(log_file)
    
    os.makedirs(output_dir, exist_ok=True)

    # 1. Remove the 'url' column
    print("Removal of column url")
    df.drop(columns=['url'], inplace=True)
    print("Column url has been removed")

    # 2. Check for differences between 'created_t' and 'created_datetime'
    print("Check for differences between 'created_t' and 'created_datetime'")
    for idx, row in df.iterrows():
        created_t = pd.to_numeric(row['created_t'], errors='coerce')
        created_datetime = pd.to_datetime(row['created_datetime'], errors='coerce')
        
        if pd.notna(created_t):
            created_t = pd.to_datetime(created_t, unit='s', errors='coerce').normalize()
        
        if pd.notna(created_datetime):
            created_datetime = created_datetime.normalize()
        
        if pd.isnull(created_datetime) and pd.notnull(created_t):
            df.at[idx, 'created_datetime'] = created_t
        elif pd.notnull(created_t) and pd.notnull(created_datetime) and created_t != created_datetime:
            with open(log_file, 'a') as f:
                f.write(f"Difference found in row {idx} between 'created_t' and 'created_datetime': {created_t} vs {created_datetime}\n")

    df.drop(columns=['created_t'], inplace=True)
    print("No bad diff between 'created_t' and 'created_datetime', 'created_t' has been dropped ")

    # 3. Check for differences between 'last_modified_t' and 'last_modified_datetime'
    print("Check for differences between 'last_modified_t' and 'last_modified_datetime'")
    for idx, row in df.iterrows():
        last_modified_t = pd.to_numeric(row['last_modified_t'], errors='coerce')
        last_modified_datetime = pd.to_datetime(row['last_modified_datetime'], errors='coerce')

        if pd.notna(last_modified_t):
            last_modified_t = pd.to_datetime(last_modified_t, unit='s', errors='coerce').normalize()

        if pd.notna(last_modified_datetime):
            last_modified_datetime = last_modified_datetime.normalize()

        if pd.isnull(last_modified_datetime) and pd.notnull(last_modified_t):
            df.at[idx, 'last_modified_datetime'] = last_modified_t
        elif pd.notnull(last_modified_t) and pd.notnull(last_modified_datetime) and last_modified_t != last_modified_datetime:
            with open(log_file, 'a') as f:
                f.write(f"Difference found in row {idx} between 'last_modified_t' and 'last_modified_datetime': {last_modified_t} vs {last_modified_datetime}\n")

    df.drop(columns=['last_modified_t'], inplace=True)
    print("No bad diff between 'last_modified_t' and 'last_modified_datetime', 'last_modified_t' has been dropped ")

    # 4. Check differences between 'countries', 'countries_tags', and 'countries_fr' and plot a bar graph
    print("Check differences between 'countries', 'countries_tags', and 'countries_fr' and plot a bar graph")
    country_combinations = df[['countries', 'countries_tags', 'countries_fr']].apply(lambda x: tuple(x), axis=1)
    country_comb_counts = country_combinations.value_counts()

    plt.figure(figsize=(10, 6))
    country_comb_counts.plot(kind='bar')
    plt.title('Country Combinations Distribution')
    plt.xlabel('Combinations (countries, countries_tags, countries_fr)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'country_combinations_distribution.png'))
    plt.close()

    # Drop 'countries_tags' and 'countries_fr' if they are identical
    if df['countries'].equals(df['countries_tags']) and df['countries'].equals(df['countries_fr']):
        df.drop(columns=['countries_tags', 'countries_fr'], inplace=True)
        print("Columns 'countries_tags' and 'countries_fr' have been dropped as they are identical to 'countries'")
    else:
        print("Difference found among 'countries', 'countries_tags', and 'countries_fr'. Keeping all columns.")

    # 5. Check differences between 'ingredients_from_palm_oil_n' and 'ingredients_that_may_be_from_palm_oil_n'
    print("Check differences between 'ingredients_from_palm_oil_n' and 'ingredients_that_may_be_from_palm_oil_n'")
    for idx, row in df.iterrows():
        if pd.isnull(row['ingredients_from_palm_oil_n']) and pd.notnull(row['ingredients_that_may_be_from_palm_oil_n']):
            df.at[idx, 'ingredients_from_palm_oil_n'] = row['ingredients_that_may_be_from_palm_oil_n']
        elif pd.notnull(row['ingredients_from_palm_oil_n']) and pd.notnull(row['ingredients_that_may_be_from_palm_oil_n']) and row['ingredients_from_palm_oil_n'] != row['ingredients_that_may_be_from_palm_oil_n']:
            with open(log_file, 'a') as f:
                f.write(f"Difference found in row {idx} between 'ingredients_from_palm_oil_n' and 'ingredients_that_may_be_from_palm_oil_n'\n")

    df.drop(columns=['ingredients_that_may_be_from_palm_oil_n'], inplace=True)
    print("Processed 'ingredients_from_palm_oil_n' and 'ingredients_that_may_be_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n' has been dropped")

    # 6. Check differences between 'nutrition_grade_fr', 'nutrition-score-fr_100g', and 'nutrition-score-uk_100g' and plot a bar graph
    print("Check differences between 'nutrition_grade_fr', 'nutrition-score-fr_100g', and 'nutrition-score-uk_100g' and plot a bar graph")
    nutrition_combinations = df[['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']].apply(lambda x: tuple(x), axis=1)
    nutrition_comb_counts = nutrition_combinations.value_counts()

    plt.figure(figsize=(10, 6))
    nutrition_comb_counts.plot(kind='bar')
    plt.title('Nutrition Combinations Distribution')
    plt.xlabel('Combinations (nutrition_grade_fr, nutrition-score-fr_100g, nutrition-score-uk_100g)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nutrition_combinations_distribution.png'))
    plt.close()

    # Drop 'nutrition-score-fr_100g' and 'nutrition-score-uk_100g' if they are identical
    if df['nutrition_grade_fr'].equals(df['nutrition-score-fr_100g']) and df['nutrition_grade_fr'].equals(df['nutrition-score-uk_100g']):
        df.drop(columns=['nutrition-score-fr_100g', 'nutrition-score-uk_100g'], inplace=True)
        print("Columns 'nutrition-score-fr_100g' and 'nutrition-score-uk_100g' have been dropped as they are identical to 'nutrition_grade_fr'")
    else:
        print("Difference found among 'nutrition_grade_fr', 'nutrition-score-fr_100g', and 'nutrition-score-uk_100g'. Keeping all columns.")

    return df
