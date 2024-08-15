import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import seaborn as sns
import logging
from sklearn.cluster import DBSCAN
from matplotlib.patches import Ellipse
from matplotlib.collections import PathCollection


# Set up logging
logging.basicConfig(level=logging.INFO)

def filter_metadata_and_dataframes(combined_metadata, dfs):
    """
    Filters combined_metadata to drop rows where 'Fill Percentage' < 50, and then filters 
    the related DataFrames to keep only the columns that remain in the filtered metadata.
    """
    # Filter out rows in combined_metadata where 'Fill Percentage' < 50
    combined_metadata = combined_metadata[combined_metadata['Fill Percentage'] >= 40].copy()
    
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
            logging.info(f"Updated DataFrame '{df_name}' to retain only relevant columns.")
        else:
            logging.warning(f"DataFrame '{df_name}' not found in the provided DataFrames.")
    
    # Generate scatter plots for each DataFrame in the metadata
    plot_scatter_with_clustering(combined_metadata)
    
    return combined_metadata, dfs


def get_bbox_properties(bbox):
    """ Extract the properties of a bounding box. """
    return {
        'left': bbox.x0,
        'right': bbox.x1,
        'top': bbox.y1,
        'bottom': bbox.y0
    }

def rectangles_intersect(r1, r2):
    """ Check if two rectangles intersect. """
    return not (r2['left'] > r1['right'] or
                r2['right'] < r1['left'] or
                r2['top'] > r1['bottom'] or
                r2['bottom'] < r1['top'])

def is_within_limits(x, y):
    xlim = plt.xlim()
    ylim = plt.ylim()
    return xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]

def adjust_position(x, y):
    """ Ensure the annotation stays within plot limits. """
    xlim = plt.xlim()
    ylim = plt.ylim()
    x_margin = 0.05 * (xlim[1] - xlim[0])
    y_margin = 0.05 * (ylim[1] - ylim[0])
    new_x = min(max(x, xlim[0] + x_margin), xlim[1] - x_margin)
    new_y = min(max(y, ylim[0] + y_margin), ylim[1] - y_margin)
    return new_x, new_y

def adjust_annotation_position(annotation, x, y, padding=10, max_attempts=10000):
    """ Adjust the position of the annotation to avoid overlap with data points or other annotations. """
    
    # Initialize the position and the last valid position
    current_x, current_y = x, y
    last_valid_x, last_valid_y = adjust_position(current_x, current_y)

    # Compute initial annotation bbox
    bbox = annotation.get_window_extent()
    bbox = bbox.transformed(plt.gca().transData.inverted())
    bbox_props = get_bbox_properties(bbox)
    
    attempts = 0
    while attempts < max_attempts:
        # Generate random movement within the range [-padding, padding]
        dx = np.random.uniform(-padding, padding)
        dy = np.random.uniform(-padding, padding)
        
        new_x = current_x + dx
        new_y = current_y + dy
        
        # Ensure the new position is within plot limits
        new_x, new_y = adjust_position(new_x, new_y)
        
        # Compute new annotation bbox
        annotation.set_position((new_x, new_y))
        bbox = annotation.get_window_extent()
        bbox = bbox.transformed(plt.gca().transData.inverted())
        bbox_props = get_bbox_properties(bbox)

        # Check for overlap with other annotations and data points
        overlap_found = False
        for artist in plt.gca().get_children():
            if isinstance(artist, plt.Line2D) or isinstance(artist, PathCollection):
                # Check for overlap with data points
                artist_bbox = artist.get_window_extent()
                artist_bbox = artist_bbox.transformed(plt.gca().transData.inverted())
                artist_bbox_props = get_bbox_properties(artist_bbox)
                if rectangles_intersect(bbox_props, artist_bbox_props):
                    overlap_found = True
                    break

            if isinstance(artist, plt.Text):
                # Check for overlap with other text annotations
                artist_bbox = artist.get_window_extent()
                artist_bbox = artist_bbox.transformed(plt.gca().transData.inverted())
                artist_bbox_props = get_bbox_properties(artist_bbox)
                if rectangles_intersect(bbox_props, artist_bbox_props):
                    overlap_found = True
                    break

            if isinstance(artist, Ellipse):
                # Check for overlap with ellipses representing clusters
                artist_bbox = artist.get_window_extent()
                artist_bbox = artist_bbox.transformed(plt.gca().transData.inverted())
                artist_bbox_props = get_bbox_properties(artist_bbox)
                if rectangles_intersect(bbox_props, artist_bbox_props):
                    overlap_found = True
                    break
        
        # If no overlap and within bounds, return the new position
        if not overlap_found and (
            bbox_props['left'] >= plt.xlim()[0] and
            bbox_props['right'] <= plt.xlim()[1] and
            bbox_props['bottom'] >= plt.ylim()[0] and
            bbox_props['top'] <= plt.ylim()[1]):
            return new_x, new_y
        
        # Update last valid position only if the new position is within bounds
        if bbox_props['left'] >= plt.xlim()[0] and \
           bbox_props['right'] <= plt.xlim()[1] and \
           bbox_props['bottom'] >= plt.ylim()[0] and \
           bbox_props['top'] <= plt.ylim()[1]:
            last_valid_x, last_valid_y = new_x, new_y
        
        # Increment attempt counter
        attempts += 1

    # Return the last valid position if max_attempts reached
    return last_valid_x, last_valid_y

def plot_scatter_with_clustering(df_metadata, graph_dir='graph'):
    # Ensure the output directory exists
    os.makedirs(graph_dir, exist_ok=True)
    
    # Filter data based on Fill Percentage
    df_filtered = df_metadata[(df_metadata['Fill Percentage'] >= 40) & (df_metadata['Fill Percentage'] <= 100)]
    
    if df_filtered.empty:
        logging.info("No data to plot after filtering.")
        return
    
    # Prepare data for clustering
    X = df_filtered[['Duplicate Percentage', 'Fill Percentage']].values
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    df_filtered['Cluster'] = clustering.labels_

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Define a color map for clusters
    unique_clusters = np.unique(df_filtered['Cluster'])
    num_clusters = len(unique_clusters)
    colors = plt.cm.get_cmap('tab10', num_clusters)
    
    # Create a dictionary to keep track of field names for each cluster
    cluster_field_dict = {cluster: [] for cluster in unique_clusters if cluster != -1}
    
    for index, row in df_filtered.iterrows():
        if row['Cluster'] != -1:
            cluster_field_dict[row['Cluster']].append(row['Column Name'])
    
    # Plot each cluster
    for cluster in unique_clusters:
        if cluster == -1:
            # Plot and annotate outliers
            outlier_data = df_filtered[df_filtered['Cluster'] == -1]
            for _, row in outlier_data.iterrows():
                plt.scatter(row['Duplicate Percentage'], row['Fill Percentage'], 
                            color='red', s=100, label='Outlier', edgecolor='black')
                annot = plt.annotate(row['Column Name'], 
                                     xy=(row['Duplicate Percentage'], row['Fill Percentage']),
                                     xytext=(row['Duplicate Percentage'], row['Fill Percentage'] - 2),
                                     fontsize=8, color='red',
                                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='none'),
                                     arrowprops=dict(arrowstyle="->", color='red'))
                
                # Adjust annotation position to avoid overlap
                new_x, new_y = adjust_annotation_position(annot, row['Duplicate Percentage'], row['Fill Percentage'])
                annot.set_position((new_x, new_y))
        else:
            cluster_data = df_filtered[df_filtered['Cluster'] == cluster]
            
            # Plot the data points
            plt.scatter(cluster_data['Duplicate Percentage'], cluster_data['Fill Percentage'], 
                        label=f'Cluster {cluster}', alpha=0.6, color=colors(cluster), s=100)
            
            # Plot an ellipse to show cluster range
            mean_dup = cluster_data['Duplicate Percentage'].mean()
            mean_fill = cluster_data['Fill Percentage'].mean()
            std_dup = cluster_data['Duplicate Percentage'].std()
            std_fill = cluster_data['Fill Percentage'].std()
            
            ellipse = Ellipse(xy=(mean_dup, mean_fill), 
                              width=2*std_dup, height=2*std_fill, 
                              edgecolor=colors(cluster), facecolor='none', linestyle='--')
            plt.gca().add_patch(ellipse)
            
            # Annotate the plot with the field names for each cluster
            field_names = cluster_field_dict[cluster]
            field_lines = [", ".join(field_names[i:i+2]) for i in range(0, len(field_names), 2)]
            field_text = "\n".join(field_lines)
            
            # Compute annotation position (left of the ellipse)
            annot_x = mean_dup
            annot_y = mean_fill
            
            # Ensure annotations are within plot limits
            annot_x, annot_y = adjust_position(annot_x, annot_y)
            
            # Add annotation and arrow in one go
            annot = plt.annotate(field_text, xy=(mean_dup, mean_fill), xytext=(annot_x, annot_y),
                                 fontsize=8, color=colors(cluster),
                                 bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors(cluster), facecolor='none'),
                                 arrowprops=dict(arrowstyle="->", color=colors(cluster), 
                                                 connectionstyle="arc3,rad=0"))
            
            # Adjust annotation position to avoid overlap
            new_x, new_y = adjust_annotation_position(annot, mean_dup, mean_fill)
            annot.set_position((new_x, new_y))

    plt.xlabel('Duplicate Percentage')
    plt.ylabel('Fill Percentage')
    plt.title('Scatter Plot with DBSCAN Clustering')
    plt.grid(True)
    
    # Save plot
    output_path = os.path.join(graph_dir, 'scatter_with_clustering.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Scatter plot with clustering saved as '{output_path}'.")

def remove_url_column(df):
    if 'url' in df.columns:
        df.drop(columns=['url'], inplace=True)
        logging.info("Column 'url' has been removed")

def check_datetime_consistency(df, timestamp_column, datetime_column, log_dir='logs'):
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
        logging.warning(f"Found {len(discrepancies)} discrepancies between '{timestamp_column}' and '{datetime_column}'. Please review the log before proceeding.")
    else:
        df.drop(columns=[timestamp_column], inplace=True)
        logging.info(f"No significant discrepancies found, '{timestamp_column}' has been dropped.")

def check_field_frequency(df, fields, temp_dir, graph_dir, generic_name):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    df[f'{generic_name}_combination'] = df.apply(lambda x: tuple(x[field] for field in fields), axis=1)
    combination_counts = df[f'{generic_name}_combination'].value_counts()

    combination_log_path = os.path.join(temp_dir, f'{generic_name}_combination_log.csv')
    combination_counts.to_csv(combination_log_path, header=['Frequency'])

    first_field = fields[0]
    field_groupings = df.groupby(first_field)[fields[1:]].nunique()
    frequency_entries = field_groupings[(field_groupings > 1).any(axis=1)]
    
    frequency_temp_path = os.path.join(temp_dir, f'frequency_{generic_name}_mappings.csv')
    frequency_entries.to_csv(frequency_temp_path)

    draw_histogram_for_field_combinations(combination_counts, graph_dir, generic_name)
    
    logging.info(f"Check the {generic_name} combination file for more details about fields frequency.")

def draw_histogram_for_field_combinations(combination_counts, graph_dir, generic_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(combination_counts, kde=True, bins=20, color='blue', alpha=0.6)
    
    mean_val = combination_counts.mean()
    median_val = combination_counts.median()
    std_val = combination_counts.std()
    outlier_threshold = mean_val + 3 * std_val
    num_outliers = (combination_counts > outlier_threshold).sum()

    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='-', label=f'Median: {median_val:.2f}')
    plt.axvline(mean_val + std_val, color='b', linestyle=':', label=f'Std Dev (+): {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='b', linestyle=':', label=f'Std Dev (-): {mean_val - std_val:.2f}')

    if num_outliers > 0:
        plt.text(outlier_threshold, plt.ylim()[1] * 0.9, f'{num_outliers} outliers detected', color='black')

    plt.title(f'Histogram of {generic_name} Combination Frequencies (All Data)')
    plt.xlabel('Frequency')
    plt.ylabel('Count of Combinations')
    plt.legend()
    plt.tight_layout()

    file_name = f"{generic_name}_combination_histogram_all_data.png"
    plt.savefig(os.path.join(graph_dir, file_name))
    plt.close()

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
    plt.savefig(os.path.join(graph_dir, clipped_file_name))
    plt.close()

    logging.info(f"{num_outliers} outliers were excluded from the second plot.")



def process_dataframe(df, log_dir='logs', temp_dir='temp', graph_dir='graph'):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    #remove_url_column(df)

    # Optimize date columns
    #check_datetime_consistency(df, 'created_t', 'created_datetime', log_dir='logs')
    #check_datetime_consistency(df, 'last_modified_t', 'last_modified_datetime', log_dir='logs')
    
    # Field frequency checks
    checks = [
        (['countries', 'countries_tags', 'countries_fr'], 'countries'),
        (['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n'], 'ingredients_palm_oil'),
        (['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'], 'nutrition'),
        (['brands_tags', 'brands'], 'brands'),
        (['additives_n', 'additives'], 'additives'),
        (['states', 'states_tags', 'states_fr'], 'states')
    ]
    for fields, generic_name in checks:
        check_field_frequency(df, fields, temp_dir, graph_dir, generic_name)

    return df