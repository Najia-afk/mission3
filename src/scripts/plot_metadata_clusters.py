import pandas as pd
import logging
import numpy as np
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

logging.basicConfig(level=logging.INFO)

def filter_metadata_and_dataframes(combined_metadata, dfs, min_fill=40):
    """
    Filters combined_metadata to drop rows where 'Fill Percentage' < min_fill, 
    and then filters the related DataFrames to keep only the columns that remain in the filtered metadata.
    
    Parameters:
    - combined_metadata: DataFrame containing metadata for all DataFrames.
    - dfs: Dictionary of DataFrames to be filtered.
    - min_fill: Minimum fill percentage to filter the metadata.
    
    Returns:
    - combined_metadata: Filtered combined metadata.
    - dfs: Updated dictionary of filtered DataFrames.
    """
    # Filter out rows in combined_metadata where 'Fill Percentage' < min_fill
    combined_metadata = combined_metadata[combined_metadata['Fill Percentage'] >= min_fill].copy()
    
    # Iterate over the filtered metadata to update the DataFrames
    for df_name in combined_metadata['DataFrame'].unique():
        if df_name in dfs:
            df = dfs[df_name]
            columns_to_keep = combined_metadata[combined_metadata['DataFrame'] == df_name]['Column Name'].tolist()
            filtered_df = df[columns_to_keep]
            dfs[df_name] = filtered_df
            logging.info(f"Updated DataFrame '{df_name}' to retain only relevant columns.")
        else:
            logging.warning(f"DataFrame '{df_name}' not found in the provided DataFrames.")
    
    return combined_metadata, dfs


def apply_clustering(df_filtered):
    """
    Apply DBSCAN clustering on the filtered metadata DataFrame.
    
    Parameters:
    - df_filtered: DataFrame after applying initial filters.
    
    Returns:
    - df_filtered: DataFrame with an additional 'Cluster' column indicating the cluster assignment.
    """
    X = df_filtered[['Duplicate Percentage', 'Fill Percentage']].values
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    df_filtered['Cluster'] = clustering.labels_
    
    return df_filtered

def plot_scatter_with_clustering(df_filtered, graph_dir='graph'):
    """
    Plot a scatter plot with DBSCAN clustering and save it to the specified directory.
    
    Parameters:
    - df_filtered: DataFrame containing the filtered and clustered data.
    - graph_dir: Directory to save the plot.
    """
    os.makedirs(graph_dir, exist_ok=True)
    
    if df_filtered.empty:
        logging.info("No data to plot after filtering.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Define a color map for clusters
    unique_clusters = np.unique(df_filtered['Cluster'])
    num_clusters = len(unique_clusters)
    colors = plt.cm.get_cmap('tab10', num_clusters)
    
    for cluster in unique_clusters:
        if cluster == -1:
            outlier_data = df_filtered[df_filtered['Cluster'] == -1]
            for _, row in outlier_data.iterrows():
                plt.scatter(row['Duplicate Percentage'], row['Fill Percentage'], 
                            color='red', s=100, label='Outlier', edgecolor='black')
        else:
            cluster_data = df_filtered[df_filtered['Cluster'] == cluster]
            plt.scatter(cluster_data['Duplicate Percentage'], cluster_data['Fill Percentage'], 
                        label=f'Cluster {cluster}', alpha=0.6, color=colors(cluster), s=100)
            
            mean_dup = cluster_data['Duplicate Percentage'].mean()
            mean_fill = cluster_data['Fill Percentage'].mean()
            std_dup = cluster_data['Duplicate Percentage'].std()
            std_fill = cluster_data['Fill Percentage'].std()
            
            ellipse = Ellipse(xy=(mean_dup, mean_fill), width=2*std_dup, height=2*std_fill, 
                              edgecolor=colors(cluster), facecolor='none', linestyle='--')
            plt.gca().add_patch(ellipse)
    
    plt.xlabel('Duplicate Percentage')
    plt.ylabel('Fill Percentage')
    plt.title('Scatter Plot with DBSCAN Clustering')
    plt.grid(True)
    
    output_path = os.path.join(graph_dir, 'scatter_with_clustering.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Scatter plot with clustering saved as '{output_path}'.")

app = Dash(__name__)

def create_layout(df_metadata):
    layout = html.Div([
        html.H1("Dynamic Data Processing App"),
        dcc.Slider(
            id='min-fill-slider',
            min=0, max=100, step=1, value=40,
            marks={str(i): str(i) for i in range(0, 101, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
            included=True,
        ),
        dcc.Slider(
            id='max-fill-slider',
            min=0, max=100, step=1, value=100,
            marks={str(i): str(i) for i in range(0, 101, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        dcc.Graph(id='graph')
    ])
    return layout

@app.callback(
    Output('graph', 'figure'),
    [Input('min-fill-slider', 'value'),
     Input('max-fill-slider', 'value')]
)
def update_graph(min_fill, max_fill, combined_metadata, dfs):
    combined_metadata, dfs = filter_metadata_and_dataframes(combined_metadata, dfs, min_fill=min_fill)
    df_filtered = apply_clustering(combined_metadata)
    plot_scatter_with_clustering(df_filtered)

    fig = px.scatter(df_filtered, x='Duplicate Percentage', y='Fill Percentage', color='Cluster')
    return fig

def run_dash_app(combined_metadata, dfs):
    app.layout = create_layout(combined_metadata)
    app.run_server(mode='inline')

if __name__ == "__main__":
    combined_metadata = pd.DataFrame()  # Replace with actual DataFrame loading
    dfs = {}  # Replace with actual DataFrame loading
    run_dash_app(combined_metadata, dfs)
