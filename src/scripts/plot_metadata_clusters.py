import pandas as pd
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import threading
import random

logging.basicConfig(level=logging.INFO)

# Global variables
combined_metadata = None
server_thread = None
app = None

def filter_metadata_and_dataframes(metadata, min_fill=40, max_fill=100):
    filtered_metadata = metadata[(metadata['Fill Percentage'] >= min_fill) &
                                 (metadata['Fill Percentage'] <= max_fill)].copy()
    return filtered_metadata

def apply_clustering(df_filtered, eps=3, min_samples=2):
    X = df_filtered[['Duplicate Percentage', 'Fill Percentage']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    df_filtered['Cluster'] = clustering.labels_
    return df_filtered

def create_layout():
    """Create the layout of the Dash app."""
    layout = html.Div([
        html.Div([
            html.H1("Clustering of Data Based on Fill and Duplicate Percentages", style={'text-align': 'center'}),
            html.Div([
                dcc.Graph(id='graph', config={'displayModeBar': True}, style={'width': '100%', 'height': '600px'}),
                html.Div([
                    html.Div([
                        html.Label('Min Fill Percentage (%)', style={'font-weight': 'bold'}),
                        dcc.Input(
                            id='min-fill-input',
                            type='number',
                            value=40,
                            min=0,
                            max=100,
                            step=1,
                            style={'width': '45px'},
                        ),
                    ], style={'padding': 10}),
                    html.Div([
                        html.Label('Max Fill Percentage (%)', style={'font-weight': 'bold'}),
                        dcc.Input(
                            id='max-fill-input',
                            type='number',
                            value=100,
                            min=0,
                            max=100,
                            step=1,
                            style={'width': '45px'},
                        ),
                    ], style={'padding': 10}),
                    html.Div([
                        html.Label('DBSCAN eps', style={'font-weight': 'bold'}),
                        dcc.Input(
                            id='eps-input',
                            type='number',
                            value=3,
                            min=0.1,
                            max=10,
                            step=0.1,
                            style={'width': '30px'},
                        ),
                    ], style={'padding': 10}),
                    html.Div([
                        html.Label('DBSCAN min_samples', style={'font-weight': 'bold'}),
                        dcc.Input(
                            id='min-samples-input',
                            type='number',
                            value=2,
                            min=1,
                            max=20,
                            step=1,
                            style={'width': '30px'},
                        ),
                    ], style={'padding': 10}),
                ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
        ], style={'background-color': 'white', 'padding': '20px', 'border-radius': '10px'})
    ], style={'width': '100%', 'margin': '0 auto'})
    
    return layout

def update_graph(min_fill, max_fill, eps, min_samples):
    """Update the graph based on user input."""
    # Filter and cluster the data
    filtered_metadata = filter_metadata_and_dataframes(combined_metadata, min_fill=min_fill, max_fill=max_fill)
    clustered_data = apply_clustering(filtered_metadata, eps=eps, min_samples=min_samples)

    # Create the scatter plot
    fig = px.scatter(clustered_data, x='Fill Percentage', y='Duplicate Percentage', color='Cluster',
                     labels={'Fill Percentage': 'Fill %', 'Duplicate Percentage': 'Duplicate %'},
                     template='plotly_white')

    # Initialize variables for annotation management
    annotation_positions = []  # Track positions to avoid overlaps

    def is_position_valid(x, y, ax_offset, ay_offset, text):
        """Check if a proposed annotation position is valid (no overlap)."""
        # Estimate the height of the annotation by counting the number of lines
        num_lines = text.count("<br>") + 1
        adjusted_distance = 10 * num_lines  # Adjust the distance based on the number of lines

        for pos in annotation_positions:
            if abs(pos[0] - (x + ax_offset)) < 50 and abs(pos[1] - (y + ay_offset)) < adjusted_distance:
                return False
        return True

    def add_annotation(x, y, text, color, is_isolated=False):
        """Add an annotation to the figure and update annotation positions."""
        # Generate positions dynamically with rotation
        positions = [(dx, dy) for dx in range(20, 2000, 5) for dy in range(20, 2000, 5)]
        rotations = [(dx, dy) for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]
        random.shuffle(rotations)

        for offset in positions:
            for rotation in rotations:
                ax_offset, ay_offset = offset[0] * rotation[0], offset[1] * rotation[1]
                if is_position_valid(x, y, ax_offset, ay_offset, text):
                    fig.add_annotation(
                        x=x,
                        y=y,
                        text=text,
                        showarrow=True,
                        arrowhead=2,
                        ax=ax_offset,
                        ay=ay_offset,
                        arrowcolor=color,
                        font=dict(size=10, color=color),
                        bordercolor=color,
                        borderwidth=1,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                    annotation_positions.append((x + ax_offset, y + ay_offset))
                    return  # Exit once a valid position is found

    # Handle isolated points (Cluster -1)
    for index, row in clustered_data[clustered_data['Cluster'] == -1].iterrows():
        x = row['Fill Percentage']
        y = row['Duplicate Percentage']
        field_text = row['Column Name']

        # Use dark blue for isolated points
        color = "DarkBlue"

        # Add the annotation near the point with a short arrow
        add_annotation(x, y, field_text, color)

    # Handle clustered points, ordered by cluster size (ascending)
    cluster_sizes = clustered_data[clustered_data['Cluster'] != -1]['Cluster'].value_counts().sort_values().index
    for cluster in cluster_sizes:
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        mean_x = cluster_data['Fill Percentage'].mean()
        mean_y = cluster_data['Duplicate Percentage'].mean()

        # Prepare text for annotation (concatenate column names)
        field_names = cluster_data['Column Name'].tolist()
        field_text_lines = [", ".join(field_names[i:i+2]) for i in range(0, len(field_names), 2)]
        field_text = "<br>".join(field_text_lines)  # Use <br> for new lines in Plotly

        # Set color based on the cluster
        color = px.colors.qualitative.Plotly[cluster % len(px.colors.qualitative.Plotly)]

        # Add the annotation with the calculated offsets
        add_annotation(mean_x, mean_y, field_text, color)

    # Customize legend and layout
    fig.update_layout(legend=dict(
        x=1,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))

    return fig


def start_dash_server():
    """Start the Dash server."""
    global app, server_thread
    if app is None:
        app = Dash(__name__)
        app.layout = create_layout()

        # Define the callback only after app is initialized
        @app.callback(
            Output('graph', 'figure'),
            [Input('min-fill-input', 'value'),
             Input('max-fill-input', 'value'),
             Input('eps-input', 'value'),
             Input('min-samples-input', 'value')]
        )
        def update_graph_callback(min_fill, max_fill, eps, min_samples):
            return update_graph(min_fill, max_fill, eps, min_samples)
        
        app.run_server(mode='inline', threaded=True)
    else:
        app.layout = create_layout()
        app.run_server(mode='inline', threaded=True)

def run_dash_app(metadata):
    """Run the Dash app with the provided metadata."""
    global combined_metadata, server_thread
    combined_metadata = metadata
    
    if server_thread and server_thread.is_alive():
        server_thread.join(timeout=1)
    
    server_thread = threading.Thread(target=start_dash_server)
    server_thread.start()

if __name__ == "__main__":
    combined_metadata = pd.DataFrame()  # Replace with actual DataFrame loading
    run_dash_app(combined_metadata)
