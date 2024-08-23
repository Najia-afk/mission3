import pandas as pd
import logging
import numpy as np
from sklearn.cluster import DBSCAN
import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import threading

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
        # Outer container to hold everything together with a white background
        html.Div([
            # Title
            html.H1("Clustering of Data Based on Fill and Duplicate Percentages", style={'text-align': 'center'}),
            
            # Inner container for the graph and controls
            html.Div([
                # Graph
                dcc.Graph(id='graph', config={'displayModeBar': True}, style={'width': '100%', 'height': '600px'}),
                
                # Controls
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
                            style={'width': '45px'},  # Slightly increased input field width
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
                            style={'width': '45px'},  # Slightly increased input field width
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
                            style={'width': '30px'},  # Slightly increased input field width
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
                            style={'width': '30px'},  # Slightly increased input field width
                        ),
                    ], style={'padding': 10}),
                ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'})
                
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
        ], style={'background-color': 'white', 'padding': '20px', 'border-radius': '10px'})
    ], style={'width': '100%', 'margin': '0 auto'})
    
    return layout


def update_graph(min_fill, max_fill, eps, min_samples):
    """Update the graph based on user input."""
    filtered_metadata = filter_metadata_and_dataframes(combined_metadata, min_fill=min_fill, max_fill=max_fill)
    clustered_data = apply_clustering(filtered_metadata, eps=eps, min_samples=min_samples)

    fig = px.scatter(clustered_data, x='Fill Percentage', y='Duplicate Percentage', color='Cluster',
                     labels={'Fill Percentage': 'Fill %', 'Duplicate Percentage': 'Duplicate %'},
                     template='plotly_white')

    # Remove legend and replace it with circles on the graph
    # fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    # fig.update_layout(showlegend=False)

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
    
    # If there's an existing server thread, stop it
    if server_thread and server_thread.is_alive():
        server_thread.join(timeout=1)
    
    # Start the Dash server in a new thread
    server_thread = threading.Thread(target=start_dash_server)
    server_thread.start()

if __name__ == "__main__":
    combined_metadata = pd.DataFrame()  # Replace with actual DataFrame loading
    run_dash_app(combined_metadata)
