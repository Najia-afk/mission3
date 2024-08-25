import pandas as pd
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import ast

logging.basicConfig(level=logging.INFO)

# Global variables
app = None

def safe_eval(x):
    try:
        x = x.replace('nan', 'None')  # Replace 'nan' with None
        return ast.literal_eval(x)
    except Exception as e:
        logging.error(f"Error parsing {x}: {e}")
        return None

def apply_clustering(df, eps=2, min_samples=5):
    """Apply DBSCAN clustering to the nutrition scores."""
    X = df[['nutrition-score-fr_100g', 'nutrition-score-uk_100g']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    df['Cluster'] = clustering.labels_
    return df

def create_layout():
    """Create the layout of the Dash app."""
    layout = html.Div([
        html.H1("Nutrition Score Clustering Dashboard"),
        html.Div([
            html.Label("Select Nutrition Score Comparison:"),
            dcc.Dropdown(
                id='score-comparison',
                options=[{'label': 'FR vs UK Scores', 'value': 'fr_vs_uk'}],
                value='fr_vs_uk',
                style={'width': '50%'}
            ),
        ], style={'padding': '10px'}),

        html.Div([
            html.Label("Select Frequency Threshold:"),
            dcc.Slider(
                id='frequency-slider',
                min=0,
                max=1,  # This will be dynamically updated in the callback
                value=0,
                marks={0: 'Low', 1: 'High'},
                step=1,
            ),
        ], style={'padding': '10px'}),

        dcc.Graph(id='cluster-bubble-chart'),
    ])
    return layout

def update_graph(comparison_mode, frequency_threshold, df):
    """Update the graph based on user input."""
    # Filter the DataFrame based on the selected frequency threshold
    filtered_df = df[df['Frequency'] >= frequency_threshold]

    # Create the scatter plot with bubble sizes
    fig = px.scatter(
        filtered_df,
        x='nutrition-score-fr_100g',
        y='nutrition-score-uk_100g',
        size='Frequency',
        color='Cluster',
        hover_name='nutrition_grade_fr',
        title=f'Nutrition Score Clusters: FR vs UK',
        labels={'nutrition-score-fr_100g': 'Nutrition Score FR', 'nutrition-score-uk_100g': 'Nutrition Score UK'},
        size_max=50
    )

    # Enhance the layout
    fig.update_layout(transition_duration=500)

    return fig

def start_dash_server(df):
    """Start the Dash server."""
    global app
    if app is None:
        app = Dash(__name__)
        app.layout = create_layout()

        @app.callback(
            Output('cluster-bubble-chart', 'figure'),
            [Input('score-comparison', 'value'),
             Input('frequency-slider', 'value')]
        )
        def update_graph_callback(comparison_mode, frequency_threshold):
            return update_graph(comparison_mode, frequency_threshold, df)

        # Start the server inline in Jupyter Notebook
        from IPython.display import display
        app.run_server(mode='inline', debug=True, port=8051)

def run_dash_app_nutriscore(df):
    """Run the Dash app with the provided nutrition score DataFrame."""
    global app

    df['nutrition_combination'] = df['nutrition_combination'].apply(safe_eval)
    df[['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']] = pd.DataFrame(
        df['nutrition_combination'].tolist(), index=df.index
    )

    # Handle missing values if necessary (e.g., remove rows with NaNs)
    df.dropna(subset=['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'], inplace=True)

    df = df.drop(columns=['nutrition_combination'])

    # Apply clustering to the nutrition scores
    df = apply_clustering(df)

    # If the app is already initialized, make sure the layout is set before starting the server
    if app is not None:
        app.layout['frequency-slider'].max = df['Frequency'].max()
        app.layout['frequency-slider'].marks = {int(freq): str(int(freq)) for freq in df['Frequency'].unique() if freq % 10000 == 0}

    start_dash_server(df)

if __name__ == "__main__":
    nutriscore_directory = 'path_to_your_data/nutrition_combination_log.csv'  # Replace with actual path
    df = pd.read_csv(nutriscore_directory)
    run_dash_app_nutriscore(df)
