import pandas as pd
import logging
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
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

def apply_clustering(df):
    """Assign colors to each nutrition grade."""
    grade_colors = {
        'a': 'green',
        'b': 'lightgreen',
        'c': 'yellow',
        'd': 'orange',
        'e': 'red'
    }
    df['Color'] = df['nutrition_grade_fr'].map(grade_colors)
    return df

def perform_regression(df):
    """Perform frequency-weighted linear regression and calculate the range for each nutrition grade."""
    grade_colors = {
        'a': 'green',
        'b': 'lightgreen',
        'c': 'yellow',
        'd': 'orange',
        'e': 'red'
    }

    regression_results = {}
    for grade in sorted(df['nutrition_grade_fr'].unique()):
        grade_df = df[df['nutrition_grade_fr'] == grade]
        X = grade_df[['nutrition-score-fr_100g']]
        y = grade_df['nutrition-score-uk_100g']
        weights = grade_df['Frequency']
        reg = LinearRegression().fit(X, y, sample_weight=weights)  # Perform weighted linear regression
        regression_results[grade] = {
            'coef': reg.coef_[0],
            'intercept': reg.intercept_,
            'min_fr': X['nutrition-score-fr_100g'].min(),
            'max_fr': X['nutrition-score-fr_100g'].max(),
            'min_y': y.min(),
            'max_y': y.max(),
            'color': grade_colors[grade]  # Add the color associated with the grade
        }
    return regression_results

def create_layout():
    """Create the layout of the Dash app."""
    layout = html.Div([
        html.H1("Nutrition Score Clustering Dashboard", style={'text-align': 'center'}),
        html.Div([
            # Left side: Graph
            html.Div([
                dcc.Graph(id='cluster-bubble-chart', config={'displayModeBar': True}, style={'width': '100%', 'height': '600px'}),
            ], style={'width': '90%', 'display': 'inline-block', 'vertical-align': 'top'}),  # Adjusted width and alignment

            # Right side: Options Panel
            html.Div([

                # Frequency Threshold Slider
                html.Div([
                    html.Label("Select Frequency Threshold:", style={'font-weight': 'bold'}),
                    dcc.Slider(
                        id='frequency-slider',
                        min=0,
                        max=1,  # Dynamically updated in the callback
                        value=1,  # Start at 100%
                        marks={i / 10: f'{i * 10}%' for i in range(11)},  # Linear percentage marks
                        step=0.01,  # Allow fine control over the slider
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], style={'padding': 10, 'margin-bottom': '20px'}),

                # Nutriscore Ranges
                html.Div([
                    html.Label("Nutriscore Ranges:", style={'font-weight': 'bold'}),
                    html.Ul(id='nutriscore-legend')
                ], style={'padding': 10}),

                # Display Options
                html.Div([
                    html.Label("Display Options:", style={'font-weight': 'bold'}),
                    dcc.Checklist(
                        id='display-options',
                        options=[
                            {'label': 'Bubbles', 'value': 'show_bubbles'},
                            {'label': 'Regression Lines', 'value': 'show_regression'},
                            {'label': 'Combination Dots', 'value': 'show_combination_dots'}
                        ],
                        value=['show_bubbles'],  # Default is to show regression lines and bubbles
                        style={'margin-bottom': '20px'}
                    ),
                ], style={'padding': 10}),

            ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-top': '70px'})  # Adjusted alignment and padding
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
    ], style={'width': '100%', 'margin': '0', 'background-color': 'white', 'padding': '1px', 'border-radius': '1px'})
    
    return layout


def update_graph(frequency_threshold, display_options, df):
    """Update the graph based on user input."""
    # Calculate cumulative frequency percentages
    df['Cumulative Frequency'] = df['Frequency'].cumsum() / df['Frequency'].sum()
    
    # Filter the DataFrame based on the selected frequency threshold
    filtered_df = df[df['Cumulative Frequency'] <= frequency_threshold]

    fig = go.Figure()

    # Conditionally add bubbles (scatter plot with size based on Frequency)
    if 'show_bubbles' in display_options:
        fig.add_trace(go.Scatter(
            x=filtered_df['nutrition-score-fr_100g'],
            y=filtered_df['nutrition-score-uk_100g'],
            mode='markers',
            name='Nutriscore Points',
            marker=dict(
                size=filtered_df['Frequency'],
                color=filtered_df['Color'],
                sizemode='area',
                sizeref=2.*max(filtered_df['Frequency'])/(100.**2),
                sizemin=0.1,
            ),
            showlegend=False
        ))

    # Conditionally perform and add regression lines
    if 'show_regression' in display_options:
        regression_results = perform_regression(filtered_df)
        for grade, params in regression_results.items():
            X_range = np.linspace(params['min_fr'], params['max_fr'], 100)
            y_range = params['coef'] * X_range + params['intercept']
            fig.add_trace(go.Scatter(
                x=X_range,
                y=y_range,
                mode='lines',
                name=f'Regression {grade.upper()}',
                line=dict(color=params['color']),  # Use the color from the regression results
                showlegend=False
            ))

    # Conditionally add combination dots
    if 'show_combination_dots' in display_options:
        fig.add_trace(go.Scatter(
            x=filtered_df['nutrition-score-fr_100g'],
            y=filtered_df['nutrition-score-uk_100g'],
            mode='markers',
            name='Combination Dots',
            marker=dict(
                size=8,
                color=filtered_df['Color'],
                opacity=0.6
            ),
            showlegend=False
        ))

    # Adjust the layout
    fig.update_layout(
        xaxis_title='Nutrition Score FR',
        yaxis_title='Nutrition Score UK',
        transition_duration=500
    )

    return fig

def update_nutriscore_legend(frequency_threshold, df):
    """Generate a legend with Nutriscore grades and their ranges."""
    # Filter the DataFrame based on the selected frequency threshold
    df['Cumulative Frequency'] = df['Frequency'].cumsum() / df['Frequency'].sum()
    filtered_df = df[df['Cumulative Frequency'] <= frequency_threshold]

    # Perform regression on the filtered data
    regression_results = perform_regression(filtered_df)

    legend_items = []
    for grade, params in regression_results.items():
        grade_df = filtered_df[filtered_df['nutrition_grade_fr'] == grade]
        if not grade_df.empty:
            min_x = grade_df['nutrition-score-fr_100g'].min()
            min_y = grade_df['nutrition-score-uk_100g'].min()
            max_x = grade_df['nutrition-score-fr_100g'].max()
            max_y = grade_df['nutrition-score-uk_100g'].max()
            legend_items.append(html.Li(f"{grade.upper()}: [FR {min_x}, UK {min_y}] - [FR {max_x}, UK {max_y}]", style={'color': params['color']}))
    
    return legend_items

def start_dash_server(df):
    """Start the Dash server."""
    global app
    if app is None:
        app = Dash(__name__)
        app.layout = create_layout()

        @app.callback(
            Output('cluster-bubble-chart', 'figure'),
            Output('nutriscore-legend', 'children'),
            [Input('frequency-slider', 'value'),
             Input('display-options', 'value')]
        )
        def update_graph_callback(frequency_threshold, display_options):
            graph = update_graph(frequency_threshold, display_options, df)
            legend = update_nutriscore_legend(frequency_threshold, df)
            return graph, legend

        # Start the server inline in Jupyter Notebook
        from IPython.display import display
        app.run_server(mode='inline', port=8051)

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

    # Assign colors to each nutrition grade
    df = apply_clustering(df)

    start_dash_server(df)

if __name__ == "__main__":
    nutriscore_directory = 'path_to_your_data/nutrition_combination_log.csv'  # Replace with actual path
    df = pd.read_csv(nutriscore_directory)
    run_dash_app_nutriscore(df)
