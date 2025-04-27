import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_nutrition_grade_plots(df, numeric_cols=None, grade_col='nutrition_grade_fr'):
    """
    Create interactive box plots showing distribution of numeric variables across nutrition grades.
    
    Args:
        df: DataFrame containing the data
        numeric_cols: List of numeric columns to analyze (if None, all numeric columns are used)
        grade_col: Column name containing nutrition grades
        
    Returns:
        A plotly figure object
    """
    # Make a copy to avoid modifying the original
    df_temp = df.copy()
    
    # Check if grade column exists
    if grade_col not in df_temp.columns:
        raise ValueError(f"Grade column '{grade_col}' not found in dataframe")
    
    # Filter out rows with missing grade values
    df_temp = df_temp[~df_temp[grade_col].isna()]
    
    # Identify numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df_temp.select_dtypes(include=['number']).columns.tolist()
        # Remove any score columns that might be directly derived from the grade
        numeric_cols = [col for col in numeric_cols if 'score' not in col.lower() or col == 'nutrition-score-fr_100g']
    
    # Create a single figure with dropdown menu
    fig = go.Figure()
    
    # Add box plots for the first variable to make them initially visible
    if len(numeric_cols) > 0:
        first_var = numeric_cols[0]
        for grade in sorted(df_temp[grade_col].unique()):
            subset = df_temp[df_temp[grade_col] == grade]
            
            # Skip grades with too few data points
            if len(subset) < 5:
                continue
                
            fig.add_trace(go.Box(
                y=subset[first_var],
                name=f'Grade {grade}',
                boxmean=True,
                marker_color=get_grade_color(grade),
                visible=True
            ))
    
    # Add all other variables as hidden traces
    for col in numeric_cols[1:]:
        for grade in sorted(df_temp[grade_col].unique()):
            subset = df_temp[df_temp[grade_col] == grade]
            
            # Skip grades with too few data points
            if len(subset) < 5:
                continue
                
            fig.add_trace(go.Box(
                y=subset[col],
                name=f'Grade {grade}',
                boxmean=True,
                marker_color=get_grade_color(grade),
                visible=False
            ))
    
    # Create dropdown menu
    buttons = []
    for i, col in enumerate(numeric_cols):
        # Calculate the number of grades (used for visibility array length)
        grade_count = len([g for g in sorted(df_temp[grade_col].unique()) 
                           if len(df_temp[df_temp[grade_col] == g]) >= 5])
        
        visibility = [False] * len(fig.data)
        for j in range(grade_count):
            visibility[i * grade_count + j] = True
            
        buttons.append(dict(
            label=format_column_name(col),
            method='update',
            args=[{'visible': visibility}, 
                  {'title': f'Distribution of {format_column_name(col)} by Nutrition Grade',
                   'yaxis': {'title': format_column_name(col)}}]
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {format_column_name(numeric_cols[0])} by Nutrition Grade',
        yaxis_title=format_column_name(numeric_cols[0]),
        xaxis_title='Nutrition Grade',
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.15,
            'y': 0.7,
        },
        {
            'buttons': [
                {
                    'method': 'restyle',
                    'label': 'Box Plot',
                    'args': [{'type': 'box'}],
                },
                {
                    'method': 'restyle',
                    'label': 'Violin Plot',
                    'args': [{'type': 'violin', 'points': False, 'box': True, 'meanline': True}],
                }
            ],
            'direction': 'right',
            'x': 0.5,
            'y': 1.10,
            'showactive': True
        }],
        boxmode='group',
        height=600,
        legend_title_text='Nutrition Grade',
        margin=dict(l=40, r=150, t=80, b=40)
    )
    
    # Add annotation for the dropdown
    fig.add_annotation(
        x=1.2,
        y=0.9,
        xref="paper",
        yref="paper",
        text="Select Nutrient:",
        showarrow=False,
        align="center"
    )
    
    return fig

def format_column_name(col_name):
    """Format column name for display in the visualization"""
    # Replace underscores and hyphens with spaces
    formatted = col_name.replace('_', ' ').replace('-', ' ')
    
    # Remove _100g suffix if present
    formatted = formatted.replace('100g', '/100g')
    
    # Capitalize words
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    return formatted

def get_grade_color(grade):
    """Return a color based on the nutrition grade"""
    grade_colors = {
        'a': '#038141',  # dark green
        'b': '#85bb2f',  # light green
        'c': '#fecb02',  # yellow
        'd': '#ee8100',  # orange
        'e': '#e63e11'   # red
    }
    
    # Default to gray if grade not found
    return grade_colors.get(grade.lower(), '#808080')
