import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

def plot_nutrition_clusters_efficient(df, frequency_thresholds=[1.0, 0.95]):
    """
    Creates a standalone interactive plot for nutrition score visualization
    with pre-computed frequency thresholds for performance optimization.
    
    Parameters:
        df (pd.DataFrame): DataFrame with nutrition score data and grades
        frequency_thresholds (list): List of frequency thresholds to pre-compute
        
    Returns:
        plotly.graph_objects.Figure: Standalone interactive figure
    """
    # Process the data - work with a copy to avoid modifying the original
    df = df.copy()
    
    # Check required columns
    required_cols = ['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']
    for col in required_cols:
        if col not in df.columns:
            if col == 'nutrition_grade_fr' and 'nutrition_grade_fr' in df.columns:
                df['nutrition_grade_fr'] = df['nutrition_grade_fr']
            elif col == 'nutrition-score-fr_100g' and 'nutrition-score-fr_100g' in df.columns:
                df['nutrition-score-fr_100g'] = df['nutrition-score-fr_100g']
            elif col == 'nutrition-score-uk_100g' and 'nutrition-score-uk_100g' in df.columns:
                df['nutrition-score-uk_100g'] = df['nutrition-score-uk_100g']
            else:
                print(f"Warning: Required column '{col}' not found")
                return None

    # Standardize column names
    df = df.rename(columns={
        'nutrition-score-fr_100g': 'fr_score',
        'nutrition-score-uk_100g': 'uk_score'
    })
    
    # Ensure grade is lowercase
    df['nutrition_grade_fr'] = df['nutrition_grade_fr'].astype(str).str.lower()
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['nutrition_grade_fr', 'fr_score', 'uk_score'])
    
    # Define colors for nutrition grades
    grade_colors = {
        'a': 'green',
        'b': 'lightgreen', 
        'c': 'yellow',
        'd': 'orange',
        'e': 'red'
    }
    
    # Apply color mapping
    df['Color'] = df['nutrition_grade_fr'].map(grade_colors)
    
    # Filter out rows with no color match (invalid grades)
    df = df[~df['Color'].isna()]
    
    # Create a count-based frequency if not already present
    if 'Frequency' not in df.columns:
        grade_counts = df.groupby(['nutrition_grade_fr', 'fr_score', 'uk_score']).size().reset_index(name='Frequency')
        # Merge the frequencies back 
        df = df.merge(grade_counts, on=['nutrition_grade_fr', 'fr_score', 'uk_score'], how='left')
    
    # Calculate cumulative frequency
    df['Cumulative Frequency'] = df['Frequency'].cumsum() / df['Frequency'].sum()
    df = df.sort_values('Cumulative Frequency')
    
    # Create the base figure
    fig = go.Figure()
    
    # Define view options
    grades = ['a', 'b', 'c', 'd', 'e']
    display_modes = ['Bubbles', 'Dots', 'Regression']

    # Create traces for all combinations but set them invisible initially
    all_traces = []
    
    # Pre-compute data for each frequency threshold
    frequency_data = {}
    for threshold in frequency_thresholds:
        # Filter by frequency threshold
        filtered_df = df[df['Cumulative Frequency'] <= threshold].copy()
        frequency_data[threshold] = {
            'df': filtered_df,
            'regression': {}
        }
        
        # Pre-compute regression for each grade
        for grade in grades:
            grade_df = filtered_df[filtered_df['nutrition_grade_fr'] == grade]
            if len(grade_df) >= 3:  # Need at least 3 points for meaningful regression
                try:
                    # Linear regression
                    X = grade_df['fr_score'].values.reshape(-1, 1)
                    y = grade_df['uk_score'].values
                    reg = LinearRegression().fit(X, y)
                    
                    # Store the regression model
                    frequency_data[threshold]['regression'][grade] = {
                        'coef': reg.coef_[0],
                        'intercept': reg.intercept_,
                        'min_x': grade_df['fr_score'].min(),
                        'max_x': grade_df['fr_score'].max()
                    }
                except Exception as e:
                    print(f"Error fitting regression for grade {grade}: {str(e)}")
    
    # First, create traces for each frequency threshold
    for i, threshold in enumerate(frequency_thresholds):
        filtered_df = frequency_data[threshold]['df']
        
        # Add bubble trace (sized by frequency)
        bubble_trace = go.Scatter(
            x=filtered_df['fr_score'],
            y=filtered_df['uk_score'],
            mode='markers',
            marker=dict(
                size=filtered_df['Frequency'],
                color=filtered_df['Color'],
                sizemode='area',
                sizeref=2.*max(filtered_df['Frequency'])/(100.**2),
                sizemin=3
            ),
            hovertemplate='<b>Grade:</b> %{text}<br>FR Score: %{x}<br>UK Score: %{y}<br>Frequency: %{marker.size:.0f}<extra></extra>',
            text=filtered_df['nutrition_grade_fr'].str.upper(),
            name=f'Threshold {threshold*100:.0f}% - Bubbles',
            visible=(i==0)  # Only first threshold visible by default
        )
        fig.add_trace(bubble_trace)
        all_traces.append({
            'threshold': threshold,
            'mode': 'Bubbles',
            'index': len(fig.data) - 1
        })
        
        # Add dots trace (same size)
        dots_trace = go.Scatter(
            x=filtered_df['fr_score'],
            y=filtered_df['uk_score'],
            mode='markers',
            marker=dict(
                size=8,
                color=filtered_df['Color'],
                opacity=0.7
            ),
            hovertemplate='<b>Grade:</b> %{text}<br>FR Score: %{x}<br>UK Score: %{y}<extra></extra>',
            text=filtered_df['nutrition_grade_fr'].str.upper(),
            name=f'Threshold {threshold*100:.0f}% - Dots',
            visible=False  # Hidden by default
        )
        fig.add_trace(dots_trace)
        all_traces.append({
            'threshold': threshold,
            'mode': 'Dots',
            'index': len(fig.data) - 1
        })
        
        # Add regression lines for each grade
        for grade in grades:
            if grade in frequency_data[threshold]['regression']:
                reg_data = frequency_data[threshold]['regression'][grade]
                x_range = np.linspace(reg_data['min_x'], reg_data['max_x'], 50)
                y_range = reg_data['coef'] * x_range + reg_data['intercept']
                
                reg_trace = go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(
                        color=grade_colors[grade],
                        width=2
                    ),
                    name=f'Threshold {threshold*100:.0f}% - Grade {grade.upper()} Trend',
                    hovertemplate=f'<b>Grade {grade.upper()}</b><br>FR: %{{x}}<br>UK: %{{y}}<extra></extra>',
                    visible=False  # Hidden by default
                )
                fig.add_trace(reg_trace)
                all_traces.append({
                    'threshold': threshold,
                    'mode': 'Regression',
                    'grade': grade,
                    'index': len(fig.data) - 1
                })
    
    # Create buttons for threshold selection
    threshold_buttons = []
    for threshold in frequency_thresholds:
        # Create visibility list for each type of display
        bubbles_vis = [False] * len(fig.data)
        dots_vis = [False] * len(fig.data)
        reg_vis = [False] * len(fig.data)
        
        # Set visibility for the right traces
        for trace in all_traces:
            if trace['threshold'] == threshold:
                if trace['mode'] == 'Bubbles':
                    bubbles_vis[trace['index']] = True
                elif trace['mode'] == 'Dots':
                    dots_vis[trace['index']] = True
                elif trace['mode'] == 'Regression':
                    reg_vis[trace['index']] = True
        
        # Add button for this threshold
        threshold_buttons.append(
            dict(
                label=f"Top {threshold*100:.0f}%",
                method="update",
                args=[
                    {"visible": bubbles_vis},  # Default to bubbles view
                    {"title": f"Nutrition Score Analysis (Top {threshold*100:.0f}% by Frequency)"}
                ]
            )
        )
    
    # Create buttons for display mode selection
    mode_buttons = []
    
    # Add button for Bubbles view
    mode_buttons.append(
        dict(
            label="Bubbles",
            method="update",
            args=[
                {"visible": [t['mode'] == 'Bubbles' for t in all_traces]},
                {"title": "Nutrition Score Analysis - Bubbles"}
            ]
        )
    )
    
    # Add button for Dots view
    mode_buttons.append(
        dict(
            label="Dots",
            method="update",
            args=[
                {"visible": [t['mode'] == 'Dots' for t in all_traces]},
                {"title": "Nutrition Score Analysis - Dots"}
            ]
        )
    )
    
    # Add button for Regression view
    mode_buttons.append(
        dict(
            label="Regression",
            method="update",
            args=[
                {"visible": [t['mode'] == 'Regression' for t in all_traces]},
                {"title": "Nutrition Score Analysis - Regression Lines"}
            ]
        )
    )
    
    # Update layout with buttons
    fig.update_layout(
        updatemenus=[
            # Frequency threshold selection menu (top)
            dict(
                buttons=threshold_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor='rgba(158, 202, 225, 0.8)',
                bordercolor='rgba(68, 68, 68, 0.8)'
            ),
            # Display mode menu (below)
            dict(
                buttons=mode_buttons,
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.05,
                yanchor="top",
                bgcolor='rgba(230, 230, 230, 0.8)',
                bordercolor='rgba(68, 68, 68, 0.5)'
            )
        ]
    )
    
    # Add annotations for legend
    legend_items = []
    for i, (grade, color) in enumerate(grade_colors.items()):
        legend_items.append(
            dict(
                x=1.02,
                y=0.9 - 0.1 * i,
                xref="paper",
                yref="paper",
                text=f"<span style='color:{color};font-weight:bold;'>■</span> Grade {grade.upper()}",
                showarrow=False,
                font=dict(family="Arial", size=12),
                align="left"
            )
        )
    
    # Update layout with correct size and title
    fig.update_layout(
        title={
            'text': 'Nutrition Score Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title='Nutrition Score FR',
        yaxis_title='Nutrition Score UK',
        height=700,
        margin=dict(t=150, r=150),
        hovermode='closest',
        template='plotly_white',
        annotations=legend_items
    )
    
    return fig