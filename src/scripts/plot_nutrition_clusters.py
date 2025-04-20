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
    # Memory optimization: Only keep necessary columns
    df = df[['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']].copy()
    
    # Standardize column names
    df.rename(columns={
        'nutrition-score-fr_100g': 'fr_score',
        'nutrition-score-uk_100g': 'uk_score'
    }, inplace=True)
    
    # Ensure grade is lowercase and drop missing values
    df['nutrition_grade_fr'] = df['nutrition_grade_fr'].astype(str).str.lower()
    df.dropna(subset=['nutrition_grade_fr', 'fr_score', 'uk_score'], inplace=True)
    
    # Define colors for nutrition grades (only valid grades)
    grade_colors = {
        'a': 'green',
        'b': 'lightgreen', 
        'c': 'yellow',
        'd': 'orange',
        'e': 'red'
    }
    
    # Apply color mapping and filter out rows with invalid grades
    df['Color'] = df['nutrition_grade_fr'].map(grade_colors)
    df = df[~df['Color'].isna()]
    
    # Memory optimization: Group by unique combinations to reduce data size
    df_grouped = df.groupby(['nutrition_grade_fr', 'fr_score', 'uk_score', 'Color']).size().reset_index(name='Frequency')
    
    # Calculate cumulative frequency (% of total data)
    df_grouped = df_grouped.sort_values('Frequency', ascending=False)
    total_freq = df_grouped['Frequency'].sum()
    df_grouped['Cumulative Percentage'] = df_grouped['Frequency'].cumsum() / total_freq * 100
    
    # Create the base figure
    fig = go.Figure()
    
    # Clear memory of original dataframe
    del df
    
    # Create traces for all combinations but set them invisible initially
    all_traces = []
    
    # Optimization: Process each threshold with minimal data copying
    for i, threshold in enumerate(frequency_thresholds):
        # Convert threshold to percentage for filtering
        threshold_pct = threshold * 100
        
        # Filter by frequency threshold
        filtered_df = df_grouped[df_grouped['Cumulative Percentage'] <= threshold_pct]
        
        if filtered_df.empty:
            continue
            
        # Create bubble trace (sized by frequency)
        bubble_trace = go.Scatter(
            x=filtered_df['fr_score'],
            y=filtered_df['uk_score'],
            mode='markers',
            marker=dict(
                size=filtered_df['Frequency'].clip(lower=3, upper=30),  # Limit size range for better viz
                color=filtered_df['Color'],
                opacity=0.7
            ),
            hovertemplate='<b>Grade:</b> %{text}<br>FR Score: %{x}<br>UK Score: %{y}<br>Count: %{marker.size:.0f}<extra></extra>',
            text=filtered_df['nutrition_grade_fr'].str.upper(),
            name=f'Top {threshold*100:.0f}%',
            visible=(i==0)  # Only first threshold visible by default
        )
        fig.add_trace(bubble_trace)
        all_traces.append({
            'threshold': threshold,
            'mode': 'bubbles',
            'index': len(fig.data) - 1
        })
        
        # Add dots trace (uniform size)
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
            name=f'Top {threshold*100:.0f}%',
            visible=False  # Hidden by default
        )
        fig.add_trace(dots_trace)
        all_traces.append({
            'threshold': threshold,
            'mode': 'dots',
            'index': len(fig.data) - 1
        })
        
        # Calculate regression lines for each grade (more memory efficient)
        for grade in grade_colors.keys():
            grade_data = filtered_df[filtered_df['nutrition_grade_fr'] == grade]
            
            if len(grade_data) < 3:  # Need at least 3 points for regression
                continue
                
            try:
                # Extract scores and reshape for linear regression
                X = grade_data['fr_score'].values.reshape(-1, 1)
                y = grade_data['uk_score'].values
                
                # Fit regression with frequency weighting
                reg = LinearRegression().fit(X, y, sample_weight=grade_data['Frequency'])
                
                # Generate points for the regression line
                x_min = grade_data['fr_score'].min()
                x_max = grade_data['fr_score'].max()
                x_vals = np.linspace(x_min, x_max, 10)  # Reduced to 10 points
                y_vals = reg.predict(x_vals.reshape(-1, 1))
                
                # Create the regression line trace
                reg_trace = go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(
                        color=grade_colors[grade],
                        width=2
                    ),
                    name=f'Grade {grade.upper()}',
                    hovertemplate=f'<b>Grade {grade.upper()}</b><br>FR: %{{x:.1f}}<br>UK: %{{y:.1f}}<extra></extra>',
                    visible=False
                )
                fig.add_trace(reg_trace)
                all_traces.append({
                    'threshold': threshold,
                    'mode': 'regression',
                    'grade': grade,
                    'index': len(fig.data) - 1
                })
            except Exception as e:
                print(f"Error calculating regression for grade {grade}: {e}")
    
    # Create buttons for frequency threshold selection
    threshold_buttons = []
    for threshold in frequency_thresholds:
        # Create visibility arrays for the threshold
        vis_array = [False] * len(fig.data)
        
        # Show bubbles for this threshold by default
        for trace in all_traces:
            if trace['threshold'] == threshold and trace['mode'] == 'bubbles':
                vis_array[trace['index']] = True
        
        threshold_buttons.append(dict(
            label=f"Top {threshold*100:.0f}%",
            method="update",
            args=[
                {"visible": vis_array},
                {"title": f"Nutrition Score Analysis (Top {threshold*100:.0f}%)"}
            ]
        ))
    
    # Create buttons for display type selection
    display_buttons = []
    
    # Bubbles button
    display_buttons.append(dict(
        label="Bubbles",
        method="update",
        args=[
            {"visible": [t['mode'] == 'bubbles' for t in all_traces]},
            {"title": "Nutrition Score Analysis - Bubbles"}
        ]
    ))
    
    # Dots button
    display_buttons.append(dict(
        label="Dots",
        method="update", 
        args=[
            {"visible": [t['mode'] == 'dots' for t in all_traces]},
            {"title": "Nutrition Score Analysis - Dots"}
        ]
    ))
    
    # Regression button
    display_buttons.append(dict(
        label="Regression Lines",
        method="update",
        args=[
            {"visible": [t['mode'] == 'regression' for t in all_traces]},
            {"title": "Nutrition Score Analysis - Regression Lines"}
        ]
    ))
    
    # Update layout with buttons
    fig.update_layout(
        updatemenus=[
            # Threshold selection
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
            # Display type selection
            dict(
                buttons=display_buttons,
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
    
    # Add grade legend as annotations
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
    
    # Update layout
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