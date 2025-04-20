import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def identify_nutrition_outliers(df, nutrient_limits):
    """
    Identify outliers in nutritional data based on established domain knowledge limits
    
    Parameters:
        df (pd.DataFrame): Input dataframe with nutritional columns
        
    Returns:
        tuple: (summary_df, df_clean) - Outlier summary DataFrame and cleaned DataFrame
    """

    # Create lower bounds (0 for most nutrients as they can't be negative)
    lower_bounds = {col: 0 for col in nutrient_limits.keys()}
    
    # Create a copy of the dataframe for outlier handling
    df_clean = df.copy()
    
    # Dictionaries to store outlier information
    outlier_info = {}
    stats_info = {}
    
    # Check each nutrient column against the limits
    for col, upper_bound in nutrient_limits.items():
        if col not in df.columns:
            continue
            
        lower_bound = lower_bounds[col]
        
        # Identify outliers based on limits (below lower or above upper)
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        
        # Skip if no data in column
        if df[col].count() == 0:
            continue
            
        # Calculate statistics
        mean = df[col].mean()
        median = df[col].median()
        std_dev = df[col].std()
        clean_mean = df_clean[col][~outliers].mean()
        clean_median = df_clean[col][~outliers].median()
        clean_std_dev = df_clean[col][~outliers].std()
        
        # Calculate percentage change in mean due to outliers
        if clean_mean != 0:  # Avoid division by zero
            mean_percent_change = ((mean - clean_mean) / clean_mean) * 100
        else:
            mean_percent_change = 0
        
        # Store statistics
        stats_info[col] = {
            'mean': mean,
            'median': median, 
            'std_dev': std_dev,
            'clean_mean': clean_mean,
            'clean_median': clean_median,
            'clean_std_dev': clean_std_dev,
            'mean_percent_change': mean_percent_change
        }
        
        # Store outlier information
        outlier_info[col] = {
            'outlier_count': outliers.sum(),
            'outlier_percentage': outliers.sum() / df[col].count() * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'extreme_min': df[col][outliers & (df[col] < lower_bound)].min() if any(outliers & (df[col] < lower_bound)) else None,
            'extreme_max': df[col][outliers & (df[col] > upper_bound)].max() if any(outliers & (df[col] > upper_bound)) else None,
            'num_below_lower': (outliers & (df[col] < lower_bound)).sum(),
            'num_above_upper': (outliers & (df[col] > upper_bound)).sum()
        }
        
        # Cap outliers in the clean dataframe
        #df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
        #df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        # Alternatively, set outliers to NaN
        df_clean.loc[df_clean[col] < lower_bound, col] = np.nan
        df_clean.loc[df_clean[col] > upper_bound, col] = np.nan
    
    # Create a table with outlier summary
    summary_df = pd.DataFrame({
        'Nutrient': [col for col in outlier_info.keys()],
        'Outlier Count': [info['outlier_count'] for info in outlier_info.values()],
        'Outlier %': [f"{info['outlier_percentage']:.2f}%" for info in outlier_info.values()],
        'Below Min': [info['num_below_lower'] for info in outlier_info.values()],
        'Above Max': [info['num_above_upper'] for info in outlier_info.values()],
        'Max Limit': [info['upper_bound'] for info in outlier_info.values()],
        'Extreme Min': [info['extreme_min'] for info in outlier_info.values()],
        'Extreme Max': [info['extreme_max'] for info in outlier_info.values()],
        'Mean (with outliers)': [stats_info[col]['mean'] for col in outlier_info.keys()],
        'Mean (w/o outliers)': [stats_info[col]['clean_mean'] for col in outlier_info.keys()],
        'Mean % Change': [stats_info[col]['mean_percent_change'] for col in outlier_info.keys()]
    })
    
    # Sort by outlier count
    summary_df = summary_df.sort_values(by='Outlier Count', ascending=False)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Outlier Counts by Nutrient", "Impact on Mean Values (% Change)"),
        vertical_spacing=0.3,
        specs=[[{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # Add bar chart for outlier counts
    fig.add_trace(
        go.Bar(
            x=summary_df['Nutrient'],
            y=summary_df['Outlier Count'],
            name='Outlier Count',
            marker_color='red',
            text=summary_df['Outlier %'],
            hovertemplate='%{x}<br>Outliers: %{y}<br>(%{text})<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add bar chart for mean percent change
    fig.add_trace(
        go.Bar(
            x=summary_df['Nutrient'],
            y=summary_df['Mean % Change'],
            name='Mean % Change',
            marker_color='purple',
            text=[f"{x:.1f}%" for x in summary_df['Mean % Change']],
            textposition='auto',
            hovertemplate='%{x}<br>Mean % Change: %{text}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Nutrition Data Outlier Analysis (Based on Domain Knowledge Limits)",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        showlegend=True,
        annotations=[
            dict(
                x=1.05,
                y=0.25,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="middle",
                text="<b>Effect of Outliers</b><br><br>" + 
                     "Positive %: Outliers increase mean<br><br>" + 
                     "Negative %: Outliers decrease mean",
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=10,
                opacity=0.9
            )
        ],
        margin=dict(r=200)  # Add right margin for annotation
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Outlier Count", row=1, col=1)
    fig.update_yaxes(title_text="Mean % Change", row=2, col=1)
    
    # Display the figure
    fig.show()
    
    print("Outlier Summary (Based on Domain Knowledge Limits):")
    display(summary_df)
    
    return summary_df, df_clean