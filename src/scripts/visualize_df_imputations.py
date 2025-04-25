import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_missing_values_comparison(df_before, df_after):
    # Calculate missing value percentages
    missing_before = df_before.isna().mean().reset_index()
    missing_before.columns = ['Column', 'Missing_Percentage'] 
    missing_before['Dataset'] = 'Before Imputation'
    
    missing_after = df_after.isna().mean().reset_index()
    missing_after.columns = ['Column', 'Missing_Percentage']
    missing_after['Dataset'] = 'After Imputation'
    
    # Combine data
    missing_df = pd.concat([missing_before, missing_after])
    
    # Create bar chart
    fig = px.bar(
        missing_df, 
        x='Column', 
        y='Missing_Percentage',
        color='Dataset',
        barmode='group',
        title='Missing Values Before vs After Imputation',
        labels={'Missing_Percentage': 'Percentage of Missing Values'},
        height=600,
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='',
        xaxis_title='',
        yaxis_title='Missing Value Percentage',
        yaxis_tickformat='.0%',
        plot_bgcolor='white',
        yaxis=dict(
            gridcolor='lightgrey',
            zeroline=False
        )
    )
    
    return fig

def plot_distribution_comparisons(df_before, df_after, n_cols=3, cat_cols=None, num_cols=None):
    """Create a distribution comparison dashboard for numerical and categorical variables."""
    
    if num_cols is None:
        num_cols = df_before.select_dtypes(include=['number']).columns.tolist()
    
    if cat_cols is None:
        cat_cols = df_before.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculate number of rows needed
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    n_rows = ((n_num + n_cat) // n_cols) + (1 if (n_num + n_cat) % n_cols else 0)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[*num_cols, *cat_cols],
        vertical_spacing=0.08
    )
    
    # Add numerical distributions
    for i, col in enumerate(num_cols):
        row, col_pos = i // n_cols + 1, i % n_cols + 1
        
        # Before imputation histogram
        fig.add_trace(
            go.Histogram(
                x=df_before[col].dropna(),
                name='Before',
                opacity=0.6,
                marker_color='#FF6B6B',
                nbinsx=30,
                showlegend=False  # Explicitly set showlegend=False for each trace
            ),
            row=row, col=col_pos
        )
        
        # After imputation histogram
        fig.add_trace(
            go.Histogram(
                x=df_after[col].dropna(),
                name='After',
                opacity=0.6,
                marker_color='#4ECDC4',
                nbinsx=30,
                showlegend=False  # Explicitly set showlegend=False for each trace
            ),
            row=row, col=col_pos
        )
    
    # Add categorical distributions
    for i, col in enumerate(cat_cols, start=len(num_cols)):
        row, col_pos = i // n_cols + 1, i % n_cols + 1
        
        before_counts = df_before[col].value_counts().head(10)
        after_counts = df_after[col].value_counts().head(10)
        
        # Before imputation bar
        fig.add_trace(
            go.Bar(
                x=before_counts.index,
                y=before_counts.values,
                name='Before',
                marker_color='#FF6B6B',
                showlegend=False  # Explicitly set showlegend=False for each trace
            ),
            row=row, col=col_pos
        )
        
        # After imputation bar
        fig.add_trace(
            go.Bar(
                x=after_counts.index,
                y=after_counts.values,
                name='After',
                marker_color='#4ECDC4',
                showlegend=False  # Explicitly set showlegend=False for each trace
            ),
            row=row, col=col_pos
        )
    
    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        title_text='Distribution Comparisons: Before vs. After Imputation',
        showlegend=False,  # Global setting
        title_x=0.5,
        plot_bgcolor='white',
        barmode='group'
    )
    
    return fig


def plot_pnns_group_changes(df_before, df_after):
    """Visualize the hierarchical relationship between PNNS groups before and after imputation."""
    
    # Count occurrences of each group combination
    before_counts = df_before.groupby(['pnns_groups_1', 'pnns_groups_2']).size().reset_index(name='count_before')
    after_counts = df_after.groupby(['pnns_groups_1', 'pnns_groups_2']).size().reset_index(name='count_after')
    
    # Merge the counts
    combined = pd.merge(before_counts, after_counts, on=['pnns_groups_1', 'pnns_groups_2'], how='outer').fillna(0)
    combined['difference'] = combined['count_after'] - combined['count_before']
    
    # Create percentage change column explicitly as float
    combined['pct_change'] = np.zeros(len(combined), dtype=float)
    
    # Calculate regular percentage changes where before count > 0
    mask_has_before = combined['count_before'] > 0
    combined.loc[mask_has_before, 'pct_change'] = (
        (combined.loc[mask_has_before, 'count_after'] - combined.loc[mask_has_before, 'count_before']) / 
        combined.loc[mask_has_before, 'count_before'] * 100
    )
    
    # For new categories (before=0, after>0), mark as "new" with a high value
    mask_new = (combined['count_before'] == 0) & (combined['count_after'] > 0)
    combined.loc[mask_new, 'pct_change'] = 100.0  # Fixed value for "new" categories as float
    
    # Cap extreme percentage changes to maintain readable color scale
    pct_change_cap = 1000  # More reasonable cap for percentage
    combined['pct_change_capped'] = combined['pct_change'].clip(0, pct_change_cap)  # Only positive values
    
    # Create labels for special cases
    combined['change_label'] = combined['pct_change'].round(1).astype(str) + '%'
    combined.loc[mask_new, 'change_label'] = 'New'
    
    # Create a sunburst chart
    fig = px.sunburst(
        combined,
        path=['pnns_groups_1', 'pnns_groups_2'],
        values='count_after',
        color='pct_change_capped',
        color_continuous_scale='plasma',  # Changed to plasma colorscale
        range_color=[0, pct_change_cap],  # Only positive range
        title='PNNS Groups After Imputation',
        hover_data={
            'count_before': True,
            'count_after': True,
            'difference': True,
            'pct_change': False,
            'pct_change_capped': False,
            'change_label': True
        }
    )
    
    # Custom hover template to better explain what's shown
    hovertemplate = (
        "<b>%{label}</b><br>" +
        "Before: %{customdata[0]}<br>" +
        "After: %{customdata[1]}<br>" +
        "Change: %{customdata[2]}" + 
        " (%{customdata[5]})<br>" +
        "<extra></extra>"
    )
    
    fig.update_traces(hovertemplate=hovertemplate)
    
    fig.update_layout(
        height=700, 
        width=700,
        title_x=0.5,
        coloraxis_colorbar=dict(
            title="% Growth",
            tickvals=[0, 250, 500, 750, 1000],
            ticktext=["0%", "250%", "500%", "750%", "≥1000%"]
        )
    )
    
    return fig

def create_stats_comparison_table(df_before, df_after):
    """Create an interactive table showing statistical changes."""
    
    numerical_cols = df_before.select_dtypes(include=['number']).columns.tolist()
    
    stats = []
    for col in numerical_cols:
        # Calculate statistics
        before_mean = df_before[col].mean()
        after_mean = df_after[col].mean()
        before_median = df_before[col].median()
        after_median = df_after[col].median()
        before_std = df_before[col].std()
        after_std = df_after[col].std()
        
        # Calculate changes
        mean_diff = after_mean - before_mean
        mean_pct = (mean_diff / before_mean * 100) if before_mean != 0 else 0
        
        stats.append({
            'Column': col,
            'Before Mean': round(before_mean, 2),
            'After Mean': round(after_mean, 2),
            'Mean Diff': round(mean_diff, 2),
            'Mean % Change': round(mean_pct, 1),
            'Before Median': round(before_median, 2),
            'After Median': round(after_median, 2),
            'Before Std': round(before_std, 2),
            'After Std': round(after_std, 2)
        })
    
    # Create table
    stats_df = pd.DataFrame(stats)
    
    # Create the table visualization
    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=list(stats_df.columns),
                fill_color='#6D9EC1',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[stats_df[col] for col in stats_df.columns],
                fill_color=[
                    ['#f2f2f2'] * len(stats_df),
                    ['#f2f2f2'] * len(stats_df),
                    ['#f2f2f2'] * len(stats_df),
                    ['#f2f2f2'] * len(stats_df),
                    [
                        '#FF9E9E' if x > 5 else 
                        '#FFD199' if x > 2 else
                        '#FFFB99' if x > 0 else
                        '#B9FFA8' if x >= -2 else
                        '#A8D8FF' for x in stats_df['Mean % Change']
                    ],
                    ['#f2f2f2'] * len(stats_df),
                    ['#f2f2f2'] * len(stats_df),
                    ['#f2f2f2'] * len(stats_df),
                    ['#f2f2f2'] * len(stats_df)
                ],
                align='center',
                font_size=11
            )
        )
    ])
    
    fig.update_layout(
        title='Statistical Changes After Imputation',
        height=200 + len(stats_df) * 25,
        title_x=0.5
    )
    
    return fig

