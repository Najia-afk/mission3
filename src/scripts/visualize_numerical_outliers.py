import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats

def create_interactive_outlier_visualization(df, outlier_threshold=1.5, use_log_scale=True):
    """
    Create an interactive visualization to explore outliers in numeric columns
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        outlier_threshold (float): IQR multiplier for outlier detection (default=1.5)
        use_log_scale (bool): Use logarithmic scale for highly skewed data (default=True)
        
    Returns:
        tuple: (summary_df, df_clean) - Outlier summary DataFrame and cleaned DataFrame
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found in the dataframe")
        return
    
    # Create a copy of the dataframe for outlier handling
    df_clean = df.copy()
    
    # Create dictionary to store outliers info
    outlier_info = {}
    stats_info = {}
    
    # Identify outliers using IQR method
    for col in numeric_cols:
        if df[col].count() == 0:
            continue
            
        # Check if the column has all positive values for log transform
        can_use_log = (df[col].min() > 0) if use_log_scale else False
        
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_threshold * IQR
        upper_bound = Q3 + outlier_threshold * IQR
        
        # Identify outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        
        # Calculate statistics
        mean = df[col].mean()
        median = df[col].median()
        std_dev = df[col].std()
        clean_mean = df[col][~outliers].mean() if (~outliers).any() else np.nan
        clean_median = df[col][~outliers].median() if (~outliers).any() else np.nan
        clean_std_dev = df[col][~outliers].std() if (~outliers).any() else np.nan
        
        # Store statistics
        stats_info[col] = {
            'mean': mean,
            'median': median, 
            'std_dev': std_dev,
            'clean_mean': clean_mean,
            'clean_median': clean_median,
            'clean_std_dev': clean_std_dev,
            'can_use_log': can_use_log,
            'skewness': df[col].skew()
        }
        
        # Store outlier information
        outlier_info[col] = {
            'outlier_count': outliers.sum(),
            'outlier_percentage': outliers.sum() / df[col].count() * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        }
        
        # Cap outliers in the cleaned dataframe
        df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
        df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
    
    # Create a table with outlier summary
    summary_df = pd.DataFrame({
        'Column': [col for col in outlier_info.keys()],
        'Outlier Count': [info['outlier_count'] for info in outlier_info.values()],
        'Outlier Percentage': [f"{info['outlier_percentage']:.2f}%" for info in outlier_info.values()],
        'Skewness': [stats_info[col]['skewness'] for col in outlier_info.keys()],
        'Mean (with outliers)': [stats_info[col]['mean'] for col in outlier_info.keys()],
        'Mean (w/o outliers)': [stats_info[col]['clean_mean'] for col in outlier_info.keys()],
        'StdDev (with outliers)': [stats_info[col]['std_dev'] for col in outlier_info.keys()],
        'StdDev (w/o outliers)': [stats_info[col]['clean_std_dev'] for col in outlier_info.keys()],
        'Lower Bound': [info['lower_bound'] for info in outlier_info.values()],
        'Upper Bound': [info['upper_bound'] for info in outlier_info.values()]
    })
    
    # Sort by outlier percentage
    summary_df = summary_df.sort_values(by='Outlier Count', ascending=False)
    
    print("Outlier Summary (threshold multiplier = {}):".format(outlier_threshold))
    display(summary_df)
    
    # Create the interactive figure - side by side layout
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=("Box Plot (with outliers)", "Distribution (without outliers)"),
        horizontal_spacing=0.1,
        specs=[[{"type": "box"}, {"type": "histogram"}]]
    )
    
    # Initialize with the first numeric column
    first_col = numeric_cols[0]
    
    # PRE-COMPUTE BOXPLOT STATISTICS instead of using raw data
    for i, col in enumerate(numeric_cols):
        # Only calculate for non-empty columns
        if df[col].count() == 0:
            continue
            
        # Pre-calculate boxplot statistics
        q1 = df[col].quantile(0.25)
        median = df[col].median()
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - outlier_threshold * iqr
        upper_fence = q3 + outlier_threshold * iqr
        
        # Get only the outliers for plotting (much fewer points)
        outlier_values = df[col][(df[col] < lower_fence) | (df[col] > upper_fence)].sample(
            min(100, ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()), 
            random_state=42
        ).tolist() if ((df[col] < lower_fence) | (df[col] > upper_fence)).any() else []
        
        # Pre-aggregate the histogram data into bins
        if i == 0:
            # For the first column, add to the plot
            fig.add_trace(
                go.Box(
                    name=col,
                    marker_color='red',
                    visible=True,
                    boxpoints='outliers',
                    jitter=0,
                    pointpos=0,
                    # Use pre-computed statistics
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower_fence],
                    upperfence=[upper_fence],
                    y=outlier_values  # Only include outlier points
                ),
                row=1, col=1
            )
            
            # Calculate histogram bins for cleaned data
            clean_data = df_clean[col].dropna()
            if len(clean_data) > 0:
                # Use Sturges' formula for bin count
                bin_count = int(np.ceil(np.log2(len(clean_data))) + 1)
                bin_count = min(50, max(10, bin_count))  # Keep bins reasonable
                
                hist, bin_edges = np.histogram(clean_data, bins=bin_count, density=True)
                
                # Use the center of each bin for x values
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=hist,
                        name='Without Outliers',
                        marker_color='blue',
                        opacity=0.7,
                        visible=True
                    ),
                    row=1, col=2
                )
                
                # Add normal distribution curve if appropriate
                clean_mean = stats_info[col]['clean_mean']
                clean_std = stats_info[col]['clean_std_dev']
                
                if pd.notna(clean_mean) and pd.notna(clean_std) and clean_std > 0:
                    x_range = np.linspace(min(bin_edges), max(bin_edges), 100)
                    pdf = stats.norm.pdf(x_range, loc=clean_mean, scale=clean_std)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=pdf,
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='green', dash='dash'),
                            visible=True
                        ),
                        row=1, col=2
                    )
    
    # Add statistical annotations for the first column
    stats_annotations = [
        f"<b>With Outliers:</b><br>" +
        f"Mean: {stats_info[first_col]['mean']:.2f}<br>" +
        f"Median: {stats_info[first_col]['median']:.2f}<br>" +
        f"StdDev: {stats_info[first_col]['std_dev']:.2f}<br>" +
        f"Skewness: {stats_info[first_col]['skewness']:.2f}<br>" +
        f"<br><b>Without Outliers:</b><br>" +
        f"Mean: {stats_info[first_col]['clean_mean']:.2f}<br>" +
        f"Median: {stats_info[first_col]['clean_median']:.2f}<br>" +
        f"StdDev: {stats_info[first_col]['clean_std_dev']:.2f}"
    ]
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.4,
        xanchor="left", yanchor="middle",
        text=stats_annotations[0],
        showarrow=False,
        font=dict(size=12),
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        bgcolor="white",
        opacity=0.8
    )
    
    # Create dropdown menu buttons
    dropdown_buttons = []
    
    # Create dropdown entries for each numeric column
    for i, col in enumerate(numeric_cols):
        if df[col].count() == 0:
            continue
            
        # Pre-calculate statistics as we did for the first column
        q1 = df[col].quantile(0.25)
        median = df[col].median()
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - outlier_threshold * iqr
        upper_fence = q3 + outlier_threshold * iqr
        
        # Get only outliers (limited sample)
        outlier_values = df[col][(df[col] < lower_fence) | (df[col] > upper_fence)].sample(
            min(100, ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()), 
            random_state=42
        ).tolist() if ((df[col] < lower_fence) | (df[col] > upper_fence)).any() else []
        
        # Pre-calculate histogram data
        clean_data = df_clean[col].dropna()
        bin_count = int(np.ceil(np.log2(len(clean_data))) + 1) if len(clean_data) > 0 else 10
        bin_count = min(50, max(10, bin_count))
        
        hist_data = {}
        if len(clean_data) > 0:
            hist, bin_edges = np.histogram(clean_data, bins=bin_count, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_data = {'centers': bin_centers, 'heights': hist, 'edges': bin_edges}
            
        # Normal distribution data if applicable
        clean_mean = stats_info[col]['clean_mean']
        clean_std = stats_info[col]['clean_std_dev']
        
        normal_curve = None
        if pd.notna(clean_mean) and pd.notna(clean_std) and clean_std > 0 and len(clean_data) > 0:
            x_range = np.linspace(min(bin_edges), max(bin_edges), 100)
            pdf = stats.norm.pdf(x_range, loc=clean_mean, scale=clean_std)
            normal_curve = {'x': x_range, 'y': pdf}
        
        # Create annotation text
        annotation_text = (
            f"<b>With Outliers:</b><br>" +
            f"Mean: {stats_info[col]['mean']:.2f}<br>" +
            f"Median: {stats_info[col]['median']:.2f}<br>" +
            f"StdDev: {stats_info[col]['std_dev']:.2f}<br>" +
            f"Skewness: {stats_info[col]['skewness']:.2f}<br>" +
            f"<br><b>Without Outliers:</b><br>" +
            f"Mean: {stats_info[col]['clean_mean']:.2f}<br>" +
            f"Median: {stats_info[col]['clean_median']:.2f}<br>" +
            f"StdDev: {stats_info[col]['clean_std_dev']:.2f}"
        )
        
        # Create args for dropdown button
        button_args = [
            # Update boxplot
            {
                'y': [outlier_values],
                'q1': [q1], 'median': [median], 'q3': [q3],
                'lowerfence': [lower_fence], 'upperfence': [upper_fence],
                'name': col
            },
            # Update histogram
            {'x': hist_data.get('centers', []), 'y': hist_data.get('heights', []), 'name': col}
        ]
        
        # Add normal curve if available
        if normal_curve:
            button_args.append({
                'x': normal_curve['x'], 'y': normal_curve['y'],
                'line': {'color': 'green', 'dash': 'dash'}
            })
        else:
            button_args.append({'x': [], 'y': []})
            
        # Update annotation
        button_args.append({'text': annotation_text})
        
        # Update plot title
        title_update = f"Outlier Analysis for {col} (threshold={outlier_threshold})"
        
        # Create the dropdown button
        dropdown_buttons.append(
            dict(
                method='update',
                label=col,
                args=[
                    {
                        # Update boxplot data
                        'y': [
                            # First trace (boxplot)
                            outlier_values  
                        ],
                        # Update boxplot stats
                        'q1': [[q1]],
                        'median': [[median]],
                        'q3': [[q3]],
                        'lowerfence': [[lower_fence]],
                        'upperfence': [[upper_fence]],
                        'name': [[col]],
                        
                        # Update histogram data (second trace)
                        'x': [None, hist_data.get('centers', []), normal_curve.get('x', []) if normal_curve else []],
                        'y': [None, hist_data.get('heights', []), normal_curve.get('y', []) if normal_curve else []]
                    },
                    {
                        # Replace the entire annotations array
                        "annotations": [
                            dict(
                                xref="paper", yref="paper",
                                x=1.02, y=0.4,
                                xanchor="left", yanchor="middle",
                                text=annotation_text,
                                showarrow=False,
                                font=dict(size=12),
                                bordercolor="black",
                                borderwidth=1,
                                borderpad=10,
                                bgcolor="white",
                                opacity=0.8
                            )
                        ],
                        # Update title
                        'title': title_update
                    }
                ]
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Outlier Analysis for {first_col} (threshold={outlier_threshold})",
        showlegend=True,
        title_x=0.5,
        margin=dict(r=200),
        height=600,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.5,
                y=1.1,
                xanchor="center",
                yanchor="top"
            )
        ]
    )
    
    # Update axes with correct labels
    fig.update_xaxes(title_text="Variable Value", row=1, col=2)
    fig.update_yaxes(title_text="Variable Value", row=1, col=1)  # Y-axis for boxplot shows the variable values
    fig.update_yaxes(title_text="Frequency Density", row=1, col=2)  # Y-axis for histogram shows frequency/density
    
    # Show the figure
    fig.show()
    
    return summary_df, df_clean