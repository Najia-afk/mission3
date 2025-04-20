import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import plotly.express as px

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
        return None, None
    
    # Create a copy of the dataframe for outlier handling
    df_clean = df.copy()
    
    # Create dictionary to store outliers info
    outlier_info = {}
    stats_info = {}
    
    # Identify outliers using IQR method
    for col in numeric_cols:
        if df[col].count() == 0:  # Skip columns with all NaN values
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
        clean_mean = df_clean[col][~outliers].mean()
        clean_median = df_clean[col][~outliers].median()
        clean_std_dev = df_clean[col][~outliers].std()
        
        # Store statistics
        stats_info[col] = {
            'mean': mean,
            'median': median, 
            'std_dev': std_dev,
            'clean_mean': clean_mean,
            'clean_median': clean_median,
            'clean_std_dev': clean_std_dev,
            'can_use_log': can_use_log,
            'skewness': df[col].skew()  # Add skewness to stats
        }
        
        # Store outlier information
        outlier_info[col] = {
            'outlier_count': outliers.sum(),
            'outlier_percentage': outliers.sum() / df[col].count() * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': outliers[outliers].index,
            'outlier_points': df[col][outliers].tolist()  # Store actual outlier values
        }
        
        # Cap outliers in the clean dataframe
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
    
    # Add box plot for the first column (with outliers)
    fig.add_trace(
        go.Box(
            y=df[first_col].dropna(), 
            name='With Outliers',
            boxmean=True,  # adds a marker for the mean
            marker_color='red',
            visible=True,
            boxpoints='outliers',  # Only show outlier points
            jitter=0,  # No jitter needed if only showing outliers
            pointpos=0
        ),
        row=1, col=1
    )
    
    # Add histogram for the first column (without outliers)
    fig.add_trace(
        go.Histogram(
            x=df_clean[first_col].dropna(),
            name='Without Outliers',
            marker_color='blue',
            opacity=0.7,
            visible=True,
            histnorm='probability density'
        ),
        row=1, col=2
    )
    
    # Add a normal distribution curve on the histogram
    clean_mean = stats_info[first_col]['clean_mean']
    clean_std = stats_info[first_col]['clean_std_dev']

    # Check for valid parameters before calculating normal distribution
    if pd.notna(clean_mean) and pd.notna(clean_std) and clean_std > 0:
        x_range = np.linspace(
            df_clean[first_col].min(), 
            df_clean[first_col].max(), 
            100
        )
        normal_y = stats.norm.pdf(x_range, loc=clean_mean, scale=clean_std)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='green', width=2),
                visible=True
            ),
            row=1, col=2
        )
    else:
        print(f"Warning: Cannot calculate normal distribution for {first_col}. Mean: {clean_mean}, StdDev: {clean_std}")
    
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
    
    # Add traces for other columns (initially hidden)
    for i, col in enumerate(numeric_cols[1:], 1):
        # Box plot (with outliers)
        fig.add_trace(
            go.Box(
                y=df[col].dropna(), 
                name='With Outliers',
                boxmean=True,
                marker_color='red',
                visible=False,
                boxpoints='outliers',  # Only show outlier points
                jitter=0,  # No jitter needed if only showing outliers
                pointpos=0
            ),
            row=1, col=1
        )
        
        # Histogram (without outliers)
        fig.add_trace(
            go.Histogram(
                x=df_clean[col].dropna(),
                name='Without Outliers',
                marker_color='blue',
                opacity=0.7,
                visible=False,
                histnorm='probability density'
            ),
            row=1, col=2
        )
        
        # Normal distribution curve for the cleaned data
        clean_mean = stats_info[col]['clean_mean']
        clean_std = stats_info[col]['clean_std_dev']
        
        # Add check for valid parameters here as well
        if pd.notna(clean_mean) and pd.notna(clean_std) and clean_std > 0:
            try:
                x_range = np.linspace(
                    df_clean[col].min(), 
                    df_clean[col].max(), 
                    100
                )
                normal_y = stats.norm.pdf(x_range, loc=clean_mean, scale=clean_std)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=normal_y,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='green', width=2),
                        visible=False
                    ),
                    row=1, col=2
                )
            except Exception as e:
                print(f"Warning: Could not create normal curve for {col}: {e}")
                # Add an empty trace to maintain the correct count
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='green', width=2),
                        visible=False
                    ),
                    row=1, col=2
                )
        else:
            print(f"Warning: Cannot calculate normal distribution for {col}. Mean: {clean_mean}, StdDev: {clean_std}")
            # Add an empty trace to maintain the correct count
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='green', width=2),
                    visible=False
                ),
                row=1, col=2
            )
        
        # Add statistical annotations for each column
        stats_annotations.append(
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
    
    # Create dropdown menu for column selection
    dropdown_buttons = []
    for i, col in enumerate(numeric_cols):
        # Calculate which traces should be visible
        visible = [False] * (len(numeric_cols) * 3)  # 3 traces per column: box, hist, scatter
        # Set visibility for the current column (index i)
        base_idx = i * 3
        if i == 0:
            base_idx = 0
        visible[base_idx] = True      # Box plot
        visible[base_idx + 1] = True  # Histogram
        visible[base_idx + 2] = True  # Normal distribution curve
        
        dropdown_buttons.append(
            dict(
                label=col,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Outlier Analysis for {col}",
                        "annotations": [
                            dict(
                                xref="paper", yref="paper",
                                x=1.02, y=0.4,
                                xanchor="left", yanchor="middle",
                                text=stats_annotations[i],
                                showarrow=False,
                                font=dict(size=12),
                                bordercolor="black",
                                borderwidth=1,
                                borderpad=10,
                                bgcolor="white",
                                opacity=0.8
                            )
                        ]
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
        updatemenus=[
            # Column selection dropdown
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top"
            )
        ]
    )
    
    # Update axes with correct labels
    fig.update_xaxes(title_text="Variable Value", row=1, col=2)
    fig.update_yaxes(title_text="Variable Value", row=1, col=1)  # Y-axis for boxplot shows the variable values
    fig.update_yaxes(title_text="Frequency Density", row=1, col=2)  # Y-axis for histogram shows frequency/density
    
    # Apply log scale for highly skewed columns
    if use_log_scale and stats_info[first_col]['can_use_log'] and abs(stats_info[first_col]['skewness']) > 2:
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_layout(title=f"Outlier Analysis for {first_col} (threshold={outlier_threshold}, log scale)")
    
    # Apply log scale to other columns when selected
    for i, col in enumerate(numeric_cols):
        # Store log scale info in the dropdown selection
        if use_log_scale and stats_info[col]['can_use_log'] and abs(stats_info[col]['skewness']) > 2:
            dropdown_buttons[i]['args'][1]['yaxis'] = {"type": "log", "title": "Variable Value (log scale)"}
            dropdown_buttons[i]['args'][1]['title'] = f"Outlier Analysis for {col} (threshold={outlier_threshold}, log scale)"
        else:
            dropdown_buttons[i]['args'][1]['yaxis'] = {"type": "linear", "title": "Variable Value"}
            dropdown_buttons[i]['args'][1]['title'] = f"Outlier Analysis for {col} (threshold={outlier_threshold})"
    
    # Update dropdown menu with modified buttons
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top"
            )
        ]
    )
    
    # Show the figure
    fig.show()
    
    return summary_df, df_clean