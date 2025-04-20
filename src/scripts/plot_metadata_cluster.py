import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import random

def plot_metadata_clusters(metadata):
    """
    Creates a standalone interactive plot with organized button groups:
    - A top row for fill rate selection
    - Parameter buttons below each fill rate
    
    Parameters:
        metadata (pd.DataFrame): The metadata DataFrame to visualize
        
    Returns:
        plotly.graph_objects.Figure: Standalone interactive figure
    """
    # Parameter options
    min_fill_values = [0, 25, 50, 75]
    eps_values = [1, 2, 3, 5, 10]
    
    # Create a base figure
    fig = go.Figure()
    
    # Storage for all traces
    all_traces = []
    
    # Generate all traces and organize by parameter combination
    for min_fill in min_fill_values:
        # Filter data for this min_fill
        filtered_data = metadata[(metadata['Fill Rate (%)'] >= min_fill) & 
                                (metadata['Fill Rate (%)'] <= 100)].copy()
        
        # Skip if no data for this min_fill
        if len(filtered_data) == 0:
            continue
            
        for eps in eps_values:
            # Apply clustering
            X = filtered_data[['Fill Rate (%)', 'Unique Rate (%)']].values
            clustering = DBSCAN(eps=eps, min_samples=2).fit(X)
            filtered_data['Cluster'] = clustering.labels_
            
            # Count clusters
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            # Add trace for this parameter combination
            scatter = go.Scatter(
                x=filtered_data['Fill Rate (%)'],
                y=filtered_data['Unique Rate (%)'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=filtered_data['Cluster'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Cluster')
                ),
                text=[f"<b>Column:</b> {col}<br><b>Fill Rate:</b> {fill:.1f}%<br><b>Unique Rate:</b> {unique:.1f}%<br><b>Cluster:</b> {cluster}" 
                      for col, fill, unique, cluster in zip(
                          filtered_data['Column Name'], 
                          filtered_data['Fill Rate (%)'],
                          filtered_data['Unique Rate (%)'],
                          filtered_data['Cluster'])],
                hoverinfo='text',
                hovertemplate='%{text}<extra></extra>',
                name=f"Min Fill: {min_fill}%, eps: {eps}",
                visible=False  # All traces start invisible
            )
            
            fig.add_trace(scatter)
            
            # Store trace information
            all_traces.append({
                'min_fill': min_fill,
                'eps': eps,
                'n_clusters': n_clusters,
                'index': len(all_traces)
            })
    
    # Make the first trace visible by default
    if all_traces:
        fig.data[all_traces[0]['index']].visible = True
    
    # Create buttons for main fill rate selector (top row)
    fill_rate_buttons = []
    
    # Position variables
    button_width = 0.15  # Width of each fill rate button group 
    padding = 0.1      # Space between button groups
    start_x = 0.0       # Starting x position
    
    # For each fill rate, create a button in the top row
    for i, min_fill in enumerate(min_fill_values):
        # Get traces for this min_fill
        traces_with_min_fill = [t for t in all_traces if t['min_fill'] == min_fill]
        
        # Skip if no traces for this min_fill
        if not traces_with_min_fill:
            continue
        
        # Calculate button position
        x_pos = start_x + (button_width + padding) * i
        
        # By default, select the first eps value for this min_fill
        default_eps = traces_with_min_fill[0]['eps']
        default_trace = traces_with_min_fill[0]
        
        # Create visibility list
        visibility = [False] * len(fig.data)
        visibility[default_trace['index']] = True
        
        # Add button for this min_fill
        fill_rate_buttons.append(
            dict(
                type="buttons",
                direction="down",
                active=0,
                buttons=[dict(
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"Metadata Clusters (Min Fill: {min_fill}%, Eps: {default_eps})"}
                    ],
                    label=f"Fill ≥ {min_fill}%"
                )],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=x_pos,
                y=1.15,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(158, 202, 225, 0.8)',
                bordercolor='rgba(68, 68, 68, 0.8)',
                font=dict(size=13, color='black', family="Arial"),
                borderwidth=2
            )
        )
        
        # Create buttons for each eps value under this fill rate
        eps_buttons = []
        for j, eps in enumerate(eps_values):
            matching_traces = [t for t in all_traces if t['min_fill'] == min_fill and t['eps'] == eps]
            if not matching_traces:
                continue
            
            trace = matching_traces[0]
            
            # Create visibility list
            eps_visibility = [False] * len(fig.data)
            eps_visibility[trace['index']] = True
            
            # Add button for this combination
            eps_buttons.append(
                dict(
                    args=[
                        {"visible": eps_visibility},
                        {"title": f"Metadata Clusters (Min Fill: {min_fill}%, Eps: {eps}, Clusters: {trace['n_clusters']})"}
                    ],
                    label=f"eps={eps}",
                    method="update"
                )
            )
        
        # Add eps buttons below this fill rate button
        if eps_buttons:
            fill_rate_buttons.append(
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    buttons=eps_buttons,
                    pad={"r": 10, "t": 10, "b": 10},
                    showactive=True,
                    x=x_pos,
                    y=1.06,
                    xanchor="left",
                    yanchor="top",
                    bgcolor='rgba(230, 230, 230, 0.8)',
                    bordercolor='rgba(68, 68, 68, 0.5)',
                    font=dict(size=12),
                    borderwidth=1
                )
            )
    
    # Add buttons to the figure
    fig.update_layout(updatemenus=fill_rate_buttons)
    
    # Update general layout
    fig.update_layout(
        title={
            'text': 'Metadata Clustering Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'family': 'Arial', 'weight': 'bold'}
        },
        xaxis_title={
            'text': 'Fill Rate (%)',
            'font': {'size': 14, 'family': 'Arial'}
        },
        yaxis_title={
            'text': 'Unique Rate (%)',
            'font': {'size': 14, 'family': 'Arial'}
        },
        height=700,
        margin=dict(t=150),  # Extra top margin for buttons
        hovermode='closest',
        template='plotly_white',
        legend_title_text='Clusters',
        font=dict(family="Arial")
    )
    
    return fig