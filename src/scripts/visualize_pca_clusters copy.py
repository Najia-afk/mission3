import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def perform_pca_analysis(df, 
                        numeric_cols=None, 
                        color_col='nutrition_grade_fr',
                        n_components=3,
                        random_state=42):
    """
    Perform PCA analysis on nutritional data and create interactive visualizations.
    
    Args:
        df: DataFrame containing nutritional data
        numeric_cols: List of numeric columns to include in PCA (if None, uses all numeric columns)
        color_col: Column to use for coloring points (default: nutrition grade)
        n_components: Number of principal components to extract
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (pca_results, feature_importance, pca_fig, biplot_fig)
    """
    # Make a copy of the dataframe
    df_pca = df.copy()
    
    # Select numeric columns if not specified
    if numeric_cols is None:
        # Get all numeric columns except scores which might be derived from the grade
        numeric_cols = df_pca.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if 'score' not in col.lower() or col == 'nutrition-score-fr_100g']
    
    # Handle any remaining missing values for PCA
    imputer = SimpleImputer(strategy='median')
    df_numeric = pd.DataFrame(
        imputer.fit_transform(df_pca[numeric_cols]), 
        columns=numeric_cols,
        index=df_pca.index
    )
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df_pca.index
    )
    
    # Add the color column if it exists
    if color_col in df_pca.columns:
        pca_df[color_col] = df_pca[color_col]
        
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Calculate feature importance
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numeric_cols
    )
    
    # Create PCA scatter plot figure
    pca_fig = go.Figure()
    
    # Check if color column exists and is categorical
    if color_col in df_pca.columns:
        # If it's a categorical column, use discrete colors
        if df_pca[color_col].dtype == 'object' or df_pca[color_col].nunique() < 10:
            color_discrete_map = {
                'a': '#038141', 'b': '#85bb2f', 'c': '#fecb02', 
                'd': '#ee8100', 'e': '#e63e11'
            }
            
            # Add traces for each nutrition grade
            for grade in sorted(df_pca[color_col].dropna().unique()):
                subset = pca_df[pca_df[color_col] == grade]
                
                pca_fig.add_trace(go.Scatter(
                    x=subset['PC1'],
                    y=subset['PC2'],
                    mode='markers',
                    name=f'Grade {grade.upper()}' if grade in ['a', 'b', 'c', 'd', 'e'] else grade,
                    marker=dict(
                        color=color_discrete_map.get(grade.lower(), '#777777'),
                        size=8,
                        opacity=0.7
                    ),
                    text=subset.index,
                    hoverinfo='text'
                ))
        else:
            # If it's a numerical column, use continuous colors
            pca_fig = px.scatter(
                pca_df, x='PC1', y='PC2', 
                color=color_col,
                hover_data=[color_col]
            )
    else:
        # If no color column specified, create simple scatter plot
        pca_fig = px.scatter(
            pca_df, x='PC1', y='PC2',
            hover_data=['PC1', 'PC2', 'PC3']
        )
    
    # Update layout
    pca_fig.update_layout(
        title=f'PCA of Nutritional Data<br><sup>Total Variance Explained: {sum(explained_variance):.1f}%</sup>',
        xaxis_title=f'PC1 ({explained_variance[0]:.1f}%)',
        yaxis_title=f'PC2 ({explained_variance[1]:.1f}%)',
        legend_title=color_col if color_col in df_pca.columns else None,
        height=600,
        width=900
    )
    
    # Create a biplot (PCA with feature vectors)
    biplot_fig = create_biplot(pca, scaled_data, numeric_cols, pca_df, color_col, explained_variance)
    
    return pca_df, feature_importance, pca_fig, biplot_fig

def create_biplot(pca, scaled_data, feature_names, pca_df, color_col, explained_variance):
    """Create a biplot showing both observations and feature vectors."""
    
    fig = go.Figure()
    
    # Add scatter points for observations
    if color_col in pca_df.columns:
        # If color column is categorical
        if pca_df[color_col].dtype == 'object' or pca_df[color_col].nunique() < 10:
            color_discrete_map = {
                'a': '#038141', 'b': '#85bb2f', 'c': '#fecb02', 
                'd': '#ee8100', 'e': '#e63e11'
            }
            
            # Add traces for each nutrition grade
            for grade in sorted(pca_df[color_col].dropna().unique()):
                subset = pca_df[pca_df[color_col] == grade]
                
                fig.add_trace(go.Scatter(
                    x=subset['PC1'],
                    y=subset['PC2'],
                    mode='markers',
                    name=f'Grade {grade.upper()}' if grade in ['a', 'b', 'c', 'd', 'e'] else grade,
                    marker=dict(
                        color=color_discrete_map.get(grade.lower(), '#777777'),
                        size=6,
                        opacity=0.5
                    ),
                    text=subset.index,
                    hoverinfo='text'
                ))
        else:
            # For numerical color column
            fig.add_trace(go.Scatter(
                x=pca_df['PC1'],
                y=pca_df['PC2'],
                mode='markers',
                marker=dict(
                    color=pca_df[color_col],
                    colorscale='Viridis',
                    size=6,
                    opacity=0.5,
                    colorbar=dict(title=color_col)
                ),
                text=pca_df.index,
                hoverinfo='text',
                name='Products'
            ))
    else:
        # No color column
        fig.add_trace(go.Scatter(
            x=pca_df['PC1'],
            y=pca_df['PC2'],
            mode='markers',
            marker=dict(size=6, opacity=0.5),
            text=pca_df.index,
            hoverinfo='text',
            name='Products'
        ))
    
    # Get the feature loadings
    loadings = pca.components_.T[:, 0:2]
    
    # Scale the loadings for visualization
    loading_scale = np.abs(pca_df[['PC1', 'PC2']]).max().max() * 0.8 / np.abs(loadings).max().max()
    
    # Add feature vectors
    for i, feature in enumerate(feature_names):
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0] * loading_scale],
            y=[0, loadings[i, 1] * loading_scale],
            mode='lines+text',
            line=dict(color='red', width=1),
            text=['', feature],
            textposition='top center',
            textfont=dict(size=10, color='darkred'),
            name=feature,
            hoverinfo='name'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Biplot: PCA with Feature Vectors<br><sup>Total Variance Explained: {sum(explained_variance):.1f}%</sup>',
        xaxis_title=f'PC1 ({explained_variance[0]:.1f}%)',
        yaxis_title=f'PC2 ({explained_variance[1]:.1f}%)',
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='grey'),
        height=700,
        width=900,
        legend_title=color_col if color_col in pca_df.columns else None
    )
    
    return fig

def perform_kmeans_clustering(pca_df, n_clusters=5, color_col=None):
    """
    Perform K-means clustering on PCA results and visualize.
    
    Args:
        pca_df: DataFrame containing PCA results
        n_clusters: Number of clusters to form
        color_col: Column name to compare clustering results with
        
    Returns:
        tuple: (clustered_df, cluster_fig)
    """
    # Make a copy to avoid modifying the original
    df_cluster = pca_df.copy()
    
    # Select only PC columns for clustering
    pc_cols = [col for col in df_cluster.columns if col.startswith('PC')]
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[pc_cols])
    
    # Create figure
    cluster_fig = go.Figure()
    
    # Add a trace for each cluster
    cluster_colors = px.colors.qualitative.Bold
    
    for i in range(n_clusters):
        cluster_data = df_cluster[df_cluster['Cluster'] == i]
        
        cluster_fig.add_trace(go.Scatter(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            mode='markers',
            marker=dict(
                size=8,
                color=cluster_colors[i % len(cluster_colors)],
                opacity=0.7
            ),
            name=f'Cluster {i+1}',
            text=cluster_data.index,
            hoverinfo='text'
        ))
    
    # Add cluster centers
    centers = kmeans.cluster_centers_
    cluster_fig.add_trace(go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode='markers',
        marker=dict(
            symbol='x',
            size=12,
            color='black',
            line=dict(width=2)
        ),
        name='Cluster Centers'
    ))
    
    # Update layout
    cluster_fig.update_layout(
        title='K-means Clustering of PCA Results',
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=600,
        width=900
    )
    
    # If color_col exists, create a comparison figure
    comparison_fig = None
    if color_col and color_col in df_cluster.columns:
        # Create a cross-tabulation of clusters vs the color column
        cross_tab = pd.crosstab(df_cluster['Cluster'], df_cluster[color_col])
        
        # Create a heat map
        comparison_fig = px.imshow(
            cross_tab,
            labels=dict(x=color_col, y="Cluster", color="Count"),
            x=cross_tab.columns,
            y=[f"Cluster {i+1}" for i in range(n_clusters)],
            color_continuous_scale="Viridis",
            title=f"Comparing K-means Clusters with {color_col}"
        )
        
        # Update layout
        comparison_fig.update_layout(
            height=500,
            width=700
        )
    
    return df_cluster, cluster_fig, comparison_fig

def create_feature_importance_plot(feature_importance):
    """
    Create a visualization of feature importance in PCA.
    
    Args:
        feature_importance: DataFrame with PCA feature loadings
        
    Returns:
        plotly figure
    """
    # Create a figure with subplots for each PC
    n_components = feature_importance.shape[1]
    fig = make_subplots(
        rows=n_components, 
        cols=1,
        subplot_titles=[f"Features Contributing to PC{i+1}" for i in range(n_components)]
    )
    
    # Add bars for each principal component
    for i in range(n_components):
        pc_col = f'PC{i+1}'
        
        # Sort features by absolute importance
        sorted_importance = feature_importance[pc_col].abs().sort_values(ascending=False)
        top_features = sorted_importance.index[:15]  # Show top 15 features
        
        # Get actual values (not absolute) for these top features
        values = feature_importance.loc[top_features, pc_col]
        
        # Add bars
        fig.add_trace(
            go.Bar(
                y=top_features,
                x=values,
                orientation='h',
                marker_color=['red' if x < 0 else 'blue' for x in values],
                name=pc_col,
                showlegend=False
            ),
            row=i+1, 
            col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300*n_components,
        width=800,
        title="Feature Importance in Principal Components",
    )
    
    # Set consistent range for x-axis
    max_abs_value = feature_importance.abs().max().max()
    for i in range(n_components):
        fig.update_xaxes(range=[-max_abs_value*1.1, max_abs_value*1.1], row=i+1, col=1)
    
    return fig

def visualize_nutrient_pca(df, numeric_cols=None, grade_col='nutrition_grade_fr', n_clusters=5):
    """
    Create complete PCA and clustering analysis for nutritional data.
    
    Args:
        df: DataFrame with nutritional data
        numeric_cols: List of numeric columns to analyze (if None, selects automatically)
        grade_col: Column with nutrition grades
        n_clusters: Number of clusters for K-means
        
    Returns:
        dict: Dictionary containing all visualizations and results
    """
    # Step 1: Perform PCA
    pca_df, feature_importance, pca_fig, biplot_fig = perform_pca_analysis(
        df, numeric_cols=numeric_cols, color_col=grade_col
    )
    
    # Step 2: Create feature importance plot
    importance_fig = create_feature_importance_plot(feature_importance)
    
    # Step 3: Perform clustering
    clustered_df, cluster_fig, comparison_fig = perform_kmeans_clustering(
        pca_df, n_clusters=n_clusters, color_col=grade_col
    )
    
    # Return all results as a dictionary
    return {
        'pca_df': pca_df,
        'feature_importance': feature_importance,
        'pca_scatterplot': pca_fig,
        'biplot': biplot_fig,
        'feature_importance_plot': importance_fig,
        'clustered_df': clustered_df,
        'cluster_plot': cluster_fig,
        'cluster_comparison': comparison_fig
    }