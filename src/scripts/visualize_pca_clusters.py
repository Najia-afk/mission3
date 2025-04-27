import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.impute import SimpleImputer

def visualize_nutrient_pca(df, numeric_cols=None, grade_col='nutrition_grade_fr', n_clusters=5, sample_size=None):
    """
    Create optimized PCA and clustering analysis for nutritional data.
    
    Args:
        df: DataFrame with nutritional data
        numeric_cols: List of numeric columns to analyze
        grade_col: Column with nutrition grades
        n_clusters: Number of clusters for K-means
        sample_size: Optional limit to sample size for large datasets
    """
    # Step 1: Perform PCA with sampling for large datasets
    pca_df, feature_importance, pca_fig, biplot_fig = perform_pca_analysis(
        df, 
        numeric_cols=numeric_cols, 
        color_col=grade_col,
        sample_size=sample_size
    )
    
    # Step 2: Create simplified feature importance plot
    importance_fig = create_feature_importance_plot(feature_importance)
    
    # Step 3: Perform optimized clustering
    clustered_df, cluster_fig, comparison_fig = perform_kmeans_clustering(
        pca_df, 
        n_clusters=n_clusters, 
        color_col=grade_col,
        use_minibatch=(len(df) > 5000)  # Use minibatch for large datasets
    )
    
    # Return all results
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

def perform_pca_analysis(df, numeric_cols=None, color_col='nutrition_grade_fr', n_components=3, random_state=42, sample_size=None):
    """
    Perform PCA analysis on nutritional data and create interactive visualizations.
    
    Args:
        df: DataFrame containing nutritional data
        numeric_cols: List of numeric columns to include in PCA
        color_col: Column to use for coloring points
        n_components: Number of principal components to extract
        random_state: Random state for reproducibility
        sample_size: Optional subsample size for large datasets
    """
    # Sample data if needed (for very large datasets)
    if sample_size and len(df) > sample_size:
        df_pca = df.sample(sample_size, random_state=random_state).copy()
    else:
        df_pca = df.copy()
    # Select numeric columns if not specified
    if numeric_cols is None:
        numeric_cols = df_pca.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if 'score' not in col.lower() or col == 'nutrition-score-fr_100g']
    
    # Handle missing values and scale
    imputer = SimpleImputer(strategy='median')
    df_numeric = pd.DataFrame(
        imputer.fit_transform(df_pca[numeric_cols]), 
        columns=numeric_cols,
        index=df_pca.index
    )
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df_pca.index
    )
    
    # Add color column
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
    
    # Create PCA scatter plot with minimal hover info
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
                        size=5,  # Smaller markers
                        opacity=0.7
                    ),
                    hoverinfo='none'  # Remove hover info
                ))
        else:
            # For numerical color column - simplified with no hover data
            pca_fig = px.scatter(
                pca_df, x='PC1', y='PC2', 
                color=color_col,
                render_mode='webgl',  # Use WebGL for better performance
                hover_data=None  # No additional hover data
            )
            # Further reduce hover info
            pca_fig.update_traces(hovertemplate='<extra></extra>')
    else:
        # Simple scatter plot with minimal hover
        pca_fig = px.scatter(
            pca_df, x='PC1', y='PC2',
            render_mode='webgl',
            hover_data=None
        )
        pca_fig.update_traces(hovertemplate='<extra></extra>')
    
    # Update layout
    pca_fig.update_layout(
        title=f'PCA of Nutritional Data<br><sup>Total Variance Explained: {sum(explained_variance):.1f}%</sup>',
        xaxis_title=f'PC1 ({explained_variance[0]:.1f}%)',
        yaxis_title=f'PC2 ({explained_variance[1]:.1f}%)',
        legend_title=color_col if color_col in df_pca.columns else None,
        height=600,
        width=900
    )
    
    # Create a simplified biplot
    biplot_fig = create_biplot(pca, scaled_data, numeric_cols, pca_df, color_col, explained_variance)
    
    return pca_df, feature_importance, pca_fig, biplot_fig

def create_biplot(pca, scaled_data, feature_names, pca_df, color_col, explained_variance):
    """Create a biplot with minimal hover information."""
    
    fig = go.Figure()
    
    # Add scatter points as a single trace if possible
    if color_col in pca_df.columns:
        # If categorical
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
                        size=4,  # Even smaller points for biplot
                        opacity=0.4
                    ),
                    hoverinfo='none'  # No hover info
                ))
        else:
            # For numerical color column - simplified
            fig.add_trace(go.Scatter(
                x=pca_df['PC1'],
                y=pca_df['PC2'],
                mode='markers',
                marker=dict(
                    color=pca_df[color_col],
                    colorscale='Viridis',
                    size=4,
                    opacity=0.4,
                    colorbar=dict(title=color_col)
                ),
                hoverinfo='none',
                name='Products'
            ))
    else:
        # No color column - simplified
        fig.add_trace(go.Scatter(
            x=pca_df['PC1'],
            y=pca_df['PC2'],
            mode='markers',
            marker=dict(size=4, opacity=0.4),
            hoverinfo='none',
            name='Products'
        ))
    
    # Get feature loadings
    loadings = pca.components_.T[:, 0:2]
    
    # Scale loadings for visualization
    loading_scale = np.abs(pca_df[['PC1', 'PC2']]).max().max() * 0.8 / np.abs(loadings).max().max()
    
    # Add feature vectors - only add text for the key features
    top_features_idx = np.argsort(-np.abs(loadings).sum(axis=1))[:10]  # Top 10 most important features
    
    for i, feature in enumerate(feature_names):
        is_important = i in top_features_idx
        
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0] * loading_scale],
            y=[0, loadings[i, 1] * loading_scale],
            mode='lines+text' if is_important else 'lines',
            line=dict(color='red', width=1),
            text=['' if not is_important else '', feature if is_important else ''],
            textposition='top center',
            textfont=dict(size=10, color='darkred'),
            name=feature,
            hoverinfo='none'  # No hover
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

def perform_kmeans_clustering(pca_df, n_clusters=5, color_col=None, use_minibatch=True):
    """Perform clustering with optimized performance."""
    # Copy only necessary columns
    df_cluster = pca_df.copy()
    
    # Select PC columns for clustering
    pc_cols = [col for col in df_cluster.columns if col.startswith('PC')]
    
    # Use minibatch for large datasets
    if use_minibatch and len(df_cluster) > 10000:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[pc_cols])
    
    # Create figure with minimal hover info
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
                size=6,
                color=cluster_colors[i % len(cluster_colors)],
                opacity=0.7
            ),
            name=f'Cluster {i+1}',
            hoverinfo='none'  # No hover info
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
        name='Cluster Centers',
        hoverinfo='none'
    ))
    
    # Update layout
    cluster_fig.update_layout(
        title='K-means Clustering of PCA Results',
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=600,
        width=900
    )
    
    # Simplified comparison figure
    comparison_fig = None
    if color_col and color_col in df_cluster.columns:
        cross_tab = pd.crosstab(df_cluster['Cluster'], df_cluster[color_col])
        
        comparison_fig = px.imshow(
            cross_tab,
            labels=dict(x=color_col, y="Cluster", color="Count"),
            x=cross_tab.columns,
            y=[f"Cluster {i+1}" for i in range(n_clusters)],
            color_continuous_scale="Viridis",
            title=f"Comparing K-means Clusters with {color_col}"
        )
        
        comparison_fig.update_layout(
            height=500,
            width=700
        )
    
    return df_cluster, cluster_fig, comparison_fig

def create_feature_importance_plot(feature_importance):
    """Create feature importance plot with top features only."""
    n_components = feature_importance.shape[1]
    
    # Show fewer components if there are many
    n_display = min(n_components, 3)  # Limit to 3 components
    
    # Create subplot figure - more compact
    fig = make_subplots(
        rows=n_display, 
        cols=1,
        subplot_titles=[f"Features Contributing to PC{i+1}" for i in range(n_display)]
    )
    
    for i in range(n_display):
        pc_col = f'PC{i+1}'
        
        # Show only top 10 features instead of 15
        sorted_importance = feature_importance[pc_col].abs().sort_values(ascending=False)
        top_features = sorted_importance.index[:10]
        
        values = feature_importance.loc[top_features, pc_col]
        
        fig.add_trace(
            go.Bar(
                y=top_features,
                x=values,
                orientation='h',
                marker_color=['red' if x < 0 else 'blue' for x in values],
                name=pc_col,
                showlegend=False,
                hoverinfo='none'  # No hover info
            ),
            row=i+1, 
            col=1
        )
    
    # Update layout - smaller plot
    fig.update_layout(
        height=200*n_display,  # Reduced height
        width=800,
        title="Feature Importance in Principal Components",
    )
    
    # Consistent range for x-axis
    max_abs_value = feature_importance.abs().max().max()
    for i in range(n_display):
        fig.update_xaxes(range=[-max_abs_value*1.1, max_abs_value*1.1], row=i+1, col=1)
    
    return fig

def visualize_nutrient_pca(df, numeric_cols=None, grade_col='nutrition_grade_fr', n_clusters=5, sample_size=None):
    """
    Create optimized PCA and clustering analysis for nutritional data.
    
    Args:
        df: DataFrame with nutritional data
        numeric_cols: List of numeric columns to analyze
        grade_col: Column with nutrition grades
        n_clusters: Number of clusters for K-means
        sample_size: Optional limit to sample size for large datasets
    """
    # Step 1: Perform PCA with sampling for large datasets
    pca_df, feature_importance, pca_fig, biplot_fig = perform_pca_analysis(
        df, 
        numeric_cols=numeric_cols, 
        color_col=grade_col,
        sample_size=sample_size
    )
    
    # Step 2: Create simplified feature importance plot
    importance_fig = create_feature_importance_plot(feature_importance)
    
    # Step 3: Perform optimized clustering
    clustered_df, cluster_fig, comparison_fig = perform_kmeans_clustering(
        pca_df, 
        n_clusters=n_clusters, 
        color_col=grade_col,
        use_minibatch=(len(df) > 10000)  # Use minibatch for large datasets
    )
    
    # Return all results
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