import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# VALIDATION CALCULATION FUNCTIONS
def check_sodium_salt_relationship(df):
    """Calculate validation metrics for the sodium-salt relationship (salt = sodium * 2.5)"""
    validation_result = {}
    df_validated = df.copy()
    
    mask_both = (~df['sodium_100g'].isna()) & (~df['salt_100g'].isna())
    if mask_both.sum() > 0:
        filtered_df = df.loc[mask_both]
        expected_salt = filtered_df['sodium_100g'] * 2.5
        actual_salt = filtered_df['salt_100g']
        
        deviation = np.abs((actual_salt - expected_salt) / expected_salt * 100)
        inconsistent_salt = (deviation > 10).sum()
        
        validation_result = {
            'Relationship': 'Sodium-Salt',
            'Description': 'salt = sodium * 2.5',
            'Total Checked': mask_both.sum(),
            'Consistent': mask_both.sum() - inconsistent_salt,
            'Inconsistent': inconsistent_salt,
            'Consistency %': (1 - inconsistent_salt / mask_both.sum()) * 100 if mask_both.sum() > 0 else 0
        }
        
        # Fix inconsistent values
        fix_indices = filtered_df[deviation > 10].index
        if len(fix_indices) > 0:
            df_validated.loc[fix_indices, 'salt_100g'] = df_validated.loc[fix_indices, 'sodium_100g'] * 2.5
    
    return validation_result, df_validated

def check_energy_macronutrients_relationship(df):
    """Calculate validation metrics for energy vs macronutrients relationship"""
    validation_result = {}
    df_validated = df.copy()
    
    # Atwater factors: protein=4kcal/g, carbs=4kcal/g, fat=9kcal/g
    mask_energy = (~df['energy_100g'].isna()) & (~df['proteins_100g'].isna()) & (~df['carbohydrates_100g'].isna()) & (~df['fat_100g'].isna())
    if mask_energy.sum() > 0:
        filtered_energy_df = df.loc[mask_energy]
        
        expected_energy = (
            filtered_energy_df['proteins_100g'] * 4 + 
            filtered_energy_df['carbohydrates_100g'] * 4 + 
            filtered_energy_df['fat_100g'] * 9
        )
        actual_energy = filtered_energy_df['energy_100g']
        
        deviation = np.abs((actual_energy - expected_energy) / expected_energy * 100)
        inconsistent_energy = (deviation > 20).sum()
        
        validation_result = {
            'Relationship': 'Energy-Macronutrients',
            'Description': 'energy ≈ proteins*4 + carbs*4 + fat*9',
            'Total Checked': mask_energy.sum(),
            'Consistent': mask_energy.sum() - inconsistent_energy,
            'Inconsistent': inconsistent_energy,
            'Consistency %': (1 - inconsistent_energy / mask_energy.sum()) * 100 if mask_energy.sum() > 0 else 0
        }
    
    return validation_result, df_validated

def check_pnns_hierarchy(df):
    """Calculate validation metrics for PNNS groups hierarchical relationship"""
    validation_result = {}
    df_validated = df.copy()
    
    if 'pnns_groups_1' in df.columns and 'pnns_groups_2' in df.columns:
        mask_pnns = (~df['pnns_groups_1'].isna()) & (~df['pnns_groups_2'].isna())
        if mask_pnns.sum() > 0:
            filtered_pnns_df = df.loc[mask_pnns]
            
            # Build parent-child mapping
            pnns_mappings = filtered_pnns_df.groupby(['pnns_groups_1', 'pnns_groups_2']).size().reset_index(name='count')
            valid_mappings = {}
            for _, row in pnns_mappings.iterrows():
                parent = row['pnns_groups_1']
                child = row['pnns_groups_2']
                if parent not in valid_mappings:
                    valid_mappings[parent] = set()
                valid_mappings[parent].add(child)
            
            # Check consistency
            inconsistent_count = 0
            inconsistent_indices = []
            
            for idx, row in filtered_pnns_df.iterrows():
                parent = row['pnns_groups_1']
                child = row['pnns_groups_2']
                
                if parent in valid_mappings and child not in valid_mappings[parent]:
                    inconsistent_count += 1
                    inconsistent_indices.append(idx)
            
            validation_result = {
                'Relationship': 'PNNS Hierarchy',
                'Description': 'pnns_groups_2 belongs to parent pnns_groups_1',
                'Total Checked': mask_pnns.sum(),
                'Consistent': mask_pnns.sum() - inconsistent_count,
                'Inconsistent': inconsistent_count,
                'Consistency %': (1 - inconsistent_count / mask_pnns.sum()) * 100 if mask_pnns.sum() > 0 else 0
            }
    
    return validation_result, df_validated

def calculate_nutrient_correlations(df):
    """Calculate correlation matrix for nutrient columns"""
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter to nutrient-related columns
    nutrient_cols = [col for col in numerical_cols if any(x in col for x in ['energy', 'fat', 'protein', 'sugar', 'salt', 'sodium', 'carbo', 'fiber'])]
    nutrient_cols = [col for col in nutrient_cols if col in df.columns]
    
    if len(nutrient_cols) < 2:
        return None, nutrient_cols
    
    # Calculate correlation matrix
    corr_df = df[nutrient_cols].corr()
    return corr_df, nutrient_cols

def identify_nutrient_clusters(corr_df):
    """Identify clusters of related nutrients based on correlation strength"""
    if corr_df is None:
        return pd.DataFrame(columns=['Feature', 'Cluster', 'Description'])
    
    try:
        # Use seaborn for hierarchical clustering
        plt.figure(figsize=(10, 8))
        sns_heat = sns.clustermap(
            corr_df, 
            cmap='RdBu_r',
            center=0,
            figsize=(10, 8)
        )
        
        # Get the cluster order
        row_order = sns_heat.dendrogram_row.reordered_ind
        clustered_features = [corr_df.index[i] for i in row_order]
        
        # Group related features
        feature_clusters = []
        current_cluster = []
        
        for i, feature in enumerate(clustered_features):
            if i == 0:
                current_cluster = [feature]
            else:
                prev_feature = clustered_features[i-1]
                if abs(corr_df.loc[feature, prev_feature]) > 0.6:  # Strong correlation threshold
                    current_cluster.append(feature)
                else:
                    feature_clusters.append(current_cluster)
                    current_cluster = [feature]
        
        # Add the last cluster if not empty
        if current_cluster:
            feature_clusters.append(current_cluster)
        
        # Create cluster descriptions
        cluster_data = []
        for i, cluster in enumerate(feature_clusters):
            for feature in cluster:
                # Determine cluster description
                if any('fat' in f.lower() for f in cluster):
                    description = "Fat-related measures"
                elif any('sugar' in f.lower() for f in cluster) or any('carbo' in f.lower() for f in cluster):
                    description = "Carbohydrate and sugar measures"
                elif any('energy' in f.lower() for f in cluster):
                    description = "Energy-related measures"
                elif any('salt' in f.lower() or 'sodium' in f.lower() for f in cluster):
                    description = "Salt and sodium measures"
                elif any('protein' in f.lower() for f in cluster):
                    description = "Protein-related measures"
                else:
                    description = f"Feature cluster {i+1}"
                    
                cluster_data.append({
                    'Feature': feature,
                    'Cluster': f"Cluster {i+1}",
                    'Description': description
                })
        
        cluster_df = pd.DataFrame(cluster_data)
        return cluster_df
    
    except Exception as e:
        print(f"Clustering failed: {str(e)}")
        return pd.DataFrame(columns=['Feature', 'Cluster', 'Description'])

def analyze_category_nutrient_relationships(df, nutrient_cols):
    """Analyze relationships between categorical variables and nutrients"""
    cat_cols = ['pnns_groups_1', 'nutrition_grade_fr']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # Return empty if no categorical columns or nutrient columns
    if len(cat_cols) == 0 or len(nutrient_cols) == 0:
        return pd.DataFrame()
    
    cat_var = cat_cols[0]  # Use first available categorical column
    
    # Get top categories
    top_categories = df[cat_var].value_counts().nlargest(6).index.tolist()
    
    # Select key nutrients to analyze
    key_nutrients = ['energy_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g', 'sugars_100g']
    key_nutrients = [n for n in key_nutrients if n in df.columns]
    
    # Calculate statistics by category
    cat_data = []
    for category in top_categories:
        subset = df[df[cat_var] == category]
        for nutrient in key_nutrients:
            if nutrient in df.columns:
                cat_data.append({
                    'Category': category,
                    'Nutrient': nutrient,
                    'Mean': subset[nutrient].mean(),
                    'Median': subset[nutrient].median(),
                    'Count': subset[nutrient].count()
                })
    
    return pd.DataFrame(cat_data) if cat_data else pd.DataFrame()

# VISUALIZATION FUNCTIONS
def plot_correlation_matrix(corr_df):
    """Visualize correlation matrix for nutrients"""
    if corr_df is None:
        fig = go.Figure()
        fig.add_annotation(text="Not enough nutrient columns for correlation analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        title="Nutrient Correlation Matrix"
    )
    
    fig.update_layout(height=700, width=700, title_x=0.5)
    return fig

def plot_category_relationships(cat_nutrient_df):
    """Visualize relationships between categories and nutrients"""
    if cat_nutrient_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough categorical data for analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    cat_var = cat_nutrient_df['Category'].iloc[0]
    fig = px.bar(
        cat_nutrient_df,
        x='Category',
        y='Mean',
        color='Nutrient',
        barmode='group',
        title=f'Mean Nutrient Content by Category',
        labels={'Mean': 'Mean Value', 'Category': cat_var},
        hover_data=['Median', 'Count']
    )
    
    fig.update_layout(height=500, title_x=0.5)
    return fig

def create_validation_summary_plot(validation_summary):
    """Create validation results visualization"""
    fig = go.Figure()
    
    if not validation_summary.empty:
        fig.add_trace(
            go.Bar(
                x=validation_summary['Relationship'],
                y=validation_summary['Consistency %'],
                text=[f"{x:.1f}%" for x in validation_summary['Consistency %']],
                textposition='auto',
                name='Consistency %',
                marker_color=['#00cc96' if x > 95 else '#ffa15a' if x > 80 else '#ef553b' for x in validation_summary['Consistency %']]
            )
        )
        
        fig.update_layout(
            title_text='Data Relationship Validation Results',
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            title_x=0.5
        )
    else:
        fig.add_annotation(text="No validation relationships could be tested",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return fig

def create_validation_rules_table():
    """Create validation rules table"""
    validation_rules = {
        'Relationship': ['Sodium-Salt', 'Energy-Macronutrients', 'PNNS Hierarchy'],
        'Rule': [
            'salt = sodium * 2.5',
            'energy ≈ proteins*4 + carbs*4 + fat*9',
            'pnns_groups_2 must belong to its parent pnns_groups_1'
        ],
        'Imputation Strategy': [
            'Calculate salt from sodium or vice versa based on reliability',
            'Recalculate energy values from macronutrients when inconsistent',
            'Impute child category based on parent and similar products'
        ]
    }
    
    rules_df = pd.DataFrame(validation_rules)
    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=list(rules_df.columns),
                fill_color='royalblue',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[rules_df[col] for col in rules_df.columns],
                fill_color='lavender',
                align='left'
            )
        )
    ])
    
    fig.update_layout(title_text="Data Validation Rules", title_x=0.5)
    return fig

def create_cluster_table(cluster_df):
    """Create nutrient clusters table"""
    if cluster_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No nutrient clusters could be identified",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=list(cluster_df.columns),
                fill_color='royalblue',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[cluster_df[col] for col in cluster_df.columns],
                fill_color='lavender',
                align='left'
            )
        )
    ])
    
    fig.update_layout(title_text="Nutrient Feature Clusters", title_x=0.5)
    return fig

# MAIN FUNCTIONS
def validate_nutritional_relationships(df):
    """
    Validate and cross-check related variables in the nutritional dataset.
    
    Args:
        df: DataFrame containing the nutritional data
        
    Returns:
        tuple: (validation_summary_df, df_validated)
    """
    # Create a copy to avoid modifying the original
    df_validated = df.copy()
    
    # Initialize a list to store validation results
    validation_results = []
    
    # Check sodium-salt relationship
    sodium_salt_result, df_validated = check_sodium_salt_relationship(df_validated)
    if sodium_salt_result:
        validation_results.append(sodium_salt_result)
    
    # Check energy-macronutrients relationship
    energy_macro_result, df_validated = check_energy_macronutrients_relationship(df_validated)
    if energy_macro_result:
        validation_results.append(energy_macro_result)
    
    # Check PNNS hierarchy
    pnns_result, df_validated = check_pnns_hierarchy(df_validated)
    if pnns_result:
        validation_results.append(pnns_result)
    
    # Create summary dataframe
    validation_summary = pd.DataFrame(validation_results)
    
    return validation_summary, df_validated

def analyze_variable_dependencies(df):
    """
    Analyze relationships and dependencies between variables.
    
    Args:
        df: DataFrame containing the nutritional data
        
    Returns:
        tuple: (correlation_fig, categorical_relationship_fig, cluster_df)
    """
    # Calculate correlations between nutrients
    corr_df, nutrient_cols = calculate_nutrient_correlations(df)
    
    # Identify nutrient clusters
    cluster_df = identify_nutrient_clusters(corr_df)
    
    # Analyze category-nutrient relationships
    cat_nutrient_df = analyze_category_nutrient_relationships(df, nutrient_cols)
    
    # Create visualizations
    corr_fig = plot_correlation_matrix(corr_df)
    cat_fig = plot_category_relationships(cat_nutrient_df)
    
    return corr_fig, cat_fig, cluster_df

def create_validation_dashboard(df):
    """
    Create a complete validation dashboard for the dataset.
    
    Args:
        df: DataFrame containing the nutritional data
        
    Returns:
        tuple: (validation_summary_df, df_validated, dashboard_fig)
    """
    # Run validation
    validation_summary, df_validated = validate_nutritional_relationships(df)
    
    # Create validation results visualization
    val_fig = create_validation_summary_plot(validation_summary)
    
    # Run dependency analysis
    corr_fig, cat_fig, cluster_df = analyze_variable_dependencies(df)
    
    # Create rules table
    rules_fig = create_validation_rules_table()
    
    # Create clusters table
    clusters_fig = create_cluster_table(cluster_df)
    
    return validation_summary, df_validated, val_fig, corr_fig, cat_fig, rules_fig, clusters_fig