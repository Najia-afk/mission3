import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_nutrient_correlations(df):
    """Calculate correlation matrix for nutrient columns"""
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter to nutrient-related columns
    #nutrient_cols = [col for col in numerical_cols if any(x in col for x in ['energy', 'fat', 'protein', 'sugar', 'salt', 'sodium', 'carbo', 'fiber'])]
    #nutrient_cols = [col for col in nutrient_cols if col in df.columns]

    nutrient_cols = numerical_cols
    
    if len(nutrient_cols) < 2:
        return None, nutrient_cols
    
    # Calculate correlation matrix
    corr_df = df[nutrient_cols].corr()
    return corr_df, nutrient_cols

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

def compare_imputation_results(df_sample, df_imputed):
    """
    Create comparison visualizations between original and imputed data
    
    Args:
        df_sample: Original DataFrame before imputation
        df_imputed: DataFrame after imputation
        
    Returns:
        tuple: (correlation_comparison_fig, category_comparison_fig)
    """
    # Compare correlation matrices
    corr_sample, nutrient_cols_sample = calculate_nutrient_correlations(df_sample)
    corr_imputed, nutrient_cols_imputed = calculate_nutrient_correlations(df_imputed)
    
    # Compare categorical relationships
    cat_sample = analyze_category_nutrient_relationships(df_sample, nutrient_cols_sample)
    cat_imputed = analyze_category_nutrient_relationships(df_imputed, nutrient_cols_imputed)
    
    # Create correlation comparison figure (side by side)
    correlation_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Original Data Correlation Matrix", "Imputed Data Correlation Matrix"],
        horizontal_spacing=0.1
    )
    
    if corr_sample is not None:
        # Add heatmap with text annotations
        correlation_fig.add_trace(
            go.Heatmap(
                z=corr_sample.values,
                x=corr_sample.columns,
                y=corr_sample.index,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                colorbar=dict(title="Correlation", x=-0.15),
                text=corr_sample.values.round(2),  # Add text values
                texttemplate="%{text:.2f}",        # Format to 2 decimal places
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
            ),
            row=1, col=1
        )
    else:
        correlation_fig.add_annotation(
            text="Not enough data for correlation analysis",
            xref="x1 domain", yref="y1 domain", 
            x=0.5, y=0.5, showarrow=False,
            row=1, col=1
        )
    
    if corr_imputed is not None:
        # Add heatmap with text annotations
        correlation_fig.add_trace(
            go.Heatmap(
                z=corr_imputed.values,
                x=corr_imputed.columns,
                y=corr_imputed.index,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                colorbar=dict(title="Correlation", x=1.07),
                text=corr_imputed.values.round(2),  # Add text values
                texttemplate="%{text:.2f}",         # Format to 2 decimal places
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
            ),
            row=1, col=2
        )
    else:
        correlation_fig.add_annotation(
            text="Not enough data for correlation analysis",
            xref="x2 domain", yref="y2 domain", 
            x=0.5, y=0.5, showarrow=False,
            row=1, col=2
        )
    
    # Adjust font size and layout for correlation matrix
    correlation_fig.update_layout(
        title="Correlation Matrix Comparison: Before vs After Imputation",
        height=700,
        title_x=0.5,
        font=dict(size=9)  # Smaller font size to fit text in cells
    )
    
    # Create categorical comparison figure (top and bottom)
    category_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Original Data Categorical Relationships", "Imputed Data Categorical Relationships"],
        vertical_spacing=0.15,
        specs=[[{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # Add original data categorical bar chart
    if not cat_sample.empty:
        for nutrient in cat_sample['Nutrient'].unique():
            df_nutrient = cat_sample[cat_sample['Nutrient'] == nutrient]
            
            # Format mean values for display
            mean_values = df_nutrient['Mean'].round(1)
            counts = df_nutrient['Count']
            
            category_fig.add_trace(
                go.Bar(
                    name=nutrient + " (Original)",
                    x=df_nutrient['Category'],
                    y=df_nutrient['Mean'],
                    text=mean_values,  # Show mean values as text
                    textposition='outside',
                    customdata=counts,  # Store count for hover
                    hovertemplate="%{x}<br>Mean: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
                    legendgroup=nutrient,
                ),
                row=1, col=1
            )
    else:
        category_fig.add_annotation(
            text="Not enough categorical data",
            xref="x1 domain", yref="y1 domain", 
            x=0.5, y=0.5, showarrow=False,
            row=1, col=1
        )
    
    # Add imputed data categorical bar chart
    if not cat_imputed.empty:
        for nutrient in cat_imputed['Nutrient'].unique():
            df_nutrient = cat_imputed[cat_imputed['Nutrient'] == nutrient]
            
            # Format mean values for display
            mean_values = df_nutrient['Mean'].round(1)
            counts = df_nutrient['Count']
            
            category_fig.add_trace(
                go.Bar(
                    name=nutrient + " (Imputed)",
                    x=df_nutrient['Category'],
                    y=df_nutrient['Mean'],
                    text=mean_values,  # Show mean values as text
                    textposition='outside',
                    customdata=counts,  # Store count for hover
                    hovertemplate="%{x}<br>Mean: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
                    legendgroup=nutrient,
                ),
                row=2, col=1
            )
    else:
        category_fig.add_annotation(
            text="Not enough categorical data",
            xref="x2 domain", yref="y2 domain", 
            x=0.5, y=0.5, showarrow=False,
            row=2, col=1
        )
    
    category_fig.update_layout(
        title="Category Nutrient Relationships: Before vs After Imputation",
        height=900,
        title_x=0.5,
        barmode='group'
    )
    
    return correlation_fig, category_fig