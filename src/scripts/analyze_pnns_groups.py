import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import plotly.express as px

def normalize_category_name(name):
    """Normalize category names by removing hyphens and standardizing format"""
    if pd.isna(name):
        return name
    normalized = name.lower().replace('-', ' ')
    normalized = ' '.join(normalized.split())
    return normalized

def combine_similar_categories(group_counts, score_cutoff=85):
    """Combine similar categories using fuzzy matching"""
    combined_counts = pd.Series(dtype='float64')
    processed_categories = set()
    
    for category, count in group_counts.items():
        if category in processed_categories:
            continue
        
        similar_categories = process.extractBests(
            category, 
            group_counts.index, 
            scorer=fuzz.ratio, 
            score_cutoff=score_cutoff
        )
        
        if len(similar_categories) > 1:
            total_count = sum(group_counts[cat[0]] for cat in similar_categories)
            standard_name = max(similar_categories, key=lambda x: group_counts[x[0]])[0]
            combined_counts[standard_name] = total_count
            processed_categories.update(cat[0] for cat in similar_categories)
        elif category not in processed_categories:
            combined_counts[category] = count
            processed_categories.add(category)
    
    return combined_counts.sort_values(ascending=False)

def create_category_mapping(group_counts, min_category_size, parent_groups=None):
    """Create mapping for categories based on size and similarity"""
    major_groups = group_counts[group_counts >= min_category_size].index.tolist()
    rare_groups = group_counts[group_counts < min_category_size].index.tolist()
    
    category_mapping = {}
    for group in group_counts.index:
        if group in rare_groups and group != 'unknown':
            if major_groups:
                match = process.extractOne(group, major_groups, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= 70:
                    category_mapping[group] = match[0]
                else:
                    if parent_groups is not None and group in parent_groups:
                        parent = parent_groups[group]
                        category_mapping[group] = f"{parent}-other"
                    else:
                        category_mapping[group] = 'Other'
            else:
                category_mapping[group] = 'Other'
        else:
            category_mapping[group] = group
            
    return category_mapping, major_groups, rare_groups

def plot_category_distribution(group_counts, ax, n_top, level):
    """Plot top categories distribution"""
    top_groups = group_counts.nlargest(n_top)
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Category': top_groups.index,
        'Count': top_groups.values
    })
    
    # Use hue parameter instead of palette
    sns.barplot(
        data=plot_df,
        x='Count', 
        y='Category',
        hue='Category',  # Add hue parameter
        legend=False,    # Disable legend
        ax=ax
    )
    
    ax.set_title(f'Top {n_top} Food {"Categories" if level == 1 else "Subcategories"} (pnns_groups_{level})')
    ax.set_xlabel('Count')

def analyze_and_simplify_food_categories(df, min_category_size=100):
    """Main function to analyze and simplify food categories"""
    df = df.copy()
    category_mappings = {}
    
    # Normalize category names
    for col in ['pnns_groups_1', 'pnns_groups_2']:
        if col in df.columns:
            df[col] = df[col].fillna(np.nan)
            df[col] = df[col].apply(normalize_category_name)
    
    # Create visualization figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    # Process level 1 categories
    if 'pnns_groups_1' in df.columns:
        group1_counts = df[~df['pnns_groups_1'].isin(['unknown', np.nan])]['pnns_groups_1'].value_counts()
        group1_counts = combine_similar_categories(group1_counts)
        level1_mapping, major_groups1, rare_groups1 = create_category_mapping(group1_counts, min_category_size)
        
        plot_category_distribution(group1_counts, axes[0], 10, 1)
        df['pnns_groups_1_simplified'] = df['pnns_groups_1'].map(level1_mapping)
        category_mappings['pnns_groups_1'] = level1_mapping
        
        # Display results for level 1
        print(f"PNNS Groups Level 1 Simplification:")
        print(f"- Original categories: {len(group1_counts)}")
        print(f"- Simplified categories: {len(df['pnns_groups_1_simplified'].unique())}")
        print(f"- Categories merged: {len(rare_groups1)}\n")
    
    # Process level 2 categories
    if 'pnns_groups_2' in df.columns:
        # Filter out unknown values
        group2_counts = df[df['pnns_groups_2'] != 'unknown']['pnns_groups_2'].value_counts()
        
        # Modified parent group calculation with safety check
        parent_groups = {}
        for group in group2_counts.index:
            mask = df['pnns_groups_2'] == group
            if mask.any():  # Check if there are any matches
                parent_values = df.loc[mask, 'pnns_groups_1'].value_counts()
                if not parent_values.empty:
                    parent_groups[group] = parent_values.index[0]
                else:
                    parent_groups[group] = 'Other'
            else:
                parent_groups[group] = 'Other'
        
        level2_mapping, major_groups2, rare_groups2 = create_category_mapping(
            group2_counts, 
            min_category_size, 
            parent_groups
        )
        
        plot_category_distribution(group2_counts, axes[1], 15, 2)
        df['pnns_groups_2_simplified'] = df['pnns_groups_2'].map(level2_mapping)
        category_mappings['pnns_groups_2'] = level2_mapping
        
        # Display results for level 2
        print(f"PNNS Groups Level 2 Simplification:")
        print(f"- Original categories: {len(group2_counts)}")
        print(f"- Simplified categories: {len(df['pnns_groups_2_simplified'].unique())}")
        print(f"- Categories merged: {len(rare_groups2)}\n")
    
    plt.tight_layout()
    plt.show()
    
    # Create hierarchical visualization
    if 'pnns_groups_1_simplified' in df.columns and 'pnns_groups_2_simplified' in df.columns:
        create_hierarchical_visualization(df)
    
    # Replace 'unknown' with np.nan in simplified columns
    if 'pnns_groups_1_simplified' in df.columns:
        df['pnns_groups_1_simplified'] = df['pnns_groups_1_simplified'].replace('unknown', np.nan)
    
    if 'pnns_groups_2_simplified' in df.columns:
        df['pnns_groups_2_simplified'] = df['pnns_groups_2_simplified'].replace('unknown', np.nan)
    
    # Drop original columns to avoid confusion
    if 'pnns_groups_1_simplified' in df.columns and 'pnns_groups_1' in df.columns:
        df.drop(columns=['pnns_groups_1'], inplace=True)
        
    if 'pnns_groups_2_simplified' in df.columns and 'pnns_groups_2' in df.columns:
        df.drop(columns=['pnns_groups_2'], inplace=True)
        
    # Rename simplified columns to standard names
    df.rename(columns={
        'pnns_groups_1_simplified': 'pnns_groups_1',
        'pnns_groups_2_simplified': 'pnns_groups_2'
    }, inplace=True)
    
    print(f"Replaced 'unknown' with np.nan and dropped original columns")
    print(f"Missing values in pnns_groups_1: {df['pnns_groups_1'].isna().sum()}")
    print(f"Missing values in pnns_groups_2: {df['pnns_groups_2'].isna().sum()}")
    
    return df, category_mappings

def create_hierarchical_visualization(df):
    """Create and display hierarchical relationship visualization"""
    df_filtered = df[
        (df['pnns_groups_1_simplified'] != 'unknown') & 
        (df['pnns_groups_2_simplified'] != 'unknown')
    ]
    
    cross_tab = pd.crosstab(
        df_filtered['pnns_groups_1_simplified'], 
        df_filtered['pnns_groups_2_simplified']
    )
    
    cross_tab_long = cross_tab.reset_index().melt(
        id_vars='pnns_groups_1_simplified',
        var_name='pnns_groups_2_simplified',
        value_name='count'
    )
    
    cross_tab_long = cross_tab_long[cross_tab_long['count'] > 0]
    
    fig = px.scatter(
        cross_tab_long,
        x='pnns_groups_1_simplified',
        y='pnns_groups_2_simplified',
        size='count',
        color='count',
        color_continuous_scale='Viridis',
        title='Hierarchical Relationship Between Simplified Food Categories',
        labels={
            'pnns_groups_1_simplified': 'Primary Food Category (Simplified)',
            'pnns_groups_2_simplified': 'Food Subcategory (Simplified)'
        }
    )
    
    fig.update_layout(
        height=1200,
        xaxis={'categoryorder': 'total descending'}
    )
    
    fig.show()