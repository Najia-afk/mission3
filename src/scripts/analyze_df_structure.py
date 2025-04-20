import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_metadata_df(df, name):
    """Create a metadata dataframe with statistics about each column."""
    total_rows = len(df)
    
    # Create metadata for each column in a single pass
    metadata = []
    for col in df.columns:
        non_null_count = df[col].count()
        unique_count = df[col].nunique()
        
        # Get most common value info if data exists
        most_common_value = "N/A"
        most_common_count = 0
        if non_null_count > 0 and not df[col].value_counts().empty:
            most_common_value = str(df[col].value_counts().index[0])
            most_common_count = df[col].value_counts().iloc[0]
            
        metadata.append({
            'DataFrame': name,
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': non_null_count,
            'Null Count': total_rows - non_null_count,
            'Fill Rate (%)': round(100 * non_null_count / total_rows if total_rows > 0 else 0, 2),
            'Unique Count': unique_count,
            'Unique Rate (%)': round(100 * unique_count / non_null_count if non_null_count > 0 else 0, 2),
            'Most Common Value': most_common_value,
            'Most Common Count': most_common_count
        })
    
    return pd.DataFrame(metadata)

def create_metadata_dfs(dfs_dict):
    """Create metadata for all dataframes in the dictionary."""
    return {name: create_metadata_df(df, name) for name, df in dfs_dict.items()}

def display_metadata_dfs(metadata_dfs_dict, fill_threshold=25):
    """Display metadata with visualizations and column categorization."""
    for name, meta_df in metadata_dfs_dict.items():
        print(f"\n=== Metadata Summary: {name} ===")
        display(meta_df.sort_values(by='Fill Rate (%)', ascending=False).head(20))
        
        # Visualize fill rate distribution
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Fill rate histogram
        sns.histplot(meta_df['Fill Rate (%)'], bins=20, kde=True, ax=axes[0])
        axes[0].set_title(f'Column Fill Rate Distribution')
        axes[0].set_xlabel('Fill Rate (%)')
        axes[0].axvline(x=50, color='red', linestyle='--', label='50% Threshold')
        axes[0].axvline(x=fill_threshold, color='green', linestyle='--', 
                       label=f'{fill_threshold}% Threshold')
        axes[0].legend()
        
        # Fill rate vs Unique rate scatter
        axes[1].scatter(meta_df['Fill Rate (%)'], meta_df['Unique Rate (%)'], alpha=0.6)
        axes[1].set_title('Fill Rate vs Unique Value Rate')
        axes[1].set_xlabel('Fill Rate (%)')
        axes[1].set_ylabel('Unique Rate (%)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Categorize columns
        high_fill = meta_df[meta_df['Fill Rate (%)'] >= fill_threshold]
        low_fill = meta_df[meta_df['Fill Rate (%)'] < fill_threshold]
        
        # Column categories
        id_like = high_fill[(high_fill['Unique Rate (%)'] > 90) & (high_fill['Unique Count'] > 1000)]
        categorical = high_fill[(high_fill['Unique Rate (%)'] <= 90) & 
                              (high_fill['Unique Rate (%)'] > 5) & 
                              (high_fill['Data Type'].str.contains('object'))]
        binary = high_fill[(high_fill['Unique Count'] <= 5) & (high_fill['Unique Count'] >= 2)]
        
        # Get remaining numeric columns with high fill rate
        numeric_cols = high_fill[
            ~high_fill['Column Name'].isin(
                id_like['Column Name'].tolist() + 
                categorical['Column Name'].tolist() + 
                binary['Column Name'].tolist()
            ) & 
            ~high_fill['Data Type'].str.contains('object')
        ]
        
        # Print column categorization summary with column names
        print("\n=== Column Categories ===")
        print(f"Total columns: {len(meta_df)}")
        print(f"• High fill rate (≥{fill_threshold}%): {len(high_fill)} columns")
        print(f"  - ID-like columns: {len(id_like)} columns")
        if len(id_like) > 0:
            print(f"    {', '.join(id_like['Column Name'].tolist())}")
        
        print(f"  - Categorical columns: {len(categorical)} columns")
        if len(categorical) > 0:
            print(f"    {', '.join(categorical['Column Name'].tolist())}")
        
        print(f"  - Binary/flag columns: {len(binary)} columns")
        if len(binary) > 0:
            print(f"    {', '.join(binary['Column Name'].tolist())}")
            
        print(f"  - Numeric columns: {len(numeric_cols)} columns")
        if len(numeric_cols) > 0:
            print(f"    {', '.join(numeric_cols['Column Name'].tolist())}")
        
        print(f"• Low fill rate (<{fill_threshold}%): {len(low_fill)} columns")
