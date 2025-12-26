import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EnhancedHierarchicalImputer(BaseEstimator, TransformerMixin):
    """Improved hierarchical imputation for related categorical variables."""
    def __init__(self, hierarchical_pairs=None):
        self.hierarchical_pairs = hierarchical_pairs or [
            ('pnns_groups_1', 'pnns_groups_2')
        ]
        self.parent_child_mappings = {}
        self.child_parent_mappings = {}
        
    def fit(self, X, y=None):
        for parent_col, child_col in self.hierarchical_pairs:
            if parent_col in X.columns and child_col in X.columns:
                valid_data = X[[parent_col, child_col]].dropna()
                
                parent_child_map = {}
                child_parent_map = {}
                
                # Learn mappings
                for child, group in valid_data.groupby(child_col):
                    if len(group) > 0:
                        most_common_parent = group[parent_col].value_counts().idxmax()
                        child_parent_map[child] = most_common_parent
                
                for parent, group in valid_data.groupby(parent_col):
                    if len(group) > 0:
                        most_common_child = group[child_col].value_counts().idxmax()
                        parent_child_map[parent] = most_common_child
                
                self.parent_child_mappings[(parent_col, child_col)] = parent_child_map
                self.child_parent_mappings[(parent_col, child_col)] = child_parent_map
        
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        for parent_col, child_col in self.hierarchical_pairs:
            if parent_col in X_out.columns and child_col in X_out.columns:
                if (parent_col, child_col) in self.parent_child_mappings:
                    parent_child_map = self.parent_child_mappings[(parent_col, child_col)]
                    child_parent_map = self.child_parent_mappings[(parent_col, child_col)]
                    
                    # Impute child based on parent (Vectorized)
                    mask_parent = X_out[parent_col].notna() & X_out[child_col].isna()
                    if mask_parent.any():
                        X_out.loc[mask_parent, child_col] = X_out.loc[mask_parent, parent_col].map(parent_child_map)
                    
                    # Impute parent based on child (Vectorized)
                    mask_child = X_out[child_col].notna() & X_out[parent_col].isna()
                    if mask_child.any():
                        X_out.loc[mask_child, parent_col] = X_out.loc[mask_child, child_col].map(child_parent_map)
        
        return X_out
