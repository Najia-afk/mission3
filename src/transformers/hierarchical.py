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
                    
                    # Impute child based on parent
                    mask_parent = X_out[parent_col].notna() & X_out[child_col].isna()
                    for idx in X_out[mask_parent].index:
                        parent_val = X_out.loc[idx, parent_col]
                        if parent_val in parent_child_map:
                            X_out.loc[idx, child_col] = parent_child_map[parent_val]
                    
                    # Impute parent based on child
                    mask_child = X_out[child_col].notna() & X_out[parent_col].isna()
                    for idx in X_out[mask_child].index:
                        child_val = X_out.loc[idx, child_col]
                        if child_val in child_parent_map:
                            X_out.loc[idx, parent_col] = child_parent_map[child_val]
        
        return X_out
