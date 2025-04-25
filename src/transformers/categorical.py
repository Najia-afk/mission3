import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CategoricalFeatureImputer(BaseEstimator, TransformerMixin):
    """Handle categorical features with KNN or mode imputation."""
    def __init__(self, min_samples_for_knn=50, knn_neighbors=5, numerical_features=None):
        self.min_samples_for_knn = min_samples_for_knn
        self.knn_neighbors = knn_neighbors
        self.column_imputers = {}
        self.valid_categories = {}
        self.numerical_features = numerical_features  # Related numerical features to use for KNN
        
    def fit(self, X, y=None):
        for col in X.columns:
            unique_vals = X[col].dropna().unique()
            if len(unique_vals) > 0:
                self.valid_categories[col] = list(unique_vals)
                
                # Store most frequent value for each column
                if len(X[col].dropna()) > 0:
                    self.column_imputers[col] = X[col].value_counts().idxmax()
                else:
                    self.column_imputers[col] = "unknown"
        
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        for col in X_out.columns:
            if col in self.column_imputers and X_out[col].isna().any():
                # Simple mode imputation for columns with few non-null values
                if X_out[col].notna().sum() <= self.min_samples_for_knn:
                    X_out[col] = X_out[col].fillna(self.column_imputers[col])
                else:
                    # Apply KNN imputation if numerical features are available
                    if self.numerical_features is not None and len(self.numerical_features) > 0:
                        X_out = self._knn_impute(X_out, col)
                    else:
                        # Fallback to mode imputation
                        X_out[col] = X_out[col].fillna(self.column_imputers[col])
        
        return X_out
    
    def _knn_impute(self, X, target_col):
        """Impute missing values using KNN with numerical features."""
        # Get available numerical features
        num_features = [f for f in self.numerical_features if f in X.columns]
        
        if len(num_features) > 0:
            # Identify rows with missing values in target column
            missing_mask = X[target_col].isna()
            
            if missing_mask.sum() > 0:
                # Get complete rows for training
                complete_rows = X[~missing_mask]
                
                if len(complete_rows) >= self.min_samples_for_knn:
                    # Extract features and target for complete rows
                    X_train = complete_rows[num_features].copy()
                    y_train = complete_rows[target_col].copy()
                    
                    # Extract features for missing rows
                    X_missing = X[missing_mask][num_features].copy()
                    
                    # Fill NAs in training and prediction data with column medians
                    for feat_col in num_features:
                        median_val = X_train[feat_col].median()
                        X_train[feat_col] = X_train[feat_col].fillna(median_val)
                        X_missing[feat_col] = X_missing[feat_col].fillna(median_val)
                    
                    # Apply scaling to normalize the features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_missing_scaled = scaler.transform(X_missing)
                    
                    # Find K nearest neighbors for each missing value
                    from sklearn.neighbors import NearestNeighbors
                    k_neighbors = min(self.knn_neighbors, len(X_train_scaled))
                    nbrs = NearestNeighbors(n_neighbors=k_neighbors)
                    nbrs.fit(X_train_scaled)
                    distances, indices = nbrs.kneighbors(X_missing_scaled)
                    
                    # Get imputed values based on nearest neighbors
                    missing_indices = X[missing_mask].index
                    
                    for i, idx in enumerate(missing_indices):
                        # Get values from nearest neighbors
                        neighbor_indices = indices[i]
                        neighbor_values = y_train.iloc[neighbor_indices].values
                        # Find most common value
                        imputed_val = pd.Series(neighbor_values).value_counts().index[0]
                        X.loc[idx, target_col] = imputed_val
        
        # Apply mode imputation for any remaining missing values
        X[target_col] = X[target_col].fillna(self.column_imputers[target_col])
        
        return X
    
    def inverse_transform(self, X):
        """Ensure imputed values are valid categories"""
        X_out = X.copy()
        
        for col, valid_vals in self.valid_categories.items():
            if col in X_out.columns:
                mask = ~X_out[col].isin(valid_vals) & X_out[col].notna()
                invalid_count = mask.sum()
                
                if invalid_count > 0:
                    print(f"Found {invalid_count} invalid values in column {col}")
                    for idx in X_out[mask].index:
                        if len(valid_vals) > 0:
                            X_out.loc[idx, col] = pd.Series(valid_vals).value_counts().index[0]
        
        return X_out