import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor

class MultiStageNumericalImputer(BaseEstimator, TransformerMixin):
    """Handle numerical features with a multi-stage imputation strategy."""
    def __init__(self, random_state=42, knn_neighbors=5, max_iter=50, n_estimators=10,
                max_missing_pct=50, min_samples=10):
        self.random_state = random_state
        self.knn_neighbors = knn_neighbors
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_missing_pct = max_missing_pct
        self.min_samples = min_samples
        self.scaler = RobustScaler()
        self.knn_imputer = None
        self.iter_imputer = None
        self.column_means = {}
        self.skip_imputation_cols = []
        
    def fit(self, X, y=None):
        # Identify columns with too many missing values
        missing_pct = X.isna().mean() * 100
        self.skip_imputation_cols = missing_pct[missing_pct > self.max_missing_pct].index.tolist()
        
        if len(self.skip_imputation_cols) > 0:
            print(f"Skipping imputation for columns with >={self.max_missing_pct}% missing values:")
            for col in self.skip_imputation_cols:
                print(f"  - {col}: {missing_pct[col]:.1f}% missing")
        
        # Store column means for final fallback
        for col in X.columns:
            if col not in self.skip_imputation_cols:
                self.column_means[col] = X[col].mean()
        
        # Scale the data for imputation
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # Set up KNN imputer
        self.knn_imputer = KNNImputer(
            n_neighbors=self.knn_neighbors,
            weights='distance',
            missing_values=np.nan
        )
        
        # Set up iterative imputer
        self.iter_imputer = IterativeImputer(
            max_iter=self.max_iter,
            tol=1e-3,
            random_state=self.random_state,
            estimator=ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=self.random_state),
            verbose=0,
            imputation_order='ascending'
        )
        
        # Fit both imputers
        self.knn_imputer.fit(X_scaled)
        self.iter_imputer.fit(X_scaled)
        
        return self
    
    def transform(self, X):
        # Store original missing mask
        missing_mask = X.isna()
        
        # Create output dataframe
        X_out = X.copy()
        
        # Scale with temporary zeros
        X_temp = X.copy().fillna(0)
        X_scaled = self.scaler.transform(X_temp)
        
        # Convert to DataFrame for tracking progress
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Stage 1: KNN Imputation
        print("  Stage 1: KNN imputation")
        try:
            X_knn_imputed = self.knn_imputer.transform(X_scaled)
            X_knn_df = pd.DataFrame(X_knn_imputed, columns=X.columns, index=X.index)
            
            # Only keep KNN imputations where it worked (not NaN)
            for col in X_scaled_df.columns:
                # Skip columns with too many missing values
                if col in self.skip_imputation_cols:
                    continue
                    
                knn_worked_mask = ~np.isnan(X_knn_df[col]) & missing_mask[col]
                X_scaled_df.loc[knn_worked_mask, col] = X_knn_df.loc[knn_worked_mask, col]
        except Exception as e:
            print(f"  KNN imputation error: {str(e)}, skipping...")
        
        # Track remaining missing values
        still_missing = X_scaled_df.isna()
        missing_count = still_missing.sum().sum()
        if missing_count > 0:
            print(f"  After KNN: {missing_count} values still missing")
            
            # Stage 2: Iterative Imputation
            print("  Stage 2: Iterative imputation")
            try:
                X_iter_imputed = self.iter_imputer.transform(X_scaled_df.fillna(0))
                X_iter_df = pd.DataFrame(X_iter_imputed, columns=X.columns, index=X.index)
                
                # Only use iterative imputation for values still missing after KNN
                for col in X_scaled_df.columns:
                    # Skip columns with too many missing values
                    if col in self.skip_imputation_cols:
                        continue
                        
                    iter_needed_mask = still_missing[col]
                    X_scaled_df.loc[iter_needed_mask, col] = X_iter_df.loc[iter_needed_mask, col]
            except Exception as e:
                print(f"  Iterative imputation error: {str(e)}, skipping...")
            
            # Track remaining missing values
            still_missing = X_scaled_df.isna()
            missing_count = still_missing.sum().sum()
            if missing_count > 0:
                print(f"  After iterative: {missing_count} values still missing")
                
                # Stage 3: Mean Imputation
                print("  Stage 3: Mean imputation")
                for col in X_scaled_df.columns:
                    # Skip columns with too many missing values
                    if col in self.skip_imputation_cols:
                        continue
                        
                    # Apply mean imputation for anything still missing
                    mean_needed_mask = still_missing[col]
                    if mean_needed_mask.any() and col in self.column_means:
                        # Need to scale the mean first
                        col_idx = list(X.columns).index(col)
                        placeholder = np.zeros((1, len(X.columns)))
                        placeholder[0, col_idx] = self.column_means[col]
                        scaled_mean = self.scaler.transform(placeholder)[0, col_idx]
                        X_scaled_df.loc[mean_needed_mask, col] = scaled_mean
        
        # Inverse transform the scaling for all values
        X_result = self.scaler.inverse_transform(X_scaled_df)
        X_result_df = pd.DataFrame(X_result, columns=X.columns, index=X.index)
        
        # Restore NaN values for skip columns
        for col in self.skip_imputation_cols:
            if col in X_out.columns:
                X_result_df.loc[missing_mask[col], col] = np.nan
        
        return X_result_df
    
class NumericCleanupTransformer(BaseEstimator, TransformerMixin):
    """Clean up numerical values like negative nutritional values."""
    def __init__(self):
        self.nutritional_cols = ['_100g', 'energy', 'fat', 'protein', 'carbohydrate', 'sugar', 'salt', 'fiber']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        for col in X_out.select_dtypes(include=['number']).columns:
            # For nutritional columns that can't be negative
            if any(keyword in col.lower() for keyword in self.nutritional_cols):
                # Fix tiny negative values
                tiny_neg_mask = (X_out[col] < 0) & (X_out[col] > -1e-5)
                if tiny_neg_mask.any():
                    X_out.loc[tiny_neg_mask, col] = 0.0
        
        return X_out