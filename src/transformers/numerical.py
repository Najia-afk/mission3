import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

class MultiStageNumericalImputer(BaseEstimator, TransformerMixin):
    """Handle numerical features with a multi-stage imputation strategy."""
    def __init__(self, random_state=42, knn_neighbors=5, max_iter=10, n_estimators=5,
                max_missing_pct=50, min_samples=10, verbose=0, n_jobs=-1):
        self.random_state = random_state
        self.knn_neighbors = knn_neighbors
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_missing_pct = max_missing_pct
        self.min_samples = min_samples
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.scaler = RobustScaler()
        self.knn_imputer = None
        self.iter_imputer = None
        self.simple_imputer = None
        self.column_means = {}
        self.skip_imputation_cols = []
        
    def fit(self, X, y=None):
        # Identify columns with too many missing values
        missing_pct = X.isna().mean() * 100
        self.skip_imputation_cols = missing_pct[missing_pct > self.max_missing_pct].index.tolist()
        
        if self.verbose > 0 and len(self.skip_imputation_cols) > 0:
            print(f"  Note: Skipping {len(self.skip_imputation_cols)} columns with too many missing values.")
        
        # Store column means for final fallback
        for col in X.columns:
            if col not in self.skip_imputation_cols:
                self.column_means[col] = X[col].mean()
        
        # Scale the data for imputation - keep NaNs so imputers can see them
        # We fit the scaler on data with filled NaNs to get valid scaling parameters
        self.scaler.fit(X.fillna(X.median().fillna(0)))
        X_scaled = self.scaler.transform(X)
        
        # Determine strategy based on dataset size
        n_samples = X.shape[0]
        
        if n_samples < 5000:
            # Small dataset: Use KNN and Iterative with Trees
            self.knn_imputer = KNNImputer(
                n_neighbors=self.knn_neighbors,
                weights='distance'
            )
            self.iter_imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                estimator=ExtraTreesRegressor(
                    n_estimators=self.n_estimators, 
                    random_state=self.random_state
                ),
                verbose=0
            )
        else:
            # Large dataset: Skip KNN, use Iterative with BayesianRidge (much faster)
            self.knn_imputer = None
            self.iter_imputer = IterativeImputer(
                max_iter=min(5, self.max_iter),
                random_state=self.random_state,
                estimator=BayesianRidge(),
                verbose=0
            )
            
        self.simple_imputer = SimpleImputer(strategy='median')
        
        # Fit imputers on scaled data with NaNs
        if self.knn_imputer:
            self.knn_imputer.fit(X_scaled)
        self.iter_imputer.fit(X_scaled)
        self.simple_imputer.fit(X_scaled.fillna(0) if hasattr(X_scaled, 'fillna') else np.nan_to_num(X_scaled))
        
        return self
    
    def transform(self, X):
        # Store original missing mask
        missing_mask = X.isna()
        
        # Create output dataframe
        X_out = X.copy()
        
        # Scale the data - keep NaNs
        X_scaled = self.scaler.transform(X)
        
        # Convert to DataFrame for tracking progress
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Stage 1: KNN Imputation (only for small datasets)
        if self.knn_imputer:
            if self.verbose > 1:
                print("  Stage 1: KNN imputation")
            try:
                X_knn_imputed = self.knn_imputer.transform(X_scaled_df)
                X_knn_df = pd.DataFrame(X_knn_imputed, columns=X.columns, index=X.index)
                
                # Update values
                for col in X_scaled_df.columns:
                    if col in self.skip_imputation_cols: continue
                    mask = missing_mask[col] & ~np.isnan(X_knn_df[col])
                    X_scaled_df.loc[mask, col] = X_knn_df.loc[mask, col]
            except Exception:
                pass
        
        # Stage 2: Iterative Imputation
        still_missing_mask = X_scaled_df.isna()
        if still_missing_mask.any().any():
            if self.verbose > 1:
                print("  Stage 2: Iterative imputation")
            try:
                X_iter_imputed = self.iter_imputer.transform(X_scaled_df)
                X_iter_df = pd.DataFrame(X_iter_imputed, columns=X.columns, index=X.index)
                
                # Update values
                for col in X_scaled_df.columns:
                    if col in self.skip_imputation_cols: continue
                    mask = still_missing_mask[col] & ~np.isnan(X_iter_df[col])
                    X_scaled_df.loc[mask, col] = X_iter_df.loc[mask, col]
            except Exception:
                pass
                
        # Stage 3: Simple Imputation (Fallback)
        still_missing_mask = X_scaled_df.isna()
        if still_missing_mask.any().any():
            if self.verbose > 1:
                print("  Stage 3: Simple imputation")
            try:
                X_simple_imputed = self.simple_imputer.transform(X_scaled_df)
                X_simple_df = pd.DataFrame(X_simple_imputed, columns=X.columns, index=X.index)
                
                for col in X_scaled_df.columns:
                    if col in self.skip_imputation_cols: continue
                    mask = still_missing_mask[col]
                    X_scaled_df.loc[mask, col] = X_simple_df.loc[mask, col]
            except Exception:
                pass
        
        # Inverse transform
        X_result = self.scaler.inverse_transform(X_scaled_df.fillna(0))
        X_result_df = pd.DataFrame(X_result, columns=X.columns, index=X.index)
        
        # Restore NaNs for skip columns
        for col in self.skip_imputation_cols:
            if col in X_result_df.columns:
                X_result_df.loc[missing_mask[col], col] = np.nan
            
        if self.verbose > 0:
            final_missing = X_result_df.isna().sum().sum()
            initial_missing = missing_mask.sum().sum()
            imputed_count = initial_missing - final_missing
            print(f"  Imputation complete: {imputed_count} values imputed, {final_missing} still missing.")
            
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