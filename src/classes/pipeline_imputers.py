import numpy as np
import pandas as pd
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]
    
    def get_feature_names_out(self, input_features=None):
        return self.columns

class NumericalFeatureImputer(BaseEstimator, TransformerMixin):
    """Handle numerical features with scaling and imputation."""
    def __init__(self, max_iter=50, n_estimators=10, random_state=42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.imputer = None
        
    def fit(self, X, y=None):
        # Apply scaling
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # Create and fit imputer
        self.imputer = IterativeImputer(
            max_iter=self.max_iter,
            tol=1e-3,
            random_state=self.random_state,
            estimator=ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=self.random_state),
            verbose=0,
            imputation_order='ascending'
        )
        self.imputer.fit(X_scaled)
        return self
    
    def transform(self, X):
        # Scale the data
        X_scaled = self.scaler.transform(X.fillna(0))
        
        # Apply imputation
        X_imputed_scaled = self.imputer.transform(X_scaled)
        
        # Inverse transform scaling
        X_imputed = self.scaler.inverse_transform(X_imputed_scaled)
        
        # Convert back to DataFrame
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

class NutritionScoreImputer(BaseEstimator, TransformerMixin):
    """Handle nutrition scores and grades with special logic."""
    def __init__(self):
        self.nutrition_grade_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
        self.inverse_mapping = {v: k for k, v in self.nutrition_grade_mapping.items()}
        self.score_col = 'nutrition-score-fr_100g'
        self.grade_col = 'nutrition_grade_fr'
        self.median_score = None
        self.mode_grade = None
        
    def fit(self, X, y=None):
        # Store median score and mode grade for imputation
        if self.score_col in X.columns:
            self.median_score = X[self.score_col].median()
        
        if self.grade_col in X.columns:
            grades_numeric = self._grades_to_numeric(X[self.grade_col])
            self.mode_grade = grades_numeric.dropna().mode().iloc[0] if not grades_numeric.dropna().empty else 3
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # Handle nutrition score
        if self.score_col in X_out.columns and self.median_score is not None:
            X_out[self.score_col] = X_out[self.score_col].fillna(self.median_score)
        
        # Handle nutrition grade
        if self.grade_col in X_out.columns:
            grades_numeric = self._grades_to_numeric(X_out[self.grade_col])
            
            # Derive grade from score where possible
            if self.score_col in X_out.columns:
                mask = grades_numeric.isna() & X_out[self.score_col].notna()
                grades_numeric.loc[mask] = X_out.loc[mask, self.score_col].map(self._score_to_grade)
            
            # Fill remaining missing with mode
            if grades_numeric.isna().any() and self.mode_grade is not None:
                grades_numeric = grades_numeric.fillna(self.mode_grade)
            
            X_out[self.grade_col] = self._numeric_to_grades(grades_numeric)
        
        return X_out
    
    def _grades_to_numeric(self, series):
        """Convert letter grades to numeric values."""
        return series.map(lambda x: self.nutrition_grade_mapping.get(str(x).lower(), np.nan))
    
    def _numeric_to_grades(self, series):
        """Convert numeric values back to letter grades."""
        # Round to nearest integer first
        series = series.round()
        
        # Map back to letter grades, ensuring values are in valid range
        return series.map(
            lambda x: self.inverse_mapping.get(
                min(max(int(x) if pd.notna(x) else 3, 1), 5),
                'c'  # Default to 'c' if conversion fails
            ) if pd.notna(x) else np.nan
        )
    
    def _score_to_grade(self, score):
        """Map nutrition score to grade value."""
        if pd.isna(score):
            return np.nan
        elif score <= 0:
            return 1  # 'a'
        elif score <= 3:
            return 2  # 'b'
        elif score <= 10:
            return 3  # 'c'
        elif score <= 18:
            return 4  # 'd'
        else:
            return 5  # 'e'

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

def create_imputation_pipeline(category_mappings=None, max_iterations=3, convergence_threshold=0.1):
    """Create a complete imputation pipeline."""
    
    # Define column types
    special_cols = ['nutrition-score-fr_100g', 'nutrition-score-uk_100g', 
                   'nutrition_grade_fr']
    
    hierarchical_cols = ['pnns_groups_1', 'pnns_groups_2']
    
    # Nutrition features for KNN imputation
    nutrition_features = [
        'energy_100g', 'fat_100g', 'saturated-fat_100g', 
        'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
        'proteins_100g', 'salt_100g', 'nutrition-score-fr_100g'
    ]
    
    # Pipeline for numerical features
    numerical_pipeline = Pipeline([
        ('imputer', NumericalFeatureImputer())
    ])
    
    # Pipeline for nutrition scores/grades
    nutrition_pipeline = Pipeline([
        ('imputer', NutritionScoreImputer())
    ])
    
    # Pipeline for hierarchical categorical features
    hierarchical_pipeline = Pipeline([
        ('imputer', EnhancedHierarchicalImputer())
    ])
    
    # Pipeline for other categorical features
    categorical_pipeline = Pipeline([
        ('imputer', CategoricalFeatureImputer(category_mappings))
    ])
    
    # Final cleanup transformer
    cleanup_pipeline = Pipeline([
        ('cleaner', NumericCleanupTransformer())
    ])
    
    def impute_pnns_iteratively(df, iterations=3):
        """Iteratively impute PNNS groups starting with the most confident cases."""
        df_result = df.copy()
        
        # PNNS columns to impute
        pnns_cols = [col for col in hierarchical_cols if col in df_result.columns]
        
        if not pnns_cols:
            return df_result
            
        print(f"Starting PNNS iterative imputation. Missing values: {df_result[pnns_cols].isna().sum().sum()}")
        
        # Filter to only available nutrition features
        available_features = [f for f in nutrition_features if f in df_result.columns]
        if not available_features:
            print("  No nutrition features available for KNN imputation.")
            return df_result
            
        for iteration in range(iterations):
            print(f"\n  PNNS Iteration {iteration+1}/{iterations}")
            
            # 1. Find products with the most features present (most context)
            feature_counts = df_result[available_features].notna().sum(axis=1)
            feature_threshold = np.percentile(feature_counts, 50 + iteration*15)  # Gradually lower threshold
            
            # 2. Focus on products with more context first
            focus_mask = feature_counts >= feature_threshold
            print(f"  Focusing on {focus_mask.sum()} products with at least {feature_threshold} nutrition features")
            
            # 3. Apply KNN imputation to this subset
            imputer = CategoricalFeatureImputer(
                min_samples_for_knn=20, 
                knn_neighbors=5, 
                numerical_features=available_features
            )
            
            # 4. Get subset to process in this iteration
            df_subset = df_result[focus_mask].copy()
            
            # 5. Impute missing values for this subset
            if not df_subset.empty:
                for col in pnns_cols:
                    if df_subset[col].isna().any():
                        print(f"    Imputing {col} for subset...")
                        imputer.fit(df_subset)
                        df_subset = imputer.transform(df_subset)
                
                # 6. Update the main dataframe with imputed values
                df_result.loc[focus_mask] = df_subset
            
            # 7. Report progress
            missing_now = df_result[pnns_cols].isna().sum().sum()
            print(f"  Missing PNNS values after iteration {iteration+1}: {missing_now}")
        
        # Final pass for any remaining missing values using all available data
        print("\n  Final pass: Imputing any remaining missing PNNS values...")
        imputer = CategoricalFeatureImputer(
            numerical_features=available_features,
            min_samples_for_knn=5
        )
        imputer.fit(df_result)
        df_result = imputer.transform(df_result)
        
        print(f"  Final missing PNNS values: {df_result[pnns_cols].isna().sum().sum()}")
        
        return df_result
    
    def impute_missing_values(df):
        """Function that applies the pipeline to a dataframe with multiple iterations."""
        print(f"Starting imputation on DataFrame with shape: {df.shape}")
        print(f"Missing values before imputation: {df.isna().sum().sum()}")
        
        # Copy data to avoid modifying the original
        df_result = df.copy()
        
        # Track missing values for convergence checking
        previous_missing = df_result.isna().sum().sum()
        initial_missing = previous_missing
        
        # Iterative imputation loop
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration+1}/{max_iterations}")
            
            # 1. Apply numerical imputation
            numerical_cols = df_result.select_dtypes(include=['number']).columns.tolist()
            for col in special_cols:
                if col in numerical_cols:
                    numerical_cols.remove(col)
                    
            if numerical_cols:
                print("Processing numerical columns...")
                df_result[numerical_cols] = numerical_pipeline.fit_transform(df_result[numerical_cols])
            
            # 2. Apply nutrition score imputation
            nutrition_cols = [col for col in special_cols if col in df_result.columns]
            if nutrition_cols:
                print("Processing nutrition scores and grades...")
                df_result[nutrition_cols] = nutrition_pipeline.fit_transform(df_result[nutrition_cols])
            
            # 3. Apply hierarchical imputation for available mappings
            hier_cols = [col for col in hierarchical_cols if col in df_result.columns]
            if hier_cols:
                print("Applying hierarchical imputation...")
                df_result[hier_cols] = hierarchical_pipeline.fit_transform(df_result[hier_cols])
                
                # 3.1 Apply iterative PNNS imputation for remaining missing values
                print("Applying iterative KNN imputation for PNNS groups...")
                df_result = impute_pnns_iteratively(df_result)
            
            # 4. Apply categorical imputation for non-hierarchical columns
            categorical_cols = df_result.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in hierarchical_cols:
                if col in categorical_cols:
                    categorical_cols.remove(col)
                    
            if categorical_cols:
                print("Processing remaining categorical columns...")
                df_result[categorical_cols] = categorical_pipeline.fit_transform(df_result[categorical_cols])
            
            # 5. Apply final cleanup
            print("Cleaning up numerical values...")
            df_result = cleanup_pipeline.fit_transform(df_result)
            
            # Check convergence
            current_missing = df_result.isna().sum().sum()
            improvement = previous_missing - current_missing
            improvement_percentage = (improvement / initial_missing * 100) if initial_missing > 0 else 0
            
            print(f"Missing values after iteration {iteration+1}: {current_missing}")
            print(f"Improvement: {improvement} values ({improvement_percentage:.2f}% of initial missing)")
            
            # Stop if we've converged or no more missing values
            if improvement_percentage < convergence_threshold or current_missing == 0:
                if current_missing == 0:
                    print("All missing values have been imputed!")
                else:
                    print(f"Convergence reached (improvement below {convergence_threshold}% threshold)")
                break
                
            previous_missing = current_missing
            
        # Detailed analysis of remaining missing values
        remaining_na = df_result.isna().sum().sum()
        if remaining_na > 0:
            print("\n=== Detailed Missing Value Analysis ===")
            # Get missing counts by column
            missing_by_column = df_result.isna().sum()
            # Filter to only columns with missing values
            missing_cols = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
            
            print(f"Total columns with missing values: {len(missing_cols)}")
            print("\nColumns with most missing values:")
            
            # Display columns with missing values in a formatted table
            for col, count in missing_cols.items():
                percentage = (count / len(df_result)) * 100
                dtype = str(df_result[col].dtype)
                print(f"  - {col:<30} | {count:>8} missing ({percentage:.2f}%) | Type: {dtype}")
            
            print("\nApplying final fallback strategy...")
            
            # Now apply the fallback strategy
            df_result = df_result.ffill().bfill()
            
            for col in df_result.columns:
                if df_result[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df_result[col]):
                        df_result[col] = df_result[col].fillna(df_result[col].median())
                    else:
                        most_common = df_result[col].value_counts().index[0] if len(df_result[col].dropna()) > 0 else "unknown"
                        df_result[col] = df_result[col].fillna(most_common)
        
        # Force garbage collection
        gc.collect()
        
        print(f"Imputation complete. Final missing values: {df_result.isna().sum().sum()}")
        
        return df_result
    
    return impute_missing_values