import gc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..transformers.numerical import MultiStageNumericalImputer, NumericCleanupTransformer
from ..transformers.categorical import CategoricalFeatureImputer
from ..transformers.hierarchical import EnhancedHierarchicalImputer
from ..transformers.special import NutritionScoreImputer


def create_imputation_pipeline(category_mappings=None, max_iterations=3, convergence_threshold=0.1):
    """Create a complete imputation pipeline."""
    
    # Define column types
    special_cols = ['nutrition-score-fr_100g', 'nutrition-score-uk_100g', 
                   'nutrition_grade_fr']
    
    hierarchical_cols = ['pnns_groups_1', 'pnns_groups_2']
    
    # Nutrition features for KNN imputation
    nutrition_features = [
         'energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'cholesterol_100g',
         'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g',
         'vitamin-a_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g'
    ]
    
    # Pipeline for numerical features
    numerical_pipeline = Pipeline([
        ('imputer', MultiStageNumericalImputer())
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
    
    def impute_pnns_iteratively(df, iterations=2):
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
            
            # # Optionally apply a final fallback strategy for remaining missing values
            # # This could be a simple forward-fill, backward-fill, or median/mode imputation
            # print("\nApplying final fallback strategy...")
            # 
            # # Now apply the fallback strategy
            # df_result = df_result.ffill().bfill()
            # 
            # for col in df_result.columns:
            #     if df_result[col].isna().any():
            #         if pd.api.types.is_numeric_dtype(df_result[col]):
            #             df_result[col] = df_result[col].fillna(df_result[col].median())
            #         else:
            #             most_common = df_result[col].value_counts().index[0] if len(df_result[col].dropna()) > 0 else "unknown"
            #             df_result[col] = df_result[col].fillna(most_common)
            #    
        # Force garbage collection
        gc.collect()
        
        print(f"Imputation complete. Final missing values: {df_result.isna().sum().sum()}")
        
        return df_result
    
    return impute_missing_values

def impute_pnns_iteratively(df, iterations=2):
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