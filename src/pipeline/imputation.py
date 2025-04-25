import gc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from ..transformers.numerical import MultiStageNumericalImputer, NumericCleanupTransformer
from ..transformers.categorical import CategoricalFeatureImputer
from ..transformers.hierarchical import EnhancedHierarchicalImputer
from ..transformers.special import NutritionScoreImputer

# Define the logger locally
def get_logger(name):
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    
    # Configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
    
    return logger

logger = get_logger(__name__)

# Define default column categories
DEFAULT_SPECIAL_COLS = ['nutrition-score-fr_100g', 'nutrition-score-uk_100g', 'nutrition_grade_fr']
DEFAULT_HIERARCHICAL_COLS = ['pnns_groups_1', 'pnns_groups_2']
DEFAULT_NUTRITION_FEATURES = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'cholesterol_100g',
    'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g',
    'vitamin-a_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g'
]

class ImputationPipeline:
    """A configurable pipeline for imputing missing values in food datasets."""
    
    def __init__(self, 
                special_cols=None, 
                hierarchical_cols=None,
                nutrition_features=None,
                category_mappings=None,
                max_iterations=3,
                convergence_threshold=0.1,
                pnns_iterations=2,
                validate_quality=True,
                confidence_thresholds=None,
                apply_constraints=True):
        """
        Initialize the imputation pipeline with customizable parameters.
        
        Parameters:
        -----------
        special_cols : list, optional
            Columns to handle with special imputation logic (nutrition scores/grades)
        hierarchical_cols : list, optional
            Columns with hierarchical relationships
        nutrition_features : list, optional
            Nutritional columns to use for predictive imputation
        category_mappings : dict, optional
            Manual mappings for categorical variables
        max_iterations : int, default=3
            Maximum number of iterations for the main imputation loop
        convergence_threshold : float, default=0.1
            Percentage improvement threshold to determine convergence
        pnns_iterations : int, default=2
            Number of iterations for PNNS group imputation
        validate_quality : bool, default=True
            Whether to validate imputation quality via cross-validation
        confidence_thresholds : dict, optional
            Confidence thresholds for different imputation levels
        apply_constraints : bool, default=True
            Whether to apply domain-specific constraints
        """
        # Initialize configuration
        self.special_cols = special_cols or DEFAULT_SPECIAL_COLS
        self.hierarchical_cols = hierarchical_cols or DEFAULT_HIERARCHICAL_COLS
        self.nutrition_features = nutrition_features or DEFAULT_NUTRITION_FEATURES
        self.category_mappings = category_mappings
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.pnns_iterations = pnns_iterations
        self.validate_quality = validate_quality
        self.apply_constraints = apply_constraints
        
        # Confidence thresholds for progressive imputation
        self.confidence_thresholds = confidence_thresholds or {
            'high': 0.8,   # High confidence imputation
            'medium': 0.5, # Medium confidence
            'low': 0.2     # Low confidence (last resort)
        }
        
        # Initialize pipelines
        self._init_pipelines()
        
        # Store feature relationships and stratum info
        self.feature_relationships = None
        self.strata = None
        
    def _init_pipelines(self):
        """Initialize the component pipelines."""
        # Pipeline for numerical features
        self.numerical_pipeline = Pipeline([
            ('imputer', MultiStageNumericalImputer())
        ])
            
        # Pipeline for nutrition scores/grades
        self.nutrition_pipeline = Pipeline([
            ('imputer', NutritionScoreImputer())
        ])
        
        # Pipeline for hierarchical categorical features
        self.hierarchical_pipeline = Pipeline([
            ('imputer', EnhancedHierarchicalImputer())
        ])
        
        # Pipeline for other categorical features
        self.categorical_pipeline = Pipeline([
            ('imputer', CategoricalFeatureImputer(self.category_mappings))
        ])
        
        # Final cleanup transformer
        self.cleanup_pipeline = Pipeline([
            ('cleaner', NumericCleanupTransformer())
        ])
    
    def _impute_with_confidence(self, X, pipeline, min_confidence=0.5):
        """
        Perform imputation and track confidence levels.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Data to impute
        pipeline : Pipeline
            Imputation pipeline to use
        min_confidence : float
            Minimum confidence threshold
            
        Returns:
        --------
        tuple
            (imputed_df, confidence_df) - the imputed data and confidence scores
        """
        try:
            # Make a copy to track what values were imputed
            X_orig = X.copy()
            na_mask = X_orig.isna()
            
            # Apply imputation
            X_imputed = pipeline.fit_transform(X)
            
            # Create confidence matrix (same shape as X)
            confidence = pd.DataFrame(1.0, index=X.index, columns=X.columns)
            
            # Set confidence values for imputed cells
            for col in X.columns:
                if na_mask[col].any():
                    # Calculate confidence based on % of non-missing values in used features
                    # This is a simplified approach - real confidence would use model metrics
                    pct_available = (~X[col].isna()).mean()
                    
                    # Adjust confidence - higher for columns with more data
                    col_confidence = min(0.95, max(min_confidence, pct_available))
                    
                    # Set confidence for imputed values
                    confidence.loc[na_mask[col], col] = col_confidence
                    
                    logger.info(f"Imputed {na_mask[col].sum()} values in '{col}' with est. confidence: {col_confidence:.2f}")
            
            return X_imputed, confidence
        
        except Exception as e:
            logger.error(f"Error in confidence-based imputation: {str(e)}")
            # Fall back to regular imputation without confidence
            return pipeline.fit_transform(X), pd.DataFrame(min_confidence, index=X.index, columns=X.columns)
    
    def _validate_imputation_quality(self, df):
        """
        Validate imputation quality through cross-validation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to validate
            
        Returns:
        --------
        dict
            Quality metrics by column
        """
        if len(df) < 1000:
            logger.warning("Dataset too small for reliable cross-validation")
            return {}
            
        logger.info("Starting imputation quality validation via cross-validation...")
        
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Only test columns with sufficient non-missing data
        testable_cols = [col for col in df_copy.columns 
                         if df_copy[col].notna().mean() > 0.5]
        
        if not testable_cols:
            logger.warning("No suitable columns for validation")
            return {}
        
        # Record metrics for each column
        quality_metrics = {}
        
        # Use a subset for validation to keep it quick
        validation_sample = df_copy.sample(min(1000, len(df_copy)), random_state=42)
        
        # For each column, mask some known values and try to recover them
        for col in testable_cols[:10]:  # Limit to first 10 columns to keep it manageable
            try:
                # Skip columns with too few values
                if validation_sample[col].notna().sum() < 50:
                    continue
                
                # Get rows where this column is not null
                test_rows = validation_sample[col].notna()
                if test_rows.sum() < 20:
                    continue
                    
                # Select 20% of these values to mask
                np.random.seed(42)
                mask_rows = np.random.choice(
                    validation_sample[test_rows].index, 
                    size=int(test_rows.sum() * 0.2), 
                    replace=False
                )
                
                # Store original values
                original_values = validation_sample.loc[mask_rows, col].copy()
                
                # Mask these values
                validation_copy = validation_sample.copy()
                validation_copy.loc[mask_rows, col] = np.nan
                
                # Apply imputation
                validation_copy = self.fit_transform(validation_copy, skip_validation=True)
                
                # Compare imputed vs original values
                imputed_values = validation_copy.loc[mask_rows, col]
                
                # Calculate metrics based on data type
                if pd.api.types.is_numeric_dtype(original_values):
                    rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
                    mae = mean_absolute_error(original_values, imputed_values)
                    quality_metrics[col] = {
                        'RMSE': rmse, 
                        'MAE': mae,
                        'MAPE': np.mean(np.abs((original_values - imputed_values) / original_values)) * 100 
                    }
                    logger.info(f"Validation - {col}: RMSE={rmse:.4f}, MAE={mae:.4f}")
                else:
                    accuracy = accuracy_score(original_values, imputed_values)
                    quality_metrics[col] = {'Accuracy': accuracy}
                    logger.info(f"Validation - {col}: Accuracy={accuracy:.2f}")
            
            except Exception as e:
                logger.warning(f"Error validating column {col}: {str(e)}")
                
        return quality_metrics
        
    def _compute_feature_relationships(self, df):
        """
        Compute relationships between features to guide imputation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset to analyze
            
        Returns:
        --------
        dict
            Relationship data between features
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Initialize relationship dict
            relationships = {}
            
            # Find columns with enough data to analyze
            analyzable_cols = [col for col in df.columns 
                              if df[col].notna().mean() > 0.5]
            
            if not analyzable_cols:
                logger.warning("No columns have sufficient non-missing values for relationship analysis")
                return {}
            
            logger.info(f"Analyzing relationships between {len(analyzable_cols)} columns...")
            
            # Sample to speed up analysis if dataframe is large
            df_sample = df.sample(min(5000, len(df)), random_state=42)
            
            # For a subset of columns (to keep runtime reasonable)
            for target_col in analyzable_cols[:min(15, len(analyzable_cols))]:
                # Skip if too many missing values
                if df_sample[target_col].isna().mean() > 0.5:
                    continue
                
                # Get potential predictor columns (exclude target)
                predictor_cols = [c for c in analyzable_cols if c != target_col]
                
                # Filter to rows where target is not null
                mask = df_sample[target_col].notna()
                X = df_sample.loc[mask, predictor_cols]
                y = df_sample.loc[mask, target_col]
                
                # Only use complete cases or impute X for feature importance
                if X.isna().sum().sum() > 0:
                    X = X.fillna(X.median()) if pd.api.types.is_numeric_dtype(X.iloc[:, 0]) else X.fillna(X.mode().iloc[0])
                
                # Skip if too few samples
                if len(y) < 50:
                    continue
                
                # Train model based on data type
                try:
                    if pd.api.types.is_numeric_dtype(y):
                        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    
                    model.fit(X, y)
                    
                    # Get feature importances
                    importances = dict(zip(predictor_cols, model.feature_importances_))
                    
                    # Only keep relevant features
                    relationships[target_col] = {
                        'important_features': {k: v for k, v in sorted(importances.items(), key=lambda i: i[1], reverse=True) 
                                               if v > 0.01}[:10],
                        'model_type': 'regression' if pd.api.types.is_numeric_dtype(y) else 'classification'
                    }
                    
                    logger.info(f"Identified {len(relationships[target_col]['important_features'])} predictive features for {target_col}")
                
                except Exception as e:
                    logger.warning(f"Error analyzing relationships for {target_col}: {str(e)}")
            
            return relationships
            
        except ImportError:
            logger.warning("RandomForest not available - skipping feature relationship analysis")
            return {}
        except Exception as e:
            logger.error(f"Error in feature relationship analysis: {str(e)}")
            return {}
    
    def _stratify_imputation(self, df):
        """
        Stratify data into clusters for more targeted imputation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to stratify
            
        Returns:
        --------
        pandas.Series
            Stratum labels for each row
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            logger.info("Stratifying data for targeted imputation...")
            
            # Select numerical columns with low missingness
            numeric_cols = df.select_dtypes(include=['number']).columns
            candidate_cols = [col for col in numeric_cols 
                            if df[col].notna().mean() > 0.6]
            
            if len(candidate_cols) < 3:
                logger.info("Not enough complete numeric columns for stratification")
                return None
            
            # Use a sample if dataset is large
            df_sample = df[candidate_cols].sample(min(5000, len(df)), random_state=42)
            
            # Complete cases only for clustering
            complete_cases = df_sample.dropna()
            if len(complete_cases) < 100:
                logger.info("Not enough complete cases for reliable stratification")
                return None
            
            # Scale the data - FIX: Convert to numpy before scaling to ensure consistent feature naming
            X = complete_cases.values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Find optimal number of clusters
            best_score = -1
            best_n = 2
            
            for n_clusters in range(2, min(10, len(complete_cases)//100)):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, clusters)
                    if score > best_score:
                        best_score = score
                        best_n = n_clusters
                except Exception as e:
                    logger.warning(f"Error testing {n_clusters} clusters: {str(e)}")
                    continue
            
            # Apply best clustering
            kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            # Function to predict cluster for any row (handles missing values)
            def predict_cluster(row):
                if row[candidate_cols].isna().any():
                    # Impute missing values with means for prediction
                    row_imputed = row[candidate_cols].copy()
                    for col in candidate_cols:
                        if pd.isna(row_imputed[col]):
                            row_imputed[col] = complete_cases[col].mean()
                    
                    # Scale and predict - FIX: Convert to numpy array consistently
                    X_row = row_imputed.values.reshape(1, -1)
                    X_row_scaled = scaler.transform(X_row)
                    return kmeans.predict(X_row_scaled)[0]
                else:
                    # Scale and predict - FIX: Convert to numpy array consistently
                    X_row = row[candidate_cols].values.reshape(1, -1)
                    X_row_scaled = scaler.transform(X_row)
                    return kmeans.predict(X_row_scaled)[0]
            
            # Apply to get cluster for all rows
            strata = df.apply(predict_cluster, axis=1)
            
            # Count samples per stratum
            stratum_counts = strata.value_counts()
            logger.info(f"Created {len(stratum_counts)} strata: {dict(stratum_counts)}")
            
            return strata
            
        except ImportError:
            logger.warning("KMeans not available - skipping stratification")
            return None
        except Exception as e:
            logger.error(f"Error in data stratification: {str(e)}")
            return None
    
    def _apply_domain_constraints(self, df):
        """
        Apply domain-specific constraints to imputed values.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to constrain
            
        Returns:
        --------
        pandas.DataFrame
            Constrained data
        """
        logger.info("Applying domain-specific constraints...")
        
        # Work on a copy to avoid modifying original
        df_result = df.copy()
        
        # 1. Non-negative nutrient values
        nutrient_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g', 
                         'sodium_100g', 'fiber_100g', 'sugars_100g', 'saturated-fat_100g']
        
        for col in nutrient_cols:
            if col in df_result.columns and pd.api.types.is_numeric_dtype(df_result[col]):
                negative_mask = df_result[col] < 0
                if negative_mask.any():
                    logger.info(f"Fixing {negative_mask.sum()} negative values in {col}")
                    df_result.loc[negative_mask, col] = 0
        
        # 2. Logical constraints between columns
        # Total fat >= saturated fat
        if 'fat_100g' in df_result.columns and 'saturated-fat_100g' in df_result.columns:
            invalid_mask = (df_result['saturated-fat_100g'] > df_result['fat_100g']) & \
                          df_result['fat_100g'].notna() & df_result['saturated-fat_100g'].notna()
            
            if invalid_mask.any():
                logger.info(f"Fixing {invalid_mask.sum()} cases where saturated fat > total fat")
                # Set saturated fat equal to total fat in these cases
                df_result.loc[invalid_mask, 'saturated-fat_100g'] = df_result.loc[invalid_mask, 'fat_100g']
        
        # Total carbs >= sugars
        if 'carbohydrates_100g' in df_result.columns and 'sugars_100g' in df_result.columns:
            invalid_mask = (df_result['sugars_100g'] > df_result['carbohydrates_100g']) & \
                          df_result['carbohydrates_100g'].notna() & df_result['sugars_100g'].notna()
            
            if invalid_mask.any():
                logger.info(f"Fixing {invalid_mask.sum()} cases where sugars > total carbohydrates")
                df_result.loc[invalid_mask, 'sugars_100g'] = df_result.loc[invalid_mask, 'carbohydrates_100g']
        
        # 3. Ensure nutrition scores are integers
        for col in ['nutrition-score-fr_100g', 'nutrition-score-uk_100g']:
            if col in df_result.columns and pd.api.types.is_numeric_dtype(df_result[col]):
                non_int_mask = df_result[col].notna() & (df_result[col] % 1 != 0)
                if non_int_mask.any():
                    logger.info(f"Rounding {non_int_mask.sum()} non-integer values in {col}")
                    df_result.loc[non_int_mask, col] = df_result.loc[non_int_mask, col].round()
        
        # 4. Ensure maximal limits
        if 'energy_100g' in df_result.columns:
            # Extremely high values are likely errors
            extreme_mask = df_result['energy_100g'] > 4000  # kcal per 100g is extremely high
            if extreme_mask.any():
                logger.info(f"Capping {extreme_mask.sum()} extreme values in energy_100g")
                df_result.loc[extreme_mask, 'energy_100g'] = 4000
                
        # Nutrients shouldn't exceed 100g per 100g
        for col in ['fat_100g', 'carbohydrates_100g', 'proteins_100g']:
            if col in df_result.columns:
                extreme_mask = df_result[col] > 100
                if extreme_mask.any():
                    logger.info(f"Capping {extreme_mask.sum()} values >100 in {col}")
                    df_result.loc[extreme_mask, col] = 100
        
        return df_result
    
    def impute_pnns_iteratively(self, df):
        """
        Iteratively impute PNNS groups starting with the most confident cases.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to process
            
        Returns:
        --------
        pandas.DataFrame
            The DataFrame with imputed PNNS values
        """
        df_result = df.copy()
        
        # PNNS columns to impute
        pnns_cols = [col for col in self.hierarchical_cols if col in df_result.columns]
        
        if not pnns_cols:
            return df_result
            
        logger.info(f"Starting PNNS iterative imputation. Missing values: {df_result[pnns_cols].isna().sum().sum()}")
        
        # Filter to only available nutrition features
        available_features = [f for f in self.nutrition_features if f in df_result.columns]
        if not available_features:
            logger.info("No nutrition features available for KNN imputation.")
            return df_result
        
        # If we have feature relationships, use them to select the best predictors
        if self.feature_relationships:
            best_features = set()
            for col in pnns_cols:
                if col in self.feature_relationships:
                    best_features.update(self.feature_relationships[col]['important_features'].keys())
            
            # Add any available feature from best_features to available_features
            available_features = list(set(available_features).union(
                set([f for f in best_features if f in df_result.columns])))
            
            logger.info(f"Using {len(available_features)} features for PNNS imputation based on feature relationships")
        
        # Progressive imputation with confidence tracking
        confidence = pd.DataFrame(index=df_result.index, 
                                 columns=pnns_cols,
                                 data=np.nan)
            
        for iteration in range(self.pnns_iterations):
            logger.info(f"PNNS Iteration {iteration+1}/{self.pnns_iterations}")
            
            # 1. Find products with the most features present (most context)
            feature_counts = df_result[available_features].notna().sum(axis=1)
            feature_threshold = np.percentile(feature_counts, 50 + iteration*15)  # Gradually lower threshold
            
            # 2. Focus on products with more context first
            focus_mask = feature_counts >= feature_threshold
            logger.info(f"Focusing on {focus_mask.sum()} products with at least {feature_threshold} nutrition features")
            
            # 3. Apply imputation to this subset
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
                    na_mask = df_subset[col].isna()
                    if na_mask.any():
                        logger.info(f"Imputing {col} for subset...")
                        imputer.fit(df_subset)
                        df_subset = imputer.transform(df_subset)
                        
                        # Set confidence proportional to iteration
                        # Higher iterations = lower confidence
                        conf_value = 0.9 - (0.15 * iteration)
                        confidence.loc[df_subset.index[na_mask], col] = conf_value
                
                # 6. Update the main dataframe with imputed values
                df_result.loc[focus_mask] = df_subset
            
            # 7. Report progress
            missing_now = df_result[pnns_cols].isna().sum().sum()
            logger.info(f"Missing PNNS values after iteration {iteration+1}: {missing_now}")
        
        # Final pass for any remaining missing values using all available data
        remaining_na = df_result[pnns_cols].isna().sum().sum()
        if remaining_na > 0:
            logger.info(f"Final pass: Imputing remaining {remaining_na} missing PNNS values...")
            imputer = CategoricalFeatureImputer(
                numerical_features=available_features,
                min_samples_for_knn=5
            )
            
            # Remember which values were missing before imputation
            na_mask = {col: df_result[col].isna() for col in pnns_cols}
            
            # Impute
            imputer.fit(df_result)
            df_result = imputer.transform(df_result)
            
            # Set low confidence for these final imputations
            for col in pnns_cols:
                if na_mask[col].any():
                    confidence.loc[na_mask[col], col] = 0.3  # Low confidence
        
        logger.info(f"Final missing PNNS values: {df_result[pnns_cols].isna().sum().sum()}")
        return df_result
    
    def fit_transform(self, df, skip_validation=False):
        """
        Apply the full imputation pipeline to a DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to impute
            
        skip_validation : bool, default=False
            Whether to skip quality validation (used internally)
            
        Returns:
        --------
        pandas.DataFrame
            Imputed DataFrame
        """
        logger.info(f"Starting enhanced imputation on DataFrame with shape: {df.shape}")
        logger.info(f"Missing values before imputation: {df.isna().sum().sum()}")
        
        # Validate quality if requested and not already doing validation
        if self.validate_quality and not skip_validation and len(df) >= 1000:
            quality_metrics = self._validate_imputation_quality(df)
            if quality_metrics:
                logger.info("Imputation quality validation results:")
                for col, metrics in quality_metrics.items():
                    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(f"  {col}: {metric_str}")
        
        # Analyze feature relationships to guide imputation
        logger.info("Analyzing feature relationships...")
        self.feature_relationships = self._compute_feature_relationships(df)
        
        # Try to stratify data for better targeted imputation
        logger.info("Creating data strata for targeted imputation...")
        self.strata = self._stratify_imputation(df)
        
        # Create confidence tracking dataframe
        confidence = pd.DataFrame(1.0, index=df.index, columns=df.columns)
        confidence[df.isna()] = np.nan  # Only track confidence for imputed values
        
        # Copy data to avoid modifying the original
        df_result = df.copy()
        
        # Track missing values for convergence checking
        previous_missing = df_result.isna().sum().sum()
        initial_missing = previous_missing
        
        # Iterative imputation loop with confidence levels
        for confidence_level, threshold in self.confidence_thresholds.items():
            logger.info(f"=== Starting {confidence_level} confidence imputation (threshold: {threshold}) ===")
            
            # Track progress within this confidence level
            conf_previous_missing = df_result.isna().sum().sum()
            
            # Iterative imputation loop
            for iteration in range(self.max_iterations):
                logger.info(f"Iteration {iteration+1}/{self.max_iterations} ({confidence_level} confidence)")
                
                # Apply different imputation strategies based on data types
                
                # 1. Apply numerical imputation
                numerical_cols = df_result.select_dtypes(include=['number']).columns.tolist()
                for col in self.special_cols:
                    if col in numerical_cols:
                        numerical_cols.remove(col)
                        
                if numerical_cols and self.strata is not None and not df_result[numerical_cols].isna().sum().sum() == 0:
                    logger.info(f"Processing numerical columns with stratified approach...")
                    
                    # Impute each stratum separately for better results
                    for stratum in sorted(self.strata.unique()):
                        stratum_mask = self.strata == stratum
                        if stratum_mask.sum() > 20:  # Only if enough samples
                            stratum_df = df_result.loc[stratum_mask]
                            if stratum_df[numerical_cols].isna().sum().sum() > 0:
                                logger.info(f"  Imputing stratum {stratum} with {stratum_mask.sum()} samples")
                                imputed_nums, num_confidence = self._impute_with_confidence(
                                    stratum_df[numerical_cols], 
                                    self.numerical_pipeline,
                                    threshold
                                )
                                df_result.loc[stratum_mask, numerical_cols] = imputed_nums
                                confidence.loc[stratum_mask, numerical_cols] = num_confidence
                elif numerical_cols and not df_result[numerical_cols].isna().sum().sum() == 0:
                    logger.info("Processing numerical columns...")
                    imputed_nums, num_confidence = self._impute_with_confidence(
                        df_result[numerical_cols], 
                        self.numerical_pipeline,
                        threshold
                    )
                    df_result[numerical_cols] = imputed_nums
                    confidence.loc[:, numerical_cols] = num_confidence
                
                # 2. Apply nutrition score imputation
                nutrition_cols = [col for col in self.special_cols if col in df_result.columns]
                if nutrition_cols and not df_result[nutrition_cols].isna().sum().sum() == 0:
                    logger.info("Processing nutrition scores and grades...")
                    imputed_nutr, nutr_confidence = self._impute_with_confidence(
                        df_result[nutrition_cols], 
                        self.nutrition_pipeline,
                        threshold
                    )
                    df_result[nutrition_cols] = imputed_nutr
                    confidence.loc[:, nutrition_cols] = nutr_confidence
                
                # 3. Apply hierarchical imputation for available mappings
                hier_cols = [col for col in self.hierarchical_cols if col in df_result.columns]
                if hier_cols and not df_result[hier_cols].isna().sum().sum() == 0:
                    logger.info("Applying hierarchical imputation...")
                    imputed_hier, hier_confidence = self._impute_with_confidence(
                        df_result[hier_cols], 
                        self.hierarchical_pipeline,
                        threshold
                    )
                    df_result[hier_cols] = imputed_hier
                    confidence.loc[:, hier_cols] = hier_confidence
                    
                    # 3.1 Apply iterative PNNS imputation for remaining missing values
                    if df_result[hier_cols].isna().sum().sum() > 0:
                        logger.info("Applying iterative KNN imputation for PNNS groups...")
                        df_result = self.impute_pnns_iteratively(df_result)
                
                # 4. Apply categorical imputation for non-hierarchical columns
                categorical_cols = df_result.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in self.hierarchical_cols:
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                        
                if categorical_cols and not df_result[categorical_cols].isna().sum().sum() == 0:
                    logger.info("Processing remaining categorical columns...")
                    
                    if self.strata is not None:
                        # Process by stratum if feasible
                        for stratum in sorted(self.strata.unique()):
                            stratum_mask = self.strata == stratum
                            if stratum_mask.sum() > 30:  # Only if enough samples
                                stratum_df = df_result.loc[stratum_mask]
                                if stratum_df[categorical_cols].isna().sum().sum() > 0:
                                    logger.info(f"  Imputing categorical vars for stratum {stratum}")
                                    imputed_cats, cat_confidence = self._impute_with_confidence(
                                        stratum_df[categorical_cols], 
                                        self.categorical_pipeline,
                                        threshold
                                    )
                                    df_result.loc[stratum_mask, categorical_cols] = imputed_cats
                                    confidence.loc[stratum_mask, categorical_cols] = cat_confidence
                    else:
                        # Process all together
                        imputed_cats, cat_confidence = self._impute_with_confidence(
                            df_result[categorical_cols], 
                            self.categorical_pipeline,
                            threshold
                        )
                        df_result[categorical_cols] = imputed_cats
                        confidence.loc[:, categorical_cols] = cat_confidence
                
                # 5. Apply domain constraints if configured
                if self.apply_constraints:
                    df_result = self._apply_domain_constraints(df_result)
                
                # Check convergence
                current_missing = df_result.isna().sum().sum()
                improvement = previous_missing - current_missing
                improvement_percentage = (improvement / initial_missing * 100) if initial_missing > 0 else 0
                
                logger.info(f"Missing values after iteration {iteration+1}: {current_missing}")
                logger.info(f"Improvement: {improvement} values ({improvement_percentage:.2f}% of initial missing)")
                
                # Stop if we've converged or no more missing values
                if current_missing == 0:
                    logger.info("All missing values have been imputed!")
                    break
                    
                if improvement == 0:
                    logger.info(f"No improvement in this iteration, continuing to next confidence level")
                    break
                    
                previous_missing = current_missing
            
            # Check if making progress at this confidence level
            conf_current_missing = df_result.isna().sum().sum()
            conf_improvement = conf_previous_missing - conf_current_missing
            logger.info(f"Improvement at {confidence_level} confidence level: {conf_improvement} values")
            
            # Break early if all values imputed
            if conf_current_missing == 0:
                logger.info("All missing values have been imputed!")
                break
        
        # Final pass to clean up any remaining missing values with mode/median
        remaining_missing = df_result.isna().sum().sum()
        if remaining_missing > 0:
            logger.info(f"Final cleanup for {remaining_missing} remaining missing values...")
            
            # For each column with missing values
            for col in df_result.columns[df_result.isna().any()]:
                if pd.api.types.is_numeric_dtype(df_result[col]):
                    # Fill numeric with median
                    fill_value = df_result[col].median()
                    if pd.notna(fill_value):
                        df_result[col].fillna(fill_value, inplace=True)
                        confidence.loc[df_result[col].isna(), col] = 0.1  # Very low confidence
                else:
                    # Fill categorical with mode
                    if not df_result[col].dropna().empty:
                        fill_value = df_result[col].mode().iloc[0] if not df_result[col].mode().empty else None
                        if pd.notna(fill_value):
                            df_result[col].fillna(fill_value, inplace=True)
                            confidence.loc[df_result[col].isna(), col] = 0.1  # Very low confidence
        
        # Detailed analysis of remaining missing values
        self._report_missing_analysis(df_result)
        
        # Save confidence data as an attribute
        self.imputation_confidence = confidence
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Imputation complete. Final missing values: {df_result.isna().sum().sum()}")
        
        return df_result
        
    def _report_missing_analysis(self, df):
        """Generate a detailed report of any remaining missing values."""
        remaining_na = df.isna().sum().sum()
        if remaining_na > 0:
            logger.info("=== Detailed Missing Value Analysis ===")
            # Get missing counts by column
            missing_by_column = df.isna().sum()
            # Filter to only columns with missing values
            missing_cols = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
            
            logger.info(f"Total columns with missing values: {len(missing_cols)}")
            logger.info("Columns with most missing values:")
            
            # Display columns with missing values in a formatted table
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                dtype = str(df[col].dtype)
                logger.info(f"  - {col:<30} | {count:>8} missing ({percentage:.2f}%) | Type: {dtype}")
        else:
            logger.info("=== No Missing Values Remaining ===")
            
    def get_confidence_report(self):
        """
        Get a report on imputation confidence.
        
        Returns:
        --------
        pandas.DataFrame
            Summary of imputation confidence by column
        """
        if not hasattr(self, 'imputation_confidence'):
            return pd.DataFrame()
            
        # Calculate mean confidence by column
        conf_summary = pd.DataFrame(index=self.imputation_confidence.columns)
        
        for col in self.imputation_confidence.columns:
            # Get only the confidence values for imputed cells
            imputed_values = self.imputation_confidence[col].dropna()
            
            if len(imputed_values) > 0:
                conf_summary.loc[col, 'imputed_count'] = len(imputed_values)
                conf_summary.loc[col, 'mean_confidence'] = imputed_values.mean()
                conf_summary.loc[col, 'high_conf_pct'] = (imputed_values >= 0.8).mean() * 100
                conf_summary.loc[col, 'low_conf_pct'] = (imputed_values < 0.5).mean() * 100
            else:
                conf_summary.loc[col, 'imputed_count'] = 0
                conf_summary.loc[col, 'mean_confidence'] = None
                conf_summary.loc[col, 'high_conf_pct'] = None
                conf_summary.loc[col, 'low_conf_pct'] = None
        
        return conf_summary.sort_values('imputed_count', ascending=False)