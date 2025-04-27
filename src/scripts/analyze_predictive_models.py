import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    accuracy_score, f1_score, confusion_matrix, classification_report
)
# Regression models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

def run_predictive_modeling(df, target_column, include_pnns=True, categorical_cols=None, numerical_cols=None, skip_plots=True):
    """
    Main function to run predictive modeling on nutrition data
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe with nutrition data
    target_column : str
        The column to predict (e.g. 'nutrition_grade_fr' or 'nutrition-score-fr_100g')
    include_pnns : bool, default=True
        Whether to include PNNS group columns as features
    categorical_cols : list, optional
        Additional categorical columns to include as features
    numerical_cols : list, optional
        Additional numerical columns to include as features
    skip_plots : bool, default=False
        Whether to skip the automatic plotting (useful for Jupyter notebooks)
    
    Returns:
    --------
    dict
        Results and best models
    """
    print(f"Target column: {target_column}")
    
    # Determine if target is categorical or numerical
    is_categorical = df[target_column].dtype == 'object' or df[target_column].nunique() < 10
    task_type = "Classification" if is_categorical else "Regression"
    print(f"Detected task type: {task_type}")
    
    # Create feature lists
    all_categorical_cols = []
    
    # Add PNNS groups if requested
    if include_pnns:
        pnns_cols = [col for col in df.columns if 'pnns_groups' in col]
        all_categorical_cols.extend(pnns_cols)
        print(f"Including PNNS columns: {pnns_cols}")
    
    # Add additional categorical columns
    if categorical_cols:
        all_categorical_cols.extend([c for c in categorical_cols if c != target_column])
    
    # Start with all numerical columns if none specified
    all_numerical_cols = numerical_cols if numerical_cols else df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target from features if it's numerical
    if target_column in all_numerical_cols:
        all_numerical_cols.remove(target_column)
    
    # Prepare feature matrix and target vector
    X = df[all_categorical_cols + all_numerical_cols]
    y = df[target_column]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Categorical features: {len(all_categorical_cols)}")
    print(f"Numerical features: {len(all_numerical_cols)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models based on task type
    if is_categorical:
        # For 'nutrition_grade_fr' classification
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVC': SVC(probability=True, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42)
        }
        
        param_grids = {
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga']
            },
            'SVC': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            },
            'GradientBoosting': {
                'n_estimators': [100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'RandomForest': {
                'n_estimators': [100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
        }
        scoring = 'f1_weighted'
    else:
        # For 'nutrition-score-fr_100g' regression
        models = {
            'ElasticNet': ElasticNet(random_state=42),
            'SVR': SVR(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42)
        }
        
        param_grids = {
            'ElasticNet': {
                'alpha': [0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            },
            'SVR': {
                'C': [0.1, 1.0],
                'kernel': ['linear', 'rbf']
            },
            'GradientBoosting': {
                'n_estimators': [100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            },
            'RandomForest': {
                'n_estimators': [100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
        }
        scoring = 'neg_mean_squared_error'
    
    # Train models and get results
    results, best_models = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, models, param_grids, 
        target_column, is_categorical, scoring
    )
    
    # Get the best model name for returning
    best_model_name = get_best_model_name(results, is_categorical)
    
    # Only create plots if skip_plots is False
    if not skip_plots:
        # Plot results
        if is_categorical:
            plot_classification_results(results, target_column)
        else:
            plot_regression_results(results, target_column)
        
        # Plot feature importance for the best model
        plot_feature_importance(
            best_models[best_model_name], X_train, target_column,
            all_categorical_cols, all_numerical_cols
        )
        
        # For classification, plot confusion matrix
        if is_categorical:
            plot_confusion_matrix(best_models[best_model_name], X_test, y_test, target_column)
    
    return {
        'results': results,
        'best_models': best_models,
        'best_model_name': best_model_name,
        'feature_matrix': X,
        'target': y,
        'is_categorical': is_categorical
    }


def create_pipeline(model, categorical_cols, numerical_cols):
    """Create a preprocessing and model pipeline."""
    
    # Create transformers for categorical and numerical columns
    transformers = []
    
    if numerical_cols:
        transformers.append(('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols))
    
    if categorical_cols:
        transformers.append(('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols))
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, param_grids, target, is_categorical, scoring):
    """Train and evaluate models for classification or regression."""
    
    # Extract categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
    results = {}
    best_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} for {target}...")
        
        # Create pipeline
        pipeline = create_pipeline(model, categorical_cols, numerical_cols)
        
        # Setup GridSearchCV
        param_grid = {f'model__{k}': v for k, v in param_grids[model_name].items()}
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, 
            scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Evaluate based on task type
        if is_categorical:
            # For classification
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name] = {
                'Accuracy': accuracy,
                'F1 Score': f1,
                'Best Params': grid_search.best_params_,
                'Classification Report': classification_report(y_test, y_pred)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
        else:
            # For regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Best Params': grid_search.best_params_
            }
            
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
    
    return results, best_models


def get_best_model_name(results, is_categorical):
    """Get name of the best model based on task type."""
    if is_categorical:
        return max(results, key=lambda m: results[m]['F1 Score'])
    else:
        return min(results, key=lambda m: results[m]['RMSE'])






