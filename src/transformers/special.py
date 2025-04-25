import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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