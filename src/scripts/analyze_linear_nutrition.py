from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_nutrition_score_relationships(df, threshold=0.95):
    """
    Extract the linear relationship between FR and UK nutrition scores for each grade.
    
    Parameters:
        df (pd.DataFrame): DataFrame with nutrition data
        threshold (float): Data percentage threshold to use (0-1.0)
        
    Returns:
        dict: Dictionary of regression models by grade
    """
    # Prepare data similar to the visualization function
    df = df[['nutrition_grade_fr', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']].copy()
    
    df.rename(columns={
        'nutrition-score-fr_100g': 'fr_score',
        'nutrition-score-uk_100g': 'uk_score'
    }, inplace=True)
    
    df['nutrition_grade_fr'] = df['nutrition_grade_fr'].astype(str).str.lower()
    df.dropna(subset=['nutrition_grade_fr', 'fr_score', 'uk_score'], inplace=True)
    
    # Define valid grades
    valid_grades = ['a', 'b', 'c', 'd', 'e']
    df = df[df['nutrition_grade_fr'].isin(valid_grades)]
    
    # Group by unique combinations to reduce data size (similar to visualization function)
    df_grouped = df.groupby(['nutrition_grade_fr', 'fr_score', 'uk_score']).size().reset_index(name='Frequency')
    
    # Calculate cumulative frequency for thresholding
    df_grouped = df_grouped.sort_values('Frequency', ascending=False)
    total_freq = df_grouped['Frequency'].sum()
    df_grouped['Cumulative Percentage'] = df_grouped['Frequency'].cumsum() / total_freq * 100
    
    # Filter by threshold
    threshold_pct = threshold * 100
    filtered_df = df_grouped[df_grouped['Cumulative Percentage'] <= threshold_pct]
    
    # Calculate regression for each grade
    regression_models = {}
    regression_equations = {}
    
    for grade in valid_grades:
        grade_data = filtered_df[filtered_df['nutrition_grade_fr'] == grade]
        
        if len(grade_data) < 3:  # Need at least 3 points for regression
            continue
            
        try:
            # Extract scores for regression
            X = grade_data['fr_score'].values.reshape(-1, 1)
            y = grade_data['uk_score'].values
            
            # Fit regression with frequency weighting
            reg = LinearRegression().fit(X, y, sample_weight=grade_data['Frequency'])
            
            # Store model
            regression_models[grade] = reg
            
            # Create equation string: UK_score = slope * FR_score + intercept
            slope = reg.coef_[0]
            intercept = reg.intercept_
            equation = f"UK_score = {slope:.4f} * FR_score + {intercept:.4f}"
            r_squared = reg.score(X, y)
            
            regression_equations[grade] = {
                'equation': equation,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared
            }
            
        except Exception as e:
            print(f"Error calculating regression for grade {grade}: {e}")
    
    print("\nNutrition Score Linear Relationships (at {:.0f}% threshold):".format(threshold*100))
    print("-" * 70)
    for grade, info in regression_equations.items():
        print(f"Grade {grade.upper()}: {info['equation']} (R² = {info['r_squared']:.4f})")
    
    return regression_models, regression_equations

def align_french_nutrition_scores(df):
    """
    Validate and align French nutrition scores with their expected values based on nutrition grade.
    After validation, the UK scores are dropped as they're not needed for French customers.
    The function also converts scores to integers and generates a comparison plot.
    
    Parameters:
        df (pd.DataFrame): DataFrame with nutrition data including grades and scores
        
    Returns:
        pd.DataFrame: DataFrame with aligned French nutrition scores
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Define grade-specific equations from the 99% threshold analysis
    grade_equations = {
        'a': {'slope': 1.0000, 'intercept': 0.0000, 'r_squared': 1.0000},
        'b': {'slope': 0.9998, 'intercept': -0.0048, 'r_squared': 0.6400},
        'c': {'slope': 1.0946, 'intercept': -0.7569, 'r_squared': 0.6044},
        'd': {'slope': 1.1226, 'intercept': -1.6237, 'r_squared': 0.6194},
        'e': {'slope': 1.3735, 'intercept': -8.6686, 'r_squared': 0.7880}
    }
    
    # Standardize column names
    if 'nutrition-score-fr_100g' in df.columns and 'nutrition-score-uk_100g' in df.columns:
        df.rename(columns={
            'nutrition-score-fr_100g': 'fr_score',
            'nutrition-score-uk_100g': 'uk_score'
        }, inplace=True)
    
    # Store original scores for comparison
    df['original_fr_score'] = df['fr_score'].copy()
    
    # Ensure grade is lowercase
    if 'nutrition_grade_fr' in df.columns:
        df['nutrition_grade_fr'] = df['nutrition_grade_fr'].astype(str).str.lower()
    
    # Create flag for tracking changes
    df['score_adjusted'] = False
    
    # Process each grade separately
    for grade, equation in grade_equations.items():
        # Filter for this grade
        grade_mask = df['nutrition_grade_fr'] == grade
        
        # Identify cases where both scores are available
        both_scores_mask = grade_mask & ~df['fr_score'].isna() & ~df['uk_score'].isna()
        
        # For rows with both scores, check if they follow the expected relationship
        df_both = df[both_scores_mask].copy()
        
        if len(df_both) > 0:
            # Calculate expected UK score based on FR score and equation
            df_both['expected_uk'] = df_both['fr_score'] * equation['slope'] + equation['intercept']
            
            # Find significant deviations (adjust threshold based on your needs)
            deviation_threshold = 0.5  # Allow 0.5 points of deviation
            deviation_mask = abs(df_both['uk_score'] - df_both['expected_uk']) > deviation_threshold
            
            # Update FR scores for cases with significant deviations
            df_both.loc[deviation_mask, 'corrected_fr'] = (df_both.loc[deviation_mask, 'uk_score'] - 
                                                        equation['intercept']) / equation['slope']
            
            # Only update scores if the correction would be significant
            significant_correction_mask = deviation_mask & (abs(df_both['fr_score'] - df_both['corrected_fr']) > 0.1)
            
            # Apply corrections to main dataframe
            for idx, row in df_both[significant_correction_mask].iterrows():
                df.loc[idx, 'fr_score'] = row['corrected_fr']
                df.loc[idx, 'score_adjusted'] = True
        
        # Process missing French scores (where UK score is available)
        fr_missing_mask = grade_mask & df['fr_score'].isna() & ~df['uk_score'].isna()
        
        # Calculate French score from UK score using the inverse of the grade equation
        for idx, row in df[fr_missing_mask].iterrows():
            df.loc[idx, 'fr_score'] = (row['uk_score'] - equation['intercept']) / equation['slope']
            df.loc[idx, 'score_adjusted'] = True
    
    # CONVERT TO INTEGER
    df['fr_score'] = df['fr_score'].round().astype('Int64')  # Int64 preserves NaN values
    
    # Print summary of adjustments
    adjusted_count = df['score_adjusted'].sum()
    total_count = len(df)
    print(f"Adjusted {adjusted_count} out of {total_count} scores ({adjusted_count/total_count:.1%})")
    
    # CREATE COMPARISON PLOT
    # Create a figure with 2 subplots stacked vertically
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Original FR Nutrition Score Distribution", 
                                       "Adjusted FR Nutrition Score Distribution"),
                        shared_xaxes=True,
                        vertical_spacing=0.1)
    
    # Compute statistics for before and after
    before_mean = df['original_fr_score'].mean()
    before_std = df['original_fr_score'].std()
    after_mean = df['fr_score'].mean()
    after_std = df['fr_score'].std()
    
    # Add histograms
    fig.add_trace(
        go.Histogram(
            x=df['original_fr_score'],
            name="Original Scores",
            marker_color='rgba(56, 108, 176, 0.7)',
            xbins=dict(start=-30, end=30, size=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=df['fr_score'],
            name="Adjusted Scores",
            marker_color='rgba(210, 76, 60, 0.7)',
            xbins=dict(start=-30, end=30, size=1)
        ),
        row=2, col=1
    )
    
    # Add vertical lines for means
    fig.add_shape(
        type="line", line=dict(dash="dash", width=2, color="blue"),
        x0=before_mean, y0=0, x1=before_mean, y1=1, xref="x", yref="paper",
        row=1, col=1
    )
    
    fig.add_shape(
        type="line", line=dict(dash="dash", width=2, color="red"),
        x0=after_mean, y0=0, x1=after_mean, y1=1, xref="x2", yref="paper",
        row=2, col=1
    )
    
    # Add annotations for mean and std
    fig.add_annotation(
        x=before_mean, y=0.85,
        text=f"Mean: {before_mean:.2f}<br>Std: {before_std:.2f}",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(color="blue", size=12),
        bgcolor="white",
        bordercolor="blue",
        borderwidth=1,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=after_mean, y=0.85,
        text=f"Mean: {after_mean:.2f}<br>Std: {after_std:.2f}",
        showarrow=False,
        xref="x2", yref="paper",
        font=dict(color="red", size=12),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1,
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Comparison of FR Nutrition Scores Before and After Alignment",
        barmode='overlay',
        height=800,
        width=1000
    )
    
    # Show plot
    fig.show()
    
    # Restore original column names
    df.rename(columns={
        'fr_score': 'nutrition-score-fr_100g',
    }, inplace=True)
    
    # Drop UK score and temporary columns
    df.drop(['uk_score', 'score_adjusted', 'original_fr_score'], axis=1, errors='ignore', inplace=True)
    
    return df