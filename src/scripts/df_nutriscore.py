import pandas as pd
import numpy as np

# Define the grade ranges
grade_ranges = {
    'A': {'FR': (-15.0, -1.0), 'UK': (-15.0, -1.0)},
    'B': {'FR': (0.0, 2.0), 'UK': (0.0, 2.0)},
    'C': {'FR': (3.0, 10.0), 'UK': (-3.0, 10.0)},
    'D': {'FR': (11.0, 18.0), 'UK': (11.0, 19.0)},
    'E': {'FR': (19.0, 40.0), 'UK': (19.0, 40.0)}
}

def check_and_standardize_nutrition_grades(df):
    """
    Checks if the columns 'nutrition-score-fr_100g', 'nutrition-score-uk_100g', and 'nutrition_grade_fr'
    match the defined grade ranges. If not, sets these fields to NaN.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the relevant columns.
    
    Returns:
    - pd.DataFrame: The modified DataFrame with invalid entries set to NaN.
    """
    
    # Ensure necessary columns are present in the DataFrame
    required_columns = ['nutrition-score-fr_100g', 'nutrition-score-uk_100g', 'nutrition_grade_fr']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: DataFrame must contain the following columns: {required_columns}")
        return df

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        fr_score = row['nutrition-score-fr_100g']
        uk_score = row['nutrition-score-uk_100g']
        grade = row['nutrition_grade_fr']
        if pd.notna(grade) and grade in grade_ranges:
            fr_range = grade_ranges[grade]['FR']
            uk_range = grade_ranges[grade]['UK']
            # Check if FR and UK scores fall within the range for the specified grade
            if not (fr_range[0] <= fr_score <= fr_range[1] and uk_range[0] <= uk_score <= uk_range[1]):
                print(f"'{grade}'  '{fr_range[0]}' <= '{fr_score}' <= '{fr_range[1]}' and '{uk_range[0]}' <= '{uk_score}' <= '{uk_range[1]}' ")
                # If they don't match, set all three fields to NaN
                df.at[index, 'nutrition-score-fr_100g'] = np.nan
                df.at[index, 'nutrition-score-uk_100g'] = np.nan
                df.at[index, 'nutrition_grade_fr'] = np.nan

    return df
