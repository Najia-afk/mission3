import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# VALIDATION CALCULATION FUNCTIONS
def check_sodium_salt_relationship(df):
    """Calculate validation metrics for the sodium-salt relationship (salt = sodium * 2.5)"""
    validation_result = {}
    df_validated = df.copy()
    
    mask_both = (~df['sodium_100g'].isna()) & (~df['salt_100g'].isna())
    if mask_both.sum() > 0:
        filtered_df = df.loc[mask_both]
        expected_salt = filtered_df['sodium_100g'] * 2.5
        actual_salt = filtered_df['salt_100g']
        
        deviation = np.abs((actual_salt - expected_salt) / expected_salt * 100)
        inconsistent_salt = (deviation > 10).sum()
        
        validation_result = {
            'Relationship': 'Sodium-Salt',
            'Description': 'salt = sodium * 2.5',
            'Total Checked': mask_both.sum(),
            'Consistent': mask_both.sum() - inconsistent_salt,
            'Inconsistent': inconsistent_salt,
            'Consistency %': (1 - inconsistent_salt / mask_both.sum()) * 100 if mask_both.sum() > 0 else 0
        }
        
        # Fix inconsistent values
        fix_indices = filtered_df[deviation > 10].index
        if len(fix_indices) > 0:
            df_validated.loc[fix_indices, 'salt_100g'] = df_validated.loc[fix_indices, 'sodium_100g'] * 2.5
    
    return validation_result, df_validated

def check_energy_macronutrients_relationship(df):
    """
    Calculate validation metrics for energy vs macronutrients relationship and
    fix inconsistencies and missing values using the Atwater factors formula.
    """
    validation_result = {}
    df_validated = df.copy()
    
    # Step 1: Fix cases where energy value doesn't align with macronutrients
    mask_all_present = (~df['energy_100g'].isna()) & (~df['proteins_100g'].isna()) & (~df['carbohydrates_100g'].isna()) & (~df['fat_100g'].isna())
    if mask_all_present.sum() > 0:
        filtered_df = df.loc[mask_all_present]
        
        expected_energy = (
            filtered_df['proteins_100g'] * 4 + 
            filtered_df['carbohydrates_100g'] * 4 + 
            filtered_df['fat_100g'] * 9
        )
        actual_energy = filtered_df['energy_100g']
        
        deviation = np.abs((actual_energy - expected_energy) / expected_energy * 100)
        inconsistent_energy = (deviation > 20).sum()
        
        # Fix inconsistent energy values
        fix_indices = filtered_df[deviation > 20].index
        if len(fix_indices) > 0:
            df_validated.loc[fix_indices, 'energy_100g'] = np.minimum(
                # Calculate energy from macronutrients but cap at 950
                np.minimum(
                    (df_validated.loc[fix_indices, 'proteins_100g'] * 4 + 
                     df_validated.loc[fix_indices, 'carbohydrates_100g'] * 4 + 
                     df_validated.loc[fix_indices, 'fat_100g'] * 9),
                    950  # Maximum reasonable energy value
                ),
                950  # Ensure we don't exceed 950 kcal/100g
            )
        
        validation_result = {
            'Relationship': 'Energy-Macronutrients',
            'Description': 'energy ≈ proteins*4 + carbs*4 + fat*9',
            'Total Checked': mask_all_present.sum(),
            'Consistent': mask_all_present.sum() - inconsistent_energy,
            'Inconsistent': inconsistent_energy,
            'Consistency %': (1 - inconsistent_energy / mask_all_present.sum()) * 100 if mask_all_present.sum() > 0 else 0
        }
    
    # Step 2: Impute missing macronutrients when possible
    # Case: Missing protein
    mask_missing_protein = (~df['energy_100g'].isna()) & (df['proteins_100g'].isna()) & (~df['carbohydrates_100g'].isna()) & (~df['fat_100g'].isna())
    if mask_missing_protein.sum() > 0:
        protein_indices = df[mask_missing_protein].index
        energy_remaining = df_validated.loc[protein_indices, 'energy_100g'] - (
            df_validated.loc[protein_indices, 'carbohydrates_100g'] * 4 + 
            df_validated.loc[protein_indices, 'fat_100g'] * 9
        )
        # Calculate protein value (ensuring non-negative and reasonable values)
        df_validated.loc[protein_indices, 'proteins_100g'] = np.maximum(0, np.minimum(energy_remaining / 4, 90))
    
    # Case: Missing carbohydrates
    mask_missing_carbs = (~df['energy_100g'].isna()) & (~df['proteins_100g'].isna()) & (df['carbohydrates_100g'].isna()) & (~df['fat_100g'].isna())
    if mask_missing_carbs.sum() > 0:
        carbs_indices = df[mask_missing_carbs].index
        energy_remaining = df_validated.loc[carbs_indices, 'energy_100g'] - (
            df_validated.loc[carbs_indices, 'proteins_100g'] * 4 + 
            df_validated.loc[carbs_indices, 'fat_100g'] * 9
        )
        # Calculate carbs value (ensuring non-negative and reasonable values)
        df_validated.loc[carbs_indices, 'carbohydrates_100g'] = np.maximum(0, np.minimum(energy_remaining / 4, 95))
    
    # Case: Missing fat
    mask_missing_fat = (~df['energy_100g'].isna()) & (~df['proteins_100g'].isna()) & (~df['carbohydrates_100g'].isna()) & (df['fat_100g'].isna())
    if mask_missing_fat.sum() > 0:
        fat_indices = df[mask_missing_fat].index
        energy_remaining = df_validated.loc[fat_indices, 'energy_100g'] - (
            df_validated.loc[fat_indices, 'proteins_100g'] * 4 + 
            df_validated.loc[fat_indices, 'carbohydrates_100g'] * 4
        )
        # Calculate fat value (ensuring non-negative and reasonable values)
        df_validated.loc[fat_indices, 'fat_100g'] = np.maximum(0, np.minimum(energy_remaining / 9, 95))
    
    # Add imputation counts to the validation result
    if validation_result:
        validation_result['Proteins Imputed'] = mask_missing_protein.sum()
        validation_result['Carbohydrates Imputed'] = mask_missing_carbs.sum()
        validation_result['Fat Imputed'] = mask_missing_fat.sum()
    
    return validation_result, df_validated

# MAIN FUNCTIONS
def validate_nutritional_relationships(df):
    """
    Validate and cross-check related variables in the nutritional dataset.
    
    Args:
        df: DataFrame containing the nutritional data
        
    Returns:
        tuple: (validation_summary_df, df_validated)
    """
    # Create a copy to avoid modifying the original
    df_validated = df.copy()
    
    # Initialize a list to store validation results
    validation_results = []
    
    # Check sodium-salt relationship
    sodium_salt_result, df_validated = check_sodium_salt_relationship(df_validated)
    if sodium_salt_result:
        validation_results.append(sodium_salt_result)
    
    # Check energy-macronutrients relationship
    energy_macro_result, df_validated = check_energy_macronutrients_relationship(df_validated)
    if energy_macro_result:
        validation_results.append(energy_macro_result)
    
    # Create summary dataframe
    validation_summary = pd.DataFrame(validation_results)
    
    return validation_summary, df_validated

def create_validation_dashboard(df):
    """
    Create a validation dashboard for the dataset.
    
    Args:
        df: DataFrame containing the nutritional data
        
    Returns:
        tuple: (validation_summary_df, df_validated)
    """
    # Run validation
    validation_summary, df_validated = validate_nutritional_relationships(df)
    
    # Return just the validation summary and validated data
    return validation_summary, df_validated