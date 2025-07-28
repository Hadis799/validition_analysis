# =============================================================================
# Validation Analysis Script
#
# This script validates the output of the fuzzy expert system against data
# collected from a standard questionnaire (e.g., Felder-Solomon).
# Key Steps:
# 1. Loads the system's results and the questionnaire's results.
# 2. Merges the two datasets based on student names.
# 3. Uses a fuzzy categorization method to determine the dominant learning
#    style pole from the system's numerical output.
# 4. Uses a crisp (midpoint) categorization for the questionnaire data.
# 5. Compares the categories for each student and each dimension to calculate
#    the classification accuracy.
# 6. Saves the detailed comparison to an Excel file.
# =============================================================================

import pandas as pd
import numpy as np
import skfuzzy as fuzz

# --- Step 1: Define File Paths ---
system_results_file = 'system_results.xlsx'
questionnaire_results_file = 'questionnaire_results.xlsx'
output_file = 'Final_Comparison_Results.xlsx'

print("--- Starting Final Validation Analysis (with Fuzzy Comparison) ---")

try:
    # --- Step 2: Read and Merge Data Files ---
    df_system = pd.read_excel(system_results_file)
    df_questionnaire = pd.read_excel(questionnaire_results_file)
    print("Excel files read successfully.")

    # Create a standardized merge key by cleaning up student names
    df_system['merge_key'] = df_system['student_name'].str.replace(' ', '_').str.strip()
    df_questionnaire['merge_key'] = df_questionnaire['student_name'].str.replace(' ', '_').str.strip()
    print("Merge key created.")

    # Merge the two dataframes to find common students
    df_merged = pd.merge(df_system, df_questionnaire, on='merge_key', how='inner', suffixes=('_system', '_questionnaire'))
    
    if df_merged.empty:
        print("\nWarning: No common students found between the two files.")
        exit()
    print(f"{len(df_merged)} common students found for comparison.")

    # --- Step 3: Define Fuzzy Membership Functions for Categorization ---
    # These generic MFs are used to interpret the numerical output (0-11)
    # from the fuzzy system into a preference for one of two poles.
    output_mf_defs = {
        'Pole1_Pure': [0, 0, 1, 3],         # Strong preference for Pole 1
        'Pole1_Leaning': [2, 3, 4, 6],      # Leaning towards Pole 1
        'Pole2_Leaning': [5, 6, 7, 9],      # Leaning towards Pole 2
        'Pole2_Pure': [8, 9, 11, 11]        # Strong preference for Pole 2
    }
    universe = np.arange(0, 11.01, 0.01)

    # --- Step 4: Define Categorization Functions ---
    def get_fuzzy_dominant_pole(score):
        """
        Determines the dominant pole preference based on the highest total membership degree.
        This provides a more nuanced categorization than a simple midpoint split.
        """
        # Calculate membership degree for Pole 1 (Pure + Leaning)
        mem_pole1_pure = fuzz.interp_membership(universe, fuzz.trapmf(universe, output_mf_defs['Pole1_Pure']), score)
        mem_pole1_leaning = fuzz.interp_membership(universe, fuzz.trapmf(universe, output_mf_defs['Pole1_Leaning']), score)
        total_mem_pole1 = mem_pole1_pure + mem_pole1_leaning

        # Calculate membership degree for Pole 2 (Pure + Leaning)
        mem_pole2_pure = fuzz.interp_membership(universe, fuzz.trapmf(universe, output_mf_defs['Pole2_Pure']), score)
        mem_pole2_leaning = fuzz.interp_membership(universe, fuzz.trapmf(universe, output_mf_defs['Pole2_Leaning']), score)
        total_mem_pole2 = mem_pole2_pure + mem_pole2_leaning

        if total_mem_pole1 > total_mem_pole2:
            return "Pole_1_Tendency"
        elif total_mem_pole2 > total_mem_pole1:
            return "Pole_2_Tendency"
        else:
            return "Undetermined" # Case where membership degrees are equal

    def get_questionnaire_category(score):
        """Categorizes questionnaire scores using a crisp midpoint threshold."""
        if score < 5.5:
            return "Pole_1_Tendency"
        else:
            return "Pole_2_Tendency"

    # Map the dimension names to the column base names in the files
    dimensions_map = {
        'Processing': 'style_score_Processing_Style',
        'Perception': 'style_score_Perception_Style',
        'Input': 'style_score_Input_Modality_Style',
        'Understanding': 'style_score_Understanding_Style'
    }

    # --- Step 5: Perform the Comparison Loop ---
    comparison_results = []
    for index, row in df_merged.iterrows():
        student_name = row['student_name_system']
        for dim_name, col_base_name in dimensions_map.items():
            # Get the scores from the merged dataframe
            system_score = row[col_base_name + '_system']
            questionnaire_score = row[col_base_name + '_questionnaire']
            
            # Categorize both scores
            system_category = get_fuzzy_dominant_pole(system_score)
            questionnaire_category = get_questionnaire_category(questionnaire_score)
            
            # Check if the categories match
            match = '✅ Yes' if system_category == questionnaire_category else '❌ No'
            
            # Append the detailed result for this student and dimension
            comparison_results.append({
                'Student_Name': student_name,
                'Dimension': dim_name,
                'System_Score': round(system_score, 2),
                'Questionnaire_Score': round(questionnaire_score, 2),
                'System_Tendency': system_category,
                'Questionnaire_Tendency': questionnaire_category,
                'Match?': match
            })

    # Create the final comparison DataFrame
    df_final_comparison = pd.DataFrame(comparison_results)

    # --- Step 6: Calculate and Display Overall Accuracy ---
    print("\n--- Final Overall Agreement Report (based on Fuzzy Comparison) ---")
    for dim_name in dimensions_map.keys():
        subset = df_final_comparison[df_final_comparison['Dimension'] == dim_name]
        match_count = subset[subset['Match?'] == '✅ Yes'].shape[0]
        total_count = subset.shape[0]
        percentage = (match_count / total_count) * 100 if total_count > 0 else 0
        print(f"Dimension {dim_name}: {percentage:.2f}% agreement (out of {total_count} students)")

    # --- Step 7: Save the Final Results ---
    df_final_comparison.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\nAnalysis completed successfully. Full results saved to '{output_file}'.")

# --- Error Handling ---
except FileNotFoundError as e:
    print(f"\nError: The file {e.filename} was not found.")
except KeyError as e:
    print(f"\nError: The column '{e}' was not found in one of the files.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
