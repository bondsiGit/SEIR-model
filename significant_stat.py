import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_1samp
import tkinter as tk
from tkinter import messagebox

def calculate_statistics(actual_infectious, actual_recovered, predicted_infectious, predicted_recovered):
    # Calculate residuals
    infectious_residuals = actual_infectious - predicted_infectious
    recovered_residuals = actual_recovered - predicted_recovered

    # Mean Squared Error (MSE)
    mse_infectious = mean_squared_error(actual_infectious, predicted_infectious)
    mse_recovered = mean_squared_error(actual_recovered, predicted_recovered)

    # R-squared
    r2_infectious = r2_score(actual_infectious, predicted_infectious)
    r2_recovered = r2_score(actual_recovered, predicted_recovered)

    # T-test for residuals
    t_stat_infectious, p_value_infectious = ttest_1samp(infectious_residuals, 0)
    t_stat_recovered, p_value_recovered = ttest_1samp(recovered_residuals, 0)

    # # Print results
    print("=== Statistical Significance Test ===")
    print(f"MSE (Infectious): {mse_infectious:.2f}")
    print(f"MSE (Recovered): {mse_recovered:.2f}")
    print(f"R-squared (Infectious): {r2_infectious:.2f}")
    print(f"R-squared (Recovered): {r2_recovered:.2f}")
    print(f"T-test (Infectious): t-stat = {t_stat_infectious:.2f}, p-value = {p_value_infectious:.2f}")
    print(f"T-test (Recovered): t-stat = {t_stat_recovered:.2f}, p-value = {p_value_recovered:.2f}")
    
    return {
        "mse_infectious": mse_infectious,
        "mse_recovered": mse_recovered,
        "r2_infectious": r2_infectious,
        "r2_recovered": r2_recovered,
        "t_stat_infectious": t_stat_infectious,
        "p_value_infectious": p_value_infectious,
        "t_stat_recovered": t_stat_recovered,
        "p_value_recovered": p_value_recovered,
    }

# Example usage
if __name__ == "__main__":
    # Load actual data from Excel
    data = pd.read_excel('PMK Karanganyar.xlsx')
    actual_infectious = data['total_sakit']
    actual_recovered = data['total_sembuh']

    # Load predicted data from SEIR model (replace with actual predictions)
    predicted_infectious = np.load('final_infectious_pred.npy')  # Replace with your SEIR model predictions
    predicted_recovered = np.load('final_recovered_pred.npy')  # Replace with your SEIR model predictions

    # Calculate statistics
    calculate_statistics(actual_infectious, actual_recovered, predicted_infectious, predicted_recovered)

