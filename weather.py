import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, norm
import matplotlib.pyplot as plt

# Set working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the CSV file
try:
    df = pd.read_csv('wetter.csv')
except FileNotFoundError:
    print("Error: The file 'wetter.csv' was not found in the current directory.")
    sys.exit(1)

# Parse the dates and create a 'month' column
failed_dates = 0
dates = []
months = []

for date_str in df.iloc[:, 0]:
    try:
        date_obj = pd.to_datetime(date_str, format='%Y-%m-%d')
        dates.append(date_obj)
        months.append(date_obj.month)
    except ValueError:
        print(f"Warning: Failed to parse date '{date_str}'. Using NaT instead.")
        dates.append(pd.NaT)
        months.append(None)
        failed_dates += 1

df['date'] = dates
df['month'] = months

# Report failed date parses
if failed_dates > 0:
    print(f"Warning: {failed_dates} dates failed to parse.")

# Get the temperature column (3rd column, index 2)
temperatures = df.iloc[:, 2]

# Compute statistics for all data
overall_avg_temp = temperatures.mean()
overall_median_temp = temperatures.median()
# Use pandas mode instead of scipy's mode
overall_mode_temp = temperatures.mode().iloc[0]
overall_std_temp = temperatures.std()
overall_min_temp = temperatures.min()
overall_max_temp = temperatures.max()
overall_q1_temp = temperatures.quantile(0.25)
overall_q3_temp = temperatures.quantile(0.75)
overall_iqr_temp = overall_q3_temp - overall_q1_temp

# Statistics for July
july_temps = temperatures[df['month'] == 7]
july_avg_temp = july_temps.mean()
july_median_temp = july_temps.median()
july_mode_temp = july_temps.mode().iloc[0] if not july_temps.empty else np.nan
july_std_temp = july_temps.std()
july_min_temp = july_temps.min() if not july_temps.empty else np.nan
july_max_temp = july_temps.max() if not july_temps.empty else np.nan
july_q1_temp = july_temps.quantile(0.25) if not july_temps.empty else np.nan
july_q3_temp = july_temps.quantile(0.75) if not july_temps.empty else np.nan
july_iqr_temp = july_q3_temp - july_q1_temp if not july_temps.empty else np.nan

# Statistics for May
may_temps = temperatures[df['month'] == 5]
may_avg_temp = may_temps.mean()
may_median_temp = may_temps.median()
may_mode_temp = may_temps.mode().iloc[0] if not may_temps.empty else np.nan
may_std_temp = may_temps.std()
may_min_temp = may_temps.min() if not may_temps.empty else np.nan
may_max_temp = may_temps.max() if not may_temps.empty else np.nan
may_q1_temp = may_temps.quantile(0.25) if not may_temps.empty else np.nan
may_q3_temp = may_temps.quantile(0.75) if not may_temps.empty else np.nan
may_iqr_temp = may_q3_temp - may_q1_temp if not may_temps.empty else np.nan

# Perform the t-test
if not july_temps.empty and not may_temps.empty:
    t_stat, p_value = ttest_ind(july_temps, may_temps, equal_var=False)
else:
    t_stat, p_value = np.nan, np.nan

# Print results - overall
print("===== Overall Temperature Statistics =====")
print(f"Mean: {overall_avg_temp:.2f}")
print(f"Median: {overall_median_temp:.2f}")
print(f"Mode: {overall_mode_temp:.2f}")
print(f"Standard Deviation: {overall_std_temp:.2f}")
print(f"Minimum: {overall_min_temp:.2f}")
print(f"Maximum: {overall_max_temp:.2f}")
print(f"Q1 (25th Percentile): {overall_q1_temp:.2f}")
print(f"Q3 (75th Percentile): {overall_q3_temp:.2f}")
print(f"IQR: {overall_iqr_temp:.2f}")
print()

# Print results - July
print("===== July Temperature Statistics =====")
if not july_temps.empty:
    print(f"Mean: {july_avg_temp:.2f}")
    print(f"Median: {july_median_temp:.2f}")
    print(f"Mode: {july_mode_temp:.2f}")
    print(f"Standard Deviation: {july_std_temp:.2f}")
    print(f"Minimum: {july_min_temp:.2f}")
    print(f"Maximum: {july_max_temp:.2f}")
    print(f"Q1 (25th Percentile): {july_q1_temp:.2f}")
    print(f"Q3 (75th Percentile): {july_q3_temp:.2f}")
    print(f"IQR: {july_iqr_temp:.2f}")
else:
    print("No data available for July")
print()

# Print results - May
print("===== May Temperature Statistics =====")
if not may_temps.empty:
    print(f"Mean: {may_avg_temp:.2f}")
    print(f"Median: {may_median_temp:.2f}")
    print(f"Mode: {may_mode_temp:.2f}")
    print(f"Standard Deviation: {may_std_temp:.2f}")
    print(f"Minimum: {may_min_temp:.2f}")
    print(f"Maximum: {may_max_temp:.2f}")
    print(f"Q1 (25th Percentile): {may_q1_temp:.2f}")
    print(f"Q3 (75th Percentile): {may_q3_temp:.2f}")
    print(f"IQR: {may_iqr_temp:.2f}")
else:
    print("No data available for May")
print()

# Print t-test results
print("===== T-Test Results =====")
if not np.isnan(t_stat) and not np.isnan(p_value):
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.6f}")

    # Conclusion about statistical significance
    if p_value < 0.05:
        print("Conclusion: The difference between July and May temperatures is statistically significant at the 0.05 level.")
    else:
        print("Conclusion: The difference between July and May temperatures is not statistically significant at the 0.05 level.")
else:
    print("Cannot perform t-test: insufficient data for July and/or May")

# Create histograms with normal distribution curve only if we have data
plt.figure(figsize=(18, 6))

# Overall temperatures histogram
if not temperatures.empty:
    plt.subplot(1, 3, 1)
    plt.hist(temperatures, bins=20, alpha=0.7, color='blue', density=True)
    plt.title('Overall Temperature Distribution')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')

    # Add normal distribution curve
    x = np.linspace(min(temperatures), max(temperatures), 100)
    plt.plot(x, norm.pdf(x, overall_avg_temp, overall_std_temp), 'r-', lw=2)

# July temperatures histogram
if not july_temps.empty:
    plt.subplot(1, 3, 2)
    plt.hist(july_temps, bins=min(15, len(july_temps)//2 + 1), alpha=0.7, color='red', density=True)
    plt.title('July Temperature Distribution')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')

    # Add normal distribution curve
    x = np.linspace(min(july_temps), max(july_temps), 100)
    plt.plot(x, norm.pdf(x, july_avg_temp, july_std_temp), 'b-', lw=2)

# May temperatures histogram
if not may_temps.empty:
    plt.subplot(1, 3, 3)
    plt.hist(may_temps, bins=min(15, len(may_temps)//2 + 1), alpha=0.7, color='green', density=True)
    plt.title('May Temperature Distribution')
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')

    # Add normal distribution curve
    x = np.linspace(min(may_temps), max(may_temps), 100)
    plt.plot(x, norm.pdf(x, may_avg_temp, may_std_temp), 'b-', lw=2)

plt.tight_layout()
plt.savefig('temperature_histograms.png')
plt.show()
