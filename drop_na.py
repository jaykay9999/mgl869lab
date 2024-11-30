import pandas as pd

# Load the combined CSV file
df_combined = pd.read_csv('combined_file_metrics_with_buggy_final_new.csv')

# Define the threshold: remove columns with more than 190,000 NaN values
threshold = 50000

# Drop columns that have NaN values above the threshold
df_combined = df_combined.loc[:, df_combined.isna().sum() < threshold]

# Save the cleaned DataFrame to a new CSV file
df_combined.to_csv('combined_file_metrics_with_buggy_final_new_drop_columns.csv', index=False)
print("Columns with more than 190,000 NaN values removed and data saved to combined_file_metrics_final_cleaned.csv")
