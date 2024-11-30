import pandas as pd

# Load the combined metrics data and bug file data
df_metrics = pd.read_csv('combined_processed_data_new.csv')
df_bugs = pd.read_csv('cleaned_bug_files_with_versions_and_commits_new.csv')

# Remove "/hive/" from the start of cleaned_file_path if it exists
df_metrics['cleaned_file_path'] = df_metrics['cleaned_file_path'].str.replace(r'^/hive/', '', regex=True)

#print(df_metrics['cleaned_file_path'].head(5))

# Create a set of tuples (file, version) from the bug files for quick lookup
buggy_files = set(zip(df_bugs['file'], df_bugs['version']))

# Add a 'buggy' column to the metrics DataFrame based on matching the cleaned file path and version
df_metrics['buggy'] = df_metrics.apply(
    lambda row: 1 if (row['cleaned_file_path'], row['version']) in buggy_files else 0, axis=1
)

# Drop the temporary cleaned_file_path column
df_metrics = df_metrics.drop(columns=['cleaned_file_path'])

# Save the updated DataFrame to a new CSV
df_metrics.to_csv('combined_file_metrics_with_buggy_final_new.csv', index=False)
print("Updated metrics data with 'buggy' column saved to combined_file_metrics_with_buggy_final.csv")
