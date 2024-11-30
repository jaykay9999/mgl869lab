import subprocess
import os
import os.path as osp
import pandas as pd


import pandas as pd

# Load the data
df_bugs = pd.read_csv("combined_file_metrics_with_buggy_final_new_drop_columns.csv")

# Print the counts of each version and buggy status
print("Overall version counts:")
print(df_bugs['version'].value_counts())

print("\nOverall buggy counts:")
print(df_bugs['buggy'].value_counts())

# Calculate the number of NaN values in each column
nan_counts = df_bugs.isna().sum()

# Print the column names and their corresponding NaN counts
print("\nNaN counts per column:")
for column, count in nan_counts.items():
    print(f"{column}: {count} NaN values")

# Calculate the number of buggy and non-buggy values for each version
buggy_counts_per_version = df_bugs.groupby(['version', 'buggy']).size().unstack(fill_value=0)

# Print the buggy and non-buggy counts for each version
print("\nBuggy and non-buggy counts per version:")
print(buggy_counts_per_version)
