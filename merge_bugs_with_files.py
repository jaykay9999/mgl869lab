import pandas as pd

# Load CSV files
df_jira_bugs = pd.read_csv('jira_bugs_new.csv')
df_bug_files = pd.read_csv('bug_files_new.csv')
df_releases = pd.read_csv('cleaned_last_commits_per_release.csv')

# Ensure 'fix_versions' is properly evaluated as a list of versions
df_jira_bugs['affected_versions'] = df_jira_bugs['affected_versions'].apply(eval)

# Merge bug files with fix versions from jira_bugs.csv
merged_data = df_bug_files.merge(df_jira_bugs, on='bug_id', how='left')

# Filter for only Java and C++ files
merged_data = merged_data[merged_data['file'].str.endswith(('.java', '.cpp'))]

# Expand each bug file to have separate rows for each fix version
expanded_data = merged_data.explode('affected_versions')
expanded_data = expanded_data.rename(columns={'affected_versions': 'version'})

# Merge with release data to add commit IDs based on version
final_data = expanded_data.merge(df_releases[['version', 'commit_id']], on='version', how='left')


#final_data = final_data.dropna()
# Save to a new CSV file
final_data.to_csv('bug_files_with_versions_and_commits_new.csv', index=False)
print("Data saved to bug_files_with_versions_and_commits.csv")
