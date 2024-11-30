import pandas as pd
import re

# Load the generated CSV file
df = pd.read_csv('bug_files_with_versions_and_commits_new.csv')

# Drop the 'bug_id' column
df = df.drop(columns=['bug_id'])

df = df.dropna()

# Standardize the 'version' column to keep only "number.number.number" format
def clean_version(version):
    # Use regex to capture only the major.minor.patch format
    print("version is : " , version)
    match = re.search(r'\b\d+\.\d+\.\d+\b', version)
    return match.group(0) if match else version

df['version'] = df['version'].apply(clean_version)

# Remove rows where version is defined as "Not Applicable" or similar
df = df[~df['version'].str.contains("Not Applicable", case=False, na=False)]

# Convert 'version' to numeric for filtering, capturing only the major version
df['major_version'] = df['version'].apply(lambda x: int(x.split('.')[0]) if re.match(r'^\d+\.\d+\.\d+$', x) else None)

# Filter to keep only rows with major version >= 2
df = df[df['major_version'] >= 2]

# Drop the temporary 'major_version' column used for filtering
df = df.drop(columns=['major_version'])

# Drop duplicate rows
df = df.drop_duplicates()

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_bug_files_with_versions_and_commits_new.csv', index=False)
print("Cleaned data saved to cleaned_bug_files_with_versions_and_commits.csv")
