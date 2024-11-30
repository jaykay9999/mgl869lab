import pandas as pd

# Load the data
df = pd.read_csv('combined_file_metrics_with_buggy_final_new_drop_columns.csv')

# Function to calculate unique value counts for each column
def print_column_value_counts(df):
    for column in df.columns:
        print(f"\nColumn: {column}")
        print(df[column].value_counts(dropna=False))

# Check numeric column preprocessing: Replace ',' with '.' and convert to float
for column in df.columns:
    if df[column].dtype == object:  # Check if the column contains strings
        try:
            df[column] = df[column].str.replace(',', '.').astype(float)
        except ValueError:
            pass  # Skip columns that can't be converted

# Print unique value counts for each column
print_column_value_counts(df)
