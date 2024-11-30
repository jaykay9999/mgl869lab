import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('combined_file_metrics_with_buggy_final_new_drop_columns.csv')

# Drop unnecessary columns except 'version' (used for filtering train/test sets)
columns_to_drop = ['Kind', 'Name', 'File', 'Entity_Uniquename', 'commit_id', 'CCViolDensityCode', 'CCViolDensityLine', 'RatioCommentToCode']
df = df.drop(columns=columns_to_drop)

# Drop rows with NaN values
df = df.dropna()

# Define the train-test pairs based on the order of versions
train_test_pairs = [
    ("2.0.0", "2.1.0"),
    ("2.1.0", "2.2.0"),
    ("2.2.0", "2.3.0"),
    ("2.3.0", "3.0.0"),
    ("3.0.0", "3.1.0"),
    ("3.1.0", "4.0.0"),
]

# Store overall metrics
overall_metrics = []

# Iterate through the train-test pairs
for train_version, test_version in train_test_pairs:
    print(f"\nTraining on version {train_version}, testing on version {test_version}")
    
    # Split the data into training and testing based on version
    train_data = df[df['version'] == train_version]
    test_data = df[df['version'] == test_version]
    
    # Debugging: Check if datasets are empty
    if train_data.empty:
        print(f"Training data is empty for version {train_version}. Skipping...")
        continue
    if test_data.empty:
        print(f"Testing data is empty for version {test_version}. Skipping...")
        continue
    
    # Drop 'version' column right before training
    X_train = train_data.drop(columns=['buggy', 'version'])
    y_train = train_data['buggy']
    X_test = test_data.drop(columns=['buggy', 'version'])
    y_test = test_data['buggy']
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for class 1
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Save metrics
    overall_metrics.append({
        "train_version": train_version,
        "test_version": test_version,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })
    
    # Print the classification report
    print(classification_report(y_test, y_pred))
    
    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{train_version} -> {test_version} (AUC = {auc:.2f})')

# Finalize and display the ROC curve
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid()
plt.show()

# Create a DataFrame for overall metrics
df_metrics = pd.DataFrame(overall_metrics)

# Save overall metrics to a CSV file
df_metrics.to_csv('random_forest_model_performance_metrics.csv', index=False)

print("\nOverall Metrics:")
print(df_metrics)
