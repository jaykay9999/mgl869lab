import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
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
feature_logs = []

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

    # Use TPOT for feature selection and hyperparameter optimization
    tpot = TPOTClassifier(
        generations=5,  # Number of generations for evolution
        population_size=20,  # Number of individuals in each generation
        verbosity=2,
        random_state=42,
        scoring='roc_auc',
        cv=5  # 5-fold cross-validation
    )
    
    tpot.fit(X_train, y_train)
    
    # Extract the best pipeline
    best_pipeline = tpot.fitted_pipeline_
    
    # Get feature importance if available
    if hasattr(best_pipeline[-1], "feature_importances_"):  # For tree-based models
        importances = best_pipeline[-1].feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        print(f"Feature importance for {train_version} -> {test_version}:\n", feature_importance)
        feature_logs.append({"train_version": train_version, "test_version": test_version, "features": feature_importance})
    else:
        print("Feature importance not available for this pipeline.")
    
    # Predict on the test set
    y_pred = tpot.predict(X_test)
    y_pred_proba = tpot.predict_proba(X_test)[:, 1]  # Probability for class 1
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Save metrics
    overall_metrics.append({
        "train_version": train_version,
        "test_version": test_version,
        "AUC": auc
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
df_metrics.to_csv('random_forest_with_genetic_algorithm.csv', index=False)

print("\nOverall Metrics:")
print(df_metrics)

# Save feature importance logs to a CSV file
for log in feature_logs:
    log["features"].to_csv(f'feature_importance_{log["train_version"]}_to_{log["test_version"]}.csv', index=False)
