import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('combined_file_metrics_with_buggy_final_new_drop_columns.csv')

# Drop unnecessary columns except 'version' (used for filtering train/test sets)
columns_to_drop = ['Kind', 'Name', 'File', 'Entity_Uniquename', 'commit_id', 'CCViolDensityCode', 'CCViolDensityLine', 'RatioCommentToCode']
df = df.drop(columns=columns_to_drop)

# Drop rows with NaN values
df = df.dropna()

# Use unique versions from train_test_pairs
train_test_pairs = [
    ("2.0.0", "2.1.0"),
    ("2.1.0", "2.2.0"),
    ("2.2.0", "2.3.0"),
    ("2.3.0", "3.0.0"),
    ("3.0.0", "3.1.0"),
    ("3.1.0", "4.0.0"),
]
unique_versions = sorted(set([pair[0] for pair in train_test_pairs] + [pair[1] for pair in train_test_pairs]))

# Store overall metrics
overall_metrics = []
combined_feature_importances = pd.DataFrame()

# Iterate through each specified version
for version in unique_versions:
    print(f"\nProcessing version: {version}")
    
    # Filter data for the current version
    version_data = df[df['version'] == version]
    
    # Skip if version data is empty
    if version_data.empty:
        print(f"Version {version} has no data. Skipping...")
        continue
    
    # Drop 'version' column
    X = version_data.drop(columns=['buggy', 'version'])
    y = version_data['buggy']
    
    # Define the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []
    precisions = []
    recalls = []
    f1_scores = []
    feature_importances = np.zeros(X.shape[1])  # Initialize feature importance array
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Accumulate feature importance
        feature_importances += model.feature_importances_
        
        # Predict probabilities and classes
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)
        
        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = classification_report(y_test, y_pred, output_dict=True)['1'].values()
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Normalize feature importance (average across folds)
    feature_importances /= skf.get_n_splits()
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances,
        "Version": version
    }).sort_values(by="Importance", ascending=False)
    
    # Append to the combined DataFrame
    combined_feature_importances = pd.concat([combined_feature_importances, feature_importance_df], ignore_index=True)
    
    # Store the average metrics for the version
    overall_metrics.append({
        "version": version,
        "Mean AUC": np.mean(auc_scores),
        "Mean Precision": np.mean(precisions),
        "Mean Recall": np.mean(recalls),
        "Mean F1-Score": np.mean(f1_scores)
    })
    
    # Plot the average ROC curve for this version
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, label=f'Version {version} (AUC = {np.mean(auc_scores):.2f})')


# Finalize and display the ROC curve
plt.title('Average ROC Curve (10-Fold Cross-Validation)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid()
plt.show()

# Create a DataFrame for overall metrics
df_metrics = pd.DataFrame(overall_metrics)

# Save overall metrics to a CSV file
df_metrics.to_csv('random_forest_performance_metrics.csv', index=False)

# Save combined feature importances to a single CSV file
combined_feature_importances.to_csv('random_forest_combined_feature_importances.csv', index=False)

print("\nOverall Metrics:")
print(df_metrics)
print("\nCombined Feature Importances saved to 'random_forest_combined_feature_importances.csv'")
