import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
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

# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be a leaf node
    'random_state': [42]  # Set random state for reproducibility
}

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
    
    # Feature selection: Select top K features based on ANOVA F-statistic
    k = min(5, X_train.shape[1])  # Select top 5 features or less if fewer columns
    selector = SelectKBest(score_func=f_classif, k=k)

    # Fit the selector and transform the training data
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get the names of the selected features
    selected_feature_names = X_train.columns[selector.get_support()]
    print(f"Top {k} selected features for training version {train_version}: {list(selected_feature_names)}")

    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        scoring='roc_auc',  # Optimize for AUC
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1  # Use all available CPU cores
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_train_selected, y_train)
    
    # Best model from grid search
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Predict on the test set
    y_pred = best_model.predict(X_test_selected)
    y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]  # Probability for class 1
    
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
df_metrics.to_csv('random_forest_model_performance_with_tuning.csv', index=False)

print("\nOverall Metrics:")
print(df_metrics)
