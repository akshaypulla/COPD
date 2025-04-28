import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import pickle
import os
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = 'copd_synthetic_dataset.csv'
TARGET_VARIABLE = 'exacerbation_risk'
OUTPUT_DIR = 'output_models'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PRIMARY_METRIC = 'f1_weighted' # Using weighted F1 for potentially imbalanced classes

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Data ---
print("1. Loading Data...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully from {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n2. Performing Exploratory Data Analysis (EDA)...")

# Convert timestamp to datetime objects (optional for EDA, but good practice)
# We might extract features later if needed, but for basic EDA, we'll exclude it.
if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Warning: Could not convert 'timestamp' column to datetime: {e}")

# Drop patient_id for EDA analysis (it's an identifier)
df_eda = df.drop(columns=['patient_id'], errors='ignore')

# Identify feature types
numerical_features = df_eda.select_dtypes(include=np.number).columns.tolist()
categorical_features = df_eda.select_dtypes(include='object').columns.tolist() # Assuming non-numeric are categorical for now
datetime_features = df_eda.select_dtypes(include=['datetime64[ns]']).columns.tolist()

# Remove target and datetime from numerical features list for EDA visualizations
if TARGET_VARIABLE in numerical_features:
    numerical_features.remove(TARGET_VARIABLE)
if datetime_features:
    for dt_feat in datetime_features:
        if dt_feat in numerical_features: # Should not happen based on select_dtypes, but safety check
             numerical_features.remove(dt_feat)

print(f"\nIdentified Numerical Features: {numerical_features}")
print(f"Identified Categorical Features: {categorical_features}")
print(f"Identified Datetime Features: {datetime_features}")
print(f"Target Variable: {TARGET_VARIABLE}")

# Summary Statistics
print("\nSummary Statistics (Numerical Features):")
print(df_eda[numerical_features + [TARGET_VARIABLE]].describe()) # Include target here

# Missing Values
print("\nMissing Values per Column:")
print(df_eda.isnull().sum())

# Target Variable Distribution
print(f"\nTarget Variable '{TARGET_VARIABLE}' Distribution:")
print(df_eda[TARGET_VARIABLE].value_counts())
print(df_eda[TARGET_VARIABLE].value_counts(normalize=True))
plt.figure(figsize=(6, 4))
sns.countplot(x=TARGET_VARIABLE, data=df_eda)
plt.title(f'Distribution of {TARGET_VARIABLE}')
plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
plt.close()
print(f"Saved target distribution plot to {OUTPUT_DIR}/target_distribution.png")


# Visualizations (saving plots to output directory)
print("\nGenerating EDA Visualizations...")

# Histograms for numerical features
print(" - Generating histograms for numerical features...")
num_plots = len(numerical_features)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols
plt.figure(figsize=(num_cols * 5, num_rows * 4))
for i, col in enumerate(numerical_features):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.histplot(df_eda[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'numerical_distributions.png'))
plt.close()
print(f"Saved numerical distribution plots to {OUTPUT_DIR}/numerical_distributions.png")

# Countplots for categorical features (if any) - Adjusting for potentially high cardinality
print(" - Generating countplots for categorical features...")
if categorical_features:
    num_cat_plots = len(categorical_features)
    num_cat_cols = 2
    num_cat_rows = (num_cat_plots + num_cat_cols - 1) // num_cat_cols
    plt.figure(figsize=(num_cat_cols * 6, num_cat_rows * 5))
    for i, col in enumerate(categorical_features):
        plt.subplot(num_cat_rows, num_cat_cols, i + 1)
        # Limit categories shown if too many
        top_n = 20
        if df_eda[col].nunique() > top_n:
            top_categories = df_eda[col].value_counts().nlargest(top_n).index
            sns.countplot(y=col, data=df_eda[df_eda[col].isin(top_categories)], order=top_categories)
            plt.title(f'Top {top_n} Categories of {col}')
        else:
            sns.countplot(y=col, data=df_eda, order=df_eda[col].value_counts().index)
            plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'categorical_distributions.png'))
    plt.close()
    print(f"Saved categorical distribution plots to {OUTPUT_DIR}/categorical_distributions.png")
else:
    print("   (No categorical features identified)")


# Boxplots: Numerical features vs. Target
print(" - Generating boxplots (numerical vs. target)...")
num_plots = len(numerical_features)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols
plt.figure(figsize=(num_cols * 5, num_rows * 4))
for i, col in enumerate(numerical_features):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(x=TARGET_VARIABLE, y=col, data=df_eda)
    plt.title(f'{col} vs. {TARGET_VARIABLE}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'numerical_vs_target_boxplots.png'))
plt.close()
print(f"Saved numerical vs. target boxplots to {OUTPUT_DIR}/numerical_vs_target_boxplots.png")

# Correlation Heatmap (Numerical Features only)
print(" - Generating correlation heatmap...")
plt.figure(figsize=(12, 10))
# Include target in correlation calculation
corr_matrix = df_eda[numerical_features + [TARGET_VARIABLE]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features (including Target)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
plt.close()
print(f"Saved correlation heatmap to {OUTPUT_DIR}/correlation_heatmap.png")

print("EDA complete.")

# --- 3. Data Preprocessing ---
print("\n3. Performing Data Preprocessing...")

# Define features (X) and target (y)
# Drop identifier, timestamp, and target variable from features
features_to_drop = [TARGET_VARIABLE, 'patient_id', 'timestamp']
X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
y = df[TARGET_VARIABLE]

# Split data into training and testing sets
print(f"Splitting data into training ({1-TEST_SIZE:.0%}) and testing ({TEST_SIZE:.0%}) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y # Stratify for classification
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Re-identify feature types based on the training set (X_train)
numerical_features_prep = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_prep = X_train.select_dtypes(include='object').columns.tolist()

print(f"\nFeatures for Preprocessing:")
print(f"  Numerical: {numerical_features_prep}")
print(f"  Categorical: {categorical_features_prep}")

# Create preprocessing pipelines for numerical and categorical features
# Numerical Pipeline: Impute missing values with KNN, then scale
numerical_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)), # Using KNN Imputer as requested
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Impute missing values with most frequent, then OneHotEncode
# Using SimpleImputer for categorical as KNN is less suitable
from sklearn.impute import SimpleImputer # Add import if not already present (it is)
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # handle_unknown='ignore' is important
])

# Create the ColumnTransformer to apply pipelines to correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features_prep),
        ('cat', categorical_pipeline, categorical_features_prep)
    ],
    remainder='passthrough' # Keep other columns (if any) - should be none here
)

# Apply preprocessing: Fit on training data, transform both train and test
print("\nApplying preprocessing (fitting on train, transforming train & test)...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after OneHotEncoding for potential use later (e.g., feature importance)
# Handle cases where there are no categorical features
try:
    feature_names_out = preprocessor.get_feature_names_out()
except AttributeError: # Older scikit-learn versions might not have get_feature_names_out
    # Manual reconstruction (simplified)
    feature_names_out = list(numerical_features_prep)
    if 'cat' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
         cat_features_encoded = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features_prep)
         feature_names_out.extend(cat_features_encoded)
    elif categorical_features_prep: # Fallback if get_feature_names_out is unavailable
        print("Warning: Could not automatically get feature names after encoding. Using generic names.")
        # This part might need adjustment based on the actual encoder output structure in older versions
        num_cat_features = X_train_processed.shape[1] - len(numerical_features_prep)
        feature_names_out.extend([f"cat_{i}" for i in range(num_cat_features)])


print(f"Shape after processing: X_train_processed={X_train_processed.shape}, X_test_processed={X_test_processed.shape}")
# print("Feature names after processing:", feature_names_out) # Optional: print feature names

# Save the fitted preprocessor
preprocessor_path = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
print(f"\nSaving the fitted preprocessor to {preprocessor_path}...")
with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor saved.")

# --- 4. Model Training and Hyperparameter Tuning ---
print("\n4. Training Models and Tuning Hyperparameters...")

# Define models and parameter grids
models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'), # Added class_weight
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'] # saga supports L1/L2 with large datasets
            # 'penalty': ['l1', 'l2'] # Solver 'liblinear' supports L1/L2, 'saga' supports L1/L2/elasticnet/none
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'), # Added class_weight
        'params': {
            'n_estimators': [100, 200], # Reduced for faster demo, expand if needed
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 3]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [100, 200], # Reduced for faster demo
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0] # Added subsample
        }
    }
}

# Store results and best models
model_results = {}
best_estimators = {}

# Perform GridSearch for each model
for model_name, config in models.items():
    print(f"\n--- Tuning {model_name} ---")
    model = config['model']
    params = config['params']

    # Note: Preprocessing is already done. We pass the processed data here.
    # If preprocessing were part of the pipeline, we'd define steps differently.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=CV_FOLDS,
        scoring=PRIMARY_METRIC,
        n_jobs=-1, # Use all available CPU cores
        verbose=1 # Show progress
    )

    print(f"Performing GridSearchCV for {model_name}...")
    grid_search.fit(X_train_processed, y_train)

    # Store results
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    model_results[model_name] = {'best_score_cv': best_score, 'best_params': best_params}
    best_estimators[model_name] = best_estimator

    print(f"Best {PRIMARY_METRIC} (CV): {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    # Save the best estimator from GridSearchCV
    model_filename = f"{model_name}_best_tuned.pkl"
    model_path = os.path.join(OUTPUT_DIR, model_filename)
    print(f"Saving best tuned {model_name} to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(best_estimator, f)
    print(f"{model_name} saved.")

print("\nModel training and hyperparameter tuning complete.")

# --- 5. Model Evaluation on Test Set ---
print("\n5. Evaluating Models on Test Set...")

test_results = {}
report_content = [] # Start building the report content

report_content.append("# COPD Exacerbation Risk Prediction Report\n")
report_content.append("## 1. Introduction")
report_content.append(f"This report details the process of building machine learning models to predict the '{TARGET_VARIABLE}' using the dataset '{DATA_PATH}'.")
report_content.append(f"The primary evaluation metric is '{PRIMARY_METRIC}'.")

report_content.append("\n## 2. Data Loading and Overview")
report_content.append(f"- Data loaded from: {DATA_PATH}")
report_content.append(f"- Initial shape: {df.shape}")
# Add more details from initial loading if needed

report_content.append("\n## 3. Exploratory Data Analysis (EDA)")
report_content.append("- Conducted summary statistics, missing value analysis, and visualizations (histograms, countplots, boxplots, correlation heatmap).")
report_content.append(f"- Target variable distribution plot saved to: {OUTPUT_DIR}/target_distribution.png")
report_content.append(f"- Numerical feature distributions saved to: {OUTPUT_DIR}/numerical_distributions.png")
if categorical_features:
    report_content.append(f"- Categorical feature distributions saved to: {OUTPUT_DIR}/categorical_distributions.png")
report_content.append(f"- Numerical vs. Target boxplots saved to: {OUTPUT_DIR}/numerical_vs_target_boxplots.png")
report_content.append(f"- Correlation heatmap saved to: {OUTPUT_DIR}/correlation_heatmap.png")
report_content.append("- Key findings from EDA should be summarized here manually after reviewing plots (e.g., significant correlations, class imbalance, feature distributions).") # Placeholder for manual summary

report_content.append("\n## 4. Data Preprocessing")
report_content.append(f"- Split data: {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test (random_state={RANDOM_STATE}, stratified).")
report_content.append("### Preprocessing Steps:")
report_content.append("  - **Numerical Features:**")
report_content.append("    - Missing values imputed using KNNImputer (n_neighbors=5).")
report_content.append("    - Features scaled using StandardScaler.")
report_content.append("  - **Categorical Features:**")
if categorical_features_prep:
    report_content.append("    - Missing values imputed using SimpleImputer (strategy='most_frequent').")
    report_content.append("    - Features encoded using OneHotEncoder (handle_unknown='ignore').")
else:
    report_content.append("    - No categorical features identified for preprocessing.")
report_content.append(f"- Preprocessing pipeline saved to: {preprocessor_path}")

report_content.append("\n## 5. Model Training and Hyperparameter Tuning")
report_content.append(f"- Models trained: {', '.join(models.keys())}")
report_content.append(f"- Hyperparameter tuning performed using GridSearchCV (CV={CV_FOLDS}, scoring='{PRIMARY_METRIC}').")
report_content.append("### Tuning Results (Best CV Scores):")
for model_name, result in model_results.items():
    report_content.append(f"  - {model_name}: {result['best_score_cv']:.4f} ({PRIMARY_METRIC})")
    report_content.append(f"    - Best Params: {result['best_params']}")
    model_filename = f"{model_name}_best_tuned.pkl"
    report_content.append(f"    - Tuned model saved to: {OUTPUT_DIR}/{model_filename}")


report_content.append("\n## 6. Model Evaluation on Test Set")
report_content.append("Evaluating tuned models on the held-out test set:")

evaluation_metrics = {}

for model_name, model in best_estimators.items():
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    y_pred = model.predict(X_test_processed)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_test_processed)
            # Adjust for binary vs multi-class ROC AUC
            if y_pred_proba.shape[1] == 2:
                y_pred_proba_auc = y_pred_proba[:, 1] # Probability of positive class
                roc_auc = roc_auc_score(y_test, y_pred_proba_auc)
            else:
                # Multi-class case: One-vs-Rest (OvR) or One-vs-One (OvO)
                # Using OvR with weighted averaging as it aligns with f1_weighted
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC for {model_name}: {e}")
            roc_auc = np.nan # Assign NaN if calculation fails
    else:
        roc_auc = np.nan # Model doesn't support predict_proba

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted') # Primary metric
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision (Weighted)': precision,
        'Recall (Weighted)': recall,
        'F1 Score (Weighted)': f1,
        'ROC AUC (Weighted OvR)': roc_auc if not np.isnan(roc_auc) else 'N/A'
    }
    evaluation_metrics[model_name] = metrics

    print(f"Test Set Metrics for {model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    # Add to report
    report_content.append(f"\n### {model_name}")
    report_content.append("  **Test Set Performance:**")
    for metric, value in metrics.items():
        report_content.append(f"    - {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"    - {metric}: {value}")
    report_content.append("  **Classification Report:**")
    report_content.append("```")
    report_content.append(classification_report(y_test, y_pred))
    report_content.append("```")
    report_content.append("  **Confusion Matrix:**")
    report_content.append("```")
    report_content.append(str(cm))
    report_content.append("```")

# --- 7. Model Comparison and Selection ---
print("\n7. Comparing Models and Selecting Best...")
report_content.append("\n## 7. Model Comparison and Selection")

# Create a DataFrame for comparison
comparison_df = pd.DataFrame(evaluation_metrics).T
comparison_df['CV F1 Score (Weighted)'] = [model_results[model]['best_score_cv'] for model in comparison_df.index]

# Reorder columns for clarity
metric_order = ['F1 Score (Weighted)', 'ROC AUC (Weighted OvR)', 'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'CV F1 Score (Weighted)']
comparison_df = comparison_df[[col for col in metric_order if col in comparison_df.columns]] # Handle missing ROC AUC

print("\nModel Performance Comparison (Test Set & CV):")
print(comparison_df.round(4))
report_content.append("\n### Performance Summary Table:")
report_content.append(comparison_df.round(4).to_markdown())

# Select best model based on the primary metric on the test set
primary_metric_col = 'F1 Score (Weighted)' # Align with PRIMARY_METRIC config if needed
best_model_name = comparison_df[primary_metric_col].idxmax()
best_model = best_estimators[best_model_name]

print(f"\nBest performing model based on Test Set '{primary_metric_col}': {best_model_name}")
report_content.append(f"\n### Final Model Selection:")
report_content.append(f"- Based on the primary evaluation metric '{primary_metric_col}' on the test set, the best performing model is **{best_model_name}**.")
report_content.append(f"- Test Set {primary_metric_col}: {comparison_df.loc[best_model_name, primary_metric_col]:.4f}")
report_content.append(f"- CV {PRIMARY_METRIC}: {model_results[best_model_name]['best_score_cv']:.4f}")


# Save the final selected model
final_model_filename = "final_selected_model.pkl"
final_model_path = os.path.join(OUTPUT_DIR, final_model_filename)
print(f"Saving final selected model ({best_model_name}) to {final_model_path}...")
with open(final_model_path, 'wb') as f:
    pickle.dump(best_model, f)
print("Final model saved.")
report_content.append(f"- The final selected model ({best_model_name}) has been saved to: {final_model_path}")

# --- 8. Feature Importance Analysis ---
print("\n8. Analyzing Feature Importance...")
report_content.append("\n## 8. Feature Importance Analysis")

# Check if the best model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    print(f"Calculating feature importances for {best_model_name}...")
    importances = best_model.feature_importances_
    # Use feature_names_out obtained after preprocessing
    feature_names = feature_names_out # Use the names derived earlier
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\nTop 15 Feature Importances:")
    print(feature_importance_df.head(15))

    report_content.append(f"\nFeature importances for the selected model ({best_model_name}):")
    report_content.append(feature_importance_df.head(15).to_markdown(index=False))


    # Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15)) # Plot top 15
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    importance_plot_path = os.path.join(OUTPUT_DIR, f'{best_model_name}_feature_importance.png')
    plt.savefig(importance_plot_path)
    plt.close()
    print(f"Saved feature importance plot to {importance_plot_path}")
    report_content.append(f"\nFeature importance plot saved to: {importance_plot_path}")

elif hasattr(best_model, 'coef_'): # For linear models like Logistic Regression
     print(f"Calculating feature coefficients for {best_model_name}...")
     # Handle potential multi-class coefficients
     if best_model.coef_.shape[0] > 1: # Multi-class
         print("  (Displaying coefficients for the first class vs rest)")
         coefficients = best_model.coef_[0]
     else: # Binary class
         coefficients = best_model.coef_.flatten()

     feature_names = feature_names_out # Use the names derived earlier
     feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
     # Sort by absolute value of coefficient
     feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
     feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)


     print("\nTop 15 Features by Absolute Coefficient:")
     print(feature_importance_df[['Feature', 'Coefficient']].head(15))

     report_content.append(f"\nFeature coefficients (magnitude indicates importance) for the selected model ({best_model_name}):")
     report_content.append(feature_importance_df[['Feature', 'Coefficient']].head(15).to_markdown(index=False))

     # Plot coefficients
     plt.figure(figsize=(10, 8))
     # Select top 15 by absolute value for plotting
     plot_df = feature_importance_df.head(15).sort_values(by='Coefficient', ascending=False)
     sns.barplot(x='Coefficient', y='Feature', data=plot_df)
     plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
     plt.tight_layout()
     importance_plot_path = os.path.join(OUTPUT_DIR, f'{best_model_name}_feature_coefficients.png')
     plt.savefig(importance_plot_path)
     plt.close()
     print(f"Saved feature coefficient plot to {importance_plot_path}")
     report_content.append(f"\nFeature coefficient plot saved to: {importance_plot_path}")

else:
    print(f"Feature importance/coefficients not directly available for {best_model_name}.")
    report_content.append(f"\nFeature importance/coefficients analysis is not directly applicable to the selected model ({best_model_name}). Permutation importance could be explored as an alternative.")


# --- 9. Conclusion and Future Work ---
print("\n9. Generating Final Report...")
report_content.append("\n## 9. Conclusion and Future Work")
report_content.append(f"\n**Conclusion:** The {best_model_name} model demonstrated the best performance on the test set for predicting '{TARGET_VARIABLE}', achieving a weighted F1-score of {comparison_df.loc[best_model_name, primary_metric_col]:.4f}. Key features influencing the predictions were identified (refer to feature importance section).")
report_content.append("\n**Limitations:**")
report_content.append("- The dataset is synthetic, which might not fully capture real-world complexities.")
report_content.append("- The analysis assumes the relationships captured in the data are stable over time.")
report_content.append("- Hyperparameter search space was limited for demonstration purposes.")
report_content.append("- Feature engineering based on domain knowledge (e.g., from 'timestamp') was minimal.")

report_content.append("\n**Future Work:**")
report_content.append("- Validate models on real-world patient data.")
report_content.append("- Explore more advanced feature engineering techniques (e.g., time-series features, interaction terms).")
report_content.append("- Experiment with other algorithms (e.g., SVM, Neural Networks).")
report_content.append("- Investigate ensemble methods combining predictions from multiple models.")
report_content.append("- Conduct a more thorough hyperparameter optimization using RandomizedSearchCV or Bayesian optimization.")
report_content.append("- Analyze model fairness and bias if sensitive attributes are present.")
report_content.append("- Deploy the best model for real-time prediction (requires further infrastructure).")

# --- 10. Save Report ---
report_path = os.path.join(OUTPUT_DIR, 'copd_analysis_report.md')
print(f"\nSaving analysis report to {report_path}...")
try:
    with open(report_path, 'w') as f:
        f.write("\n".join(report_content))
    print("Report saved successfully.")
except Exception as e:
    print(f"Error saving report: {e}")

print("\n--- Analysis Complete ---")