# COPD Exacerbation Risk Prediction Report

## 1. Introduction
This report details the process of building machine learning models to predict the 'exacerbation_risk' using the dataset 'copd_synthetic_dataset.csv'.
The primary evaluation metric is 'f1_weighted'.

## 2. Data Loading and Overview
- Data loaded from: copd_synthetic_dataset.csv
- Initial shape: (100000, 17)

## 3. Exploratory Data Analysis (EDA)
- Conducted summary statistics, missing value analysis, and visualizations (histograms, countplots, boxplots, correlation heatmap).
- Target variable distribution plot saved to: output_models/target_distribution.png
- Numerical feature distributions saved to: output_models/numerical_distributions.png
- Numerical vs. Target boxplots saved to: output_models/numerical_vs_target_boxplots.png
- Correlation heatmap saved to: output_models/correlation_heatmap.png
- Key findings from EDA should be summarized here manually after reviewing plots (e.g., significant correlations, class imbalance, feature distributions).

## 4. Data Preprocessing
- Split data: 80% train / 20% test (random_state=42, stratified).
### Preprocessing Steps:
  - **Numerical Features:**
    - Missing values imputed using KNNImputer (n_neighbors=5).
    - Features scaled using StandardScaler.
  - **Categorical Features:**
    - No categorical features identified for preprocessing.
- Preprocessing pipeline saved to: output_models/preprocessor.pkl

## 5. Model Training and Hyperparameter Tuning
- Models trained: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Hyperparameter tuning performed using GridSearchCV (CV=5, scoring='f1_weighted').
### Tuning Results (Best CV Scores):
  - LogisticRegression: 0.8587 (f1_weighted)
    - Best Params: {'C': 100, 'solver': 'saga'}
    - Tuned model saved to: output_models/LogisticRegression_best_tuned.pkl
  - RandomForestClassifier: 0.9918 (f1_weighted)
    - Best Params: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
    - Tuned model saved to: output_models/RandomForestClassifier_best_tuned.pkl
  - GradientBoostingClassifier: 0.9952 (f1_weighted)
    - Best Params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 1.0}
    - Tuned model saved to: output_models/GradientBoostingClassifier_best_tuned.pkl

## 6. Model Evaluation on Test Set
Evaluating tuned models on the held-out test set:

### LogisticRegression
  **Test Set Performance:**
    - Accuracy: 0.8393
    - Precision (Weighted): 0.9092
    - Recall (Weighted): 0.8393
    - F1 Score (Weighted): 0.8595
    - ROC AUC (Weighted OvR): 0.9271
  **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.44      0.88      0.59      2612
           1       0.98      0.83      0.90     17388

    accuracy                           0.84     20000
   macro avg       0.71      0.86      0.74     20000
weighted avg       0.91      0.84      0.86     20000

```
  **Confusion Matrix:**
```
[[ 2307   305]
 [ 2909 14479]]
```

### RandomForestClassifier
  **Test Set Performance:**
    - Accuracy: 0.9930
    - Precision (Weighted): 0.9930
    - Recall (Weighted): 0.9930
    - F1 Score (Weighted): 0.9930
    - ROC AUC (Weighted OvR): 0.9996
  **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.97      0.97      0.97      2612
           1       1.00      1.00      1.00     17388

    accuracy                           0.99     20000
   macro avg       0.98      0.98      0.98     20000
weighted avg       0.99      0.99      0.99     20000

```
  **Confusion Matrix:**
```
[[ 2541    71]
 [   68 17320]]
```

### GradientBoostingClassifier
  **Test Set Performance:**
    - Accuracy: 0.9969
    - Precision (Weighted): 0.9968
    - Recall (Weighted): 0.9969
    - F1 Score (Weighted): 0.9968
    - ROC AUC (Weighted OvR): 0.9999
  **Classification Report:**
```
              precision    recall  f1-score   support

           0       0.99      0.98      0.99      2612
           1       1.00      1.00      1.00     17388

    accuracy                           1.00     20000
   macro avg       0.99      0.99      0.99     20000
weighted avg       1.00      1.00      1.00     20000

```
  **Confusion Matrix:**
```
[[ 2570    42]
 [   21 17367]]
```

## 7. Model Comparison and Selection

### Performance Summary Table:
|                            |   F1 Score (Weighted) |   ROC AUC (Weighted OvR) |   Accuracy |   Precision (Weighted) |   Recall (Weighted) |   CV F1 Score (Weighted) |
|:---------------------------|----------------------:|-------------------------:|-----------:|-----------------------:|--------------------:|-------------------------:|
| LogisticRegression         |                0.8595 |                   0.9271 |     0.8393 |                 0.9092 |              0.8393 |                   0.8587 |
| RandomForestClassifier     |                0.993  |                   0.9996 |     0.993  |                 0.993  |              0.993  |                   0.9918 |
| GradientBoostingClassifier |                0.9968 |                   0.9999 |     0.9968 |                 0.9968 |              0.9968 |                   0.9952 |

### Final Model Selection:
- Based on the primary evaluation metric 'F1 Score (Weighted)' on the test set, the best performing model is **GradientBoostingClassifier**.
- Test Set F1 Score (Weighted): 0.9968
- CV f1_weighted: 0.9952
- The final selected model (GradientBoostingClassifier) has been saved to: output_models/final_selected_model.pkl

## 8. Feature Importance Analysis

Feature importances for the selected model (GradientBoostingClassifier):
| Feature            |   Importance |
|:-------------------|-------------:|
| num__FEV1          |  0.301004    |
| num__FVC           |  0.178394    |
| num__medications   |  0.177322    |
| num__SpO2          |  0.109946    |
| num__PM2.5         |  0.107807    |
| num__pollen        |  0.105731    |
| num__wheeze        |  0.0137156   |
| num__cough         |  0.00413657  |
| num__HRV           |  0.00170656  |
| num__NO2           |  0.000167995 |
| num__humidity      |  2.82918e-05 |
| num__activity      |  2.64979e-05 |
| num__sleep         |  1.44193e-05 |
| num__comorbidities |  1.0449e-06  |

Feature importance plot saved to: output_models/GradientBoostingClassifier_feature_importance.png

## 9. Conclusion and Future Work

**Conclusion:** The GradientBoostingClassifier model demonstrated the best performance on the test set for predicting 'exacerbation_risk', achieving a weighted F1-score of 0.9968. Key features influencing the predictions were identified (refer to feature importance section).

**Limitations:**
- The dataset is synthetic, which might not fully capture real-world complexities.
- The analysis assumes the relationships captured in the data are stable over time.
- Hyperparameter search space was limited for demonstration purposes.
- Feature engineering based on domain knowledge (e.g., from 'timestamp') was minimal.

**Future Work:**
- Validate models on real-world patient data.
- Explore more advanced feature engineering techniques (e.g., time-series features, interaction terms).
- Experiment with other algorithms (e.g., SVM, Neural Networks).
- Investigate ensemble methods combining predictions from multiple models.
- Conduct a more thorough hyperparameter optimization using RandomizedSearchCV or Bayesian optimization.
- Analyze model fairness and bias if sensitive attributes are present.
- Deploy the best model for real-time prediction (requires further infrastructure).