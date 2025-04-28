# COPD Exacerbation Risk Prediction

## Project Description

This project implements a machine learning pipeline to predict the risk of COPD exacerbation based on a synthetic dataset. The pipeline includes data loading, exploratory data analysis (EDA), data preprocessing, training and hyperparameter tuning of multiple classification models, model evaluation, and feature importance analysis.

## Setup and Installation

To set up and run this project, you need to have Python installed. It is recommended to use a virtual environment.

1.  **Clone the repository (if applicable):**

    ```bash
    # If this project is in a repository, provide clone instructions here.
    # Example: git clone <repository_url>
    # cd <repository_directory>
    ```

    _(Note: Assuming the user is already in the project directory /Users/akshaypulla/Desktop/COPD)_

2.  **Install dependencies:**
    Navigate to the project directory and install the required libraries using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

## Usage

1.  **Ensure data is available:**
    Place the synthetic dataset file (`copd_synthetic_dataset.csv`) in the project's root directory.

2.  **Run the analysis script:**
    Execute the main Python script from the terminal:

    ```bash
    python copd_analysis.py
    ```

    The script will perform the following steps:

    - Load the data.
    - Conduct Exploratory Data Analysis (EDA) and save plots to the `output_models` directory.
    - Preprocess the data (handling missing values, scaling, encoding).
    - Train and tune Logistic Regression, Random Forest, and Gradient Boosting classifiers using GridSearchCV.
    - Evaluate the tuned models on a test set.
    - Select the best performing model based on the weighted F1 score.
    - Save the fitted preprocessor and the best performing model to the `output_models` directory.
    - Generate a `copd_analysis_report.md` file in the `output_models` directory summarizing the process and results.

## Features

- **Data Loading:** Reads the COPD synthetic dataset from a CSV file.
- **Exploratory Data Analysis (EDA):**
  - Generates summary statistics.
  - Identifies missing values.
  - Visualizes target variable distribution.
  - Creates histograms for numerical features.
  - Generates countplots for categorical features.
  - Produces boxplots showing numerical features vs. the target variable.
  - Generates a correlation heatmap for numerical features.
  - Saves all plots to the `output_models` directory.
- **Data Preprocessing:**
  - Splits data into training and testing sets.
  - Uses a ColumnTransformer with pipelines for numerical and categorical features.
  - Applies KNN Imputer and StandardScaler for numerical features.
  - Applies Simple Imputer (most frequent) and OneHotEncoder for categorical features.
  - Saves the fitted preprocessor.
- **Model Training and Tuning:**
  - Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
  - Performs hyperparameter tuning using GridSearchCV with cross-validation.
  - Uses weighted F1 score as the primary evaluation metric during tuning.
  - Saves the best tuned model for each algorithm.
- **Model Evaluation:**
  - Evaluates the best tuned models on the held-out test set.
  - Calculates and reports Accuracy, Precision, Recall, F1 Score (weighted), and ROC AUC (weighted OvR).
  - Prints classification reports and confusion matrices.
- **Model Comparison and Selection:**
  - Compares model performance based on test set metrics and CV scores.
  - Selects the best performing model based on the weighted F1 score on the test set.
  - Saves the final selected model.
- **Feature Importance:**
  - Analyzes and reports feature importances for tree-based models (Random Forest and Gradient Boosting).
  - Identifies the top features influencing the model's predictions.
- **Reporting:**
  - Generates a detailed markdown report (`copd_analysis_report.md`) summarizing the entire process, including EDA findings, preprocessing steps, tuning results, test set evaluation metrics, model comparison, and feature importance.

## Output

The script will create an `output_models` directory (if it doesn't exist) and save the following files:

- `target_distribution.png`: Plot of the target variable distribution.
- `numerical_distributions.png`: Histograms of numerical features.
- `categorical_distributions.png`: Countplots of categorical features (if any).
- `numerical_vs_target_boxplots.png`: Boxplots of numerical features vs. target.
- `correlation_heatmap.png`: Correlation matrix of numerical features.
- `preprocessor.pkl`: The fitted data preprocessor object.
- `LogisticRegression_best_tuned.pkl`: The best tuned Logistic Regression model.
- `RandomForestClassifier_best_tuned.pkl`: The best tuned Random Forest model.
- `GradientBoostingClassifier_best_tuned.pkl`: The best tuned Gradient Boosting model.
- `final_selected_model.pkl`: The overall best performing model on the test set.
- `GradientBoostingClassifier_feature_importance.png`: Feature importance plot for Gradient Boosting (if selected as best).
- `copd_analysis_report.md`: A markdown report detailing the analysis process and results.

## Data

The project uses a synthetic dataset (`copd_synthetic_dataset.csv`) which is expected to be in the root directory. The dataset should contain features relevant to COPD and a target variable indicating exacerbation risk.

## Contributing

_(Add information on how to contribute if this is an open-source project)_

## License

_(Add license information if applicable)_

## Acknowledgements

_(Add acknowledgements if needed)_
