# Cholesterol Level Regression and Heart Disease Prediction

This repository contains two machine learning analyses using the UCI Heart Disease dataset:
1. **Cholesterol Level Regression** - A regression task using ElasticNet to predict cholesterol levels.
2. **Heart Disease Prediction** - A classification task using Logistic Regression and k-Nearest Neighbors (k-NN) to predict the presence of heart disease.

## Table of Contents
1. [Cholesterol Level Regression](#cholesterol-level-regression)
    - ElasticNet Regression
2. [Heart Disease Prediction](#heart-disease-prediction)
    - Logistic Regression
    - k-NN Classifier
3. [Installation Instructions](#installation-instructions)
4. [Usage Instructions](#usage-instructions)

---

## Cholesterol Level Regression

### Overview
In this section, we use ElasticNet regression to predict cholesterol levels (`chol`) based on the other features in the dataset.

### Code Explanation
1. **Data Loading and Preprocessing:**
   - The dataset is loaded using `pandas`, and missing values in the features are imputed with the mean of each column.
   - The features are scaled using `StandardScaler`.

2. **ElasticNet Regression:**
   - The ElasticNet model is trained with varying values of the hyperparameters: `alpha` (regularization strength) and `l1_ratio` (balance between Lasso and Ridge regularization).
   - A grid search is performed over a range of `alpha` and `l1_ratio` values.
   - The model's performance is evaluated using R² (coefficient of determination) and RMSE (Root Mean Squared Error).

3. **Plotting:**
   - Heatmaps of **R²** and **RMSE** are plotted to visualize the effect of different hyperparameter combinations on model performance.

4. **Best Configuration Extraction:**
   - The best combination of `alpha` and `l1_ratio` for both R² and RMSE is extracted and printed.

---

## Heart Disease Prediction

### Overview
In this section, we use a UCI Heart Disease dataset to predict the presence or absence of heart disease. Two machine learning models are implemented: **Logistic Regression** and **k-NN (k-Nearest Neighbors)**.

### Code Explanation
1. **Data Loading and Preprocessing:**
   - The dataset is loaded using `pandas`, and the target variable (`num`) is transformed into binary format: `1` for presence of heart disease and `0` for absence.
   - Missing values are handled by imputation using the median of each column.
   - Features are scaled using `StandardScaler`.

2. **Modeling:**
   - Logistic Regression and k-NN models are trained and evaluated using a grid search to tune hyperparameters such as regularization strength for Logistic Regression and number of neighbors and distance metric for k-NN.
   - Performance metrics such as accuracy, F1-score, ROC AUC score, and average precision score are computed.
   
3. **Plotting:**
   - The **AUROC** (Area Under the Receiver Operating Characteristic Curve) and **AUPRC** (Area Under the Precision-Recall Curve) are plotted for both models.

---

## Installation Instructions

1. **Clone the repository** to your local machine:
    ```bash
    git clone <repository-url>
    ```

2. **Install required dependencies:**
   You can install the necessary libraries using `pip` or `conda`. Here is the list of required packages:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   
   Using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

---

## Usage Instructions

1. **Prepare the dataset:**
   Ensure that the dataset `heart_disease_uci.csv` is placed at the specified path or update the path in the code.

2. **Run the cholesterol level regression model:**
   Execute the code for ElasticNet regression to tune the hyperparameters and visualize the performance using heatmaps for R² and RMSE.

3. **Run the heart disease prediction model:**
   Execute the code for Logistic Regression and k-NN. The notebook will automatically train the models and generate the evaluation metrics and plots.

4. **Examine the outputs:**
   - The models will print the evaluation metrics and display the ROC and Precision-Recall curves.
   - The cholesterol regression model will print the best configurations for R² and RMSE, and the corresponding heatmaps will be displayed.

---

## Example Output

### Cholesterol Level Regression:


Best R² Configuration: alpha = 0.01, l1_ratio = 0.4, R² = 0.76
Best RMSE Configuration: alpha = 0.1, l1_ratio = 0.6, RMSE = 22.56

### Heart Disease Prediction 

Logistic Regression Best Parameters: {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}
Logistic Regression Accuracy: 0.85
Logistic Regression F1 Score: 0.84
Logistic Regression ROC AUC Score: 0.89
Logistic Regression Average Precision Score: 0.88

k-NN Best Parameters: {'n_neighbors': 5, 'metric': 'euclidean'}
k-NN Accuracy: 0.82
k-NN F1 Score: 0.80
k-NN ROC AUC Score: 0.86
k-NN Average Precision Score: 0.83
