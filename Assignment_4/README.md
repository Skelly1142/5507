# Survival Analysis and Machine Learning Models for Clinical Data

## Overview
This Python script performs survival analysis on clinical data using **Kaplan-Meier estimation**, **Cox Proportional Hazards models**, and **Random Survival Forests**. The goal is to analyze the impact of different treatment modalities on patient survival and evaluate the predictive power of statistical and machine learning models.

## Table of Contents
1. [Kaplan-Meier Survival Analysis](#kaplan-meier-survival-analysis)
    - Survival Curve Estimation
2. [Comparing Treatment Groups](#comparing-treatment-groups)
    - Log-Rank Test
3. [Cox Proportional Hazards Model](#cox-proportional-hazards-model)
4. [Random Survival Forest Model](#random-survival-forest-model)
5. [Feature Importance Analysis](#feature-importance-analysis)
6. [Installation Instructions](#installation-instructions)
7. [Usage Instructions](#usage-instructions)
8. [Example Output](#example-output)

---

## Kaplan-Meier Survival Analysis

### Overview
Kaplan-Meier estimation is used to compute and visualize survival probabilities over time for different treatment groups.

### Process
1. **Data Preparation**:
   - Load the clinical dataset.
   - Convert relevant columns (e.g., **radiotherapy start date, last follow-up, and date of death**) into a standardized datetime format.
   - Compute **survival time** for each patient.

2. **Kaplan-Meier Curve Estimation**:
   - Group patients by treatment type (**ChemoRT, RT alone, RT + EGFRI**).
   - Compute and plot **Kaplan-Meier survival curves** for each group.

---

## Comparing Treatment Groups

### Overview
The **Log-Rank Test** is used to determine if survival distributions differ significantly between treatment groups.

### Process
- Conduct **pairwise comparisons** of survival curves between:
  - **ChemoRT vs. RT alone**
  - **ChemoRT vs. RT + EGFRI**
  - **RT alone vs. RT + EGFRI**
- Output **p-values** to assess statistical significance.

---

## Cox Proportional Hazards Model

### Overview
The **Cox Proportional Hazards Model** is used to assess the impact of covariates (e.g., age, sex, treatment) on survival.

### Process
1. **Preprocessing**:
   - One-hot encode categorical variables (e.g., **sex**).
   - Standardize numerical variables (e.g., **age, dose**).

2. **Model Fitting**:
   - Fit the **Cox model** using **penalized regression (L1 regularization)**.
   - Display **hazard ratios** for each covariate.

3. **Proportional Hazards Assumption**:
   - Validate the model using a **Proportional Hazards Assumption test**.

---

## Random Survival Forest Model

### Overview
A **Random Survival Forest** (RSF) model is trained to predict survival outcomes based on patient features.

### Process
1. **Data Preprocessing**:
   - Encode categorical variables and normalize numerical features.
   - Transform survival data into a structured format.

2. **Model Training**:
   - Train an **RSF model** with **100 decision trees**.
   - Use a **training-validation split** (80%-20%).

3. **Evaluation**:
   - Compute the **Concordance Index (C-index)** to measure predictive accuracy.

---

## Feature Importance Analysis

### Overview
Permutation-based feature importance is used to determine which variables most influence survival predictions.

### Process
- Compute **feature importance scores** for **Random Survival Forest**.
- Plot **bar charts** showing key predictive features.

---

## Installation Instructions

1. **Install required dependencies**:
    ```
    pip install pandas numpy matplotlib lifelines scikit-survival scikit-learn openpyxl
    ```
---

## Usage Instructions

### Prepare the dataset:
- Ensure that the clinical dataset is placed in the correct directory.
- Update file paths in the script if necessary.

### Run the Kaplan-Meier analysis:
- Execute the script to generate **survival curves for different treatment groups**.

### Run the log-rank test:
- The script will compare treatment groups and **output statistical significance (p-values).**

### Train the Cox Proportional Hazards Model:
- The model will fit patient data and **output hazard ratios and proportional hazards test results.**

### Train the Random Survival Forest Model:
- The script will train the model and **compute the C-index for performance evaluation.**

### Analyze feature importance:
- The script will generate a **feature importance plot.**

---
