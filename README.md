# Titanic Survival Prediction using Decision Tree

This Project contains the implementation of a **Decision Tree Classifier** to predict passenger survival on the Titanic. This project includes data exploration, preprocessing, model training, and submission to the Kaggle competition.

---

##  Project Overview

The objective of this project is to build a predictive model using passengers data. A `DecisionTreeClassifier` is configured with controlled complexity to handle class imbalance and prevent overfitting. The dataset is sourced from the famous [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data).

---

## Pipeline Workflow

### 1. Data Loading
- Loading Datasets as Pandas DataFrames from the local directory

### 2. Exploratory Data Analysis
- Visualizing survival distributions using `seaborn.countplot`
- Visualizing feature correlations using `seaborn.heatmap`

### 3. Data Preprocessing
- Droping unused features to eliminate noise and prevent overfitting
- Encoding categorical features using One-Hot Encoding (`pd.get_dummies`)
- Handling missing values using median imputation for numerical data

### 4. Model Training
- Training and initializing a Scikit-Learn `DecisionTreeClassifier` with hyperparameter tuning to ensure robust generalization
- Visualizing feature importances to interpret model decisions
- Predicting the labels (`0` for deceased, `1` for survived)

### 5. Model Evaluation
- Accuracy score
- Confusion matrix
- Recall score
- precision score
- f1 score

### 6. Inference
- Making predictions on the unseen test set and generating a Kaggle-compliant submission file exported as `titanic_submission.csv`.

---

## Project Structure

```
├── src/
│   └── Decision_Tree.ipynb
├── submission/
│   |── submission_result.png
│   └── titanic_submission.csv
├── visualizations/
│   |── countplot_survival_rate.png
│   |── heatmap_correlation_matrix.png
│   └── feature_importances.png
├── .gitignore
└── README.md
```
