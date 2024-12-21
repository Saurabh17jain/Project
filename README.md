# Fuel Cell Performance Prediction Using PyCaret

# Project Overview

This project focuses on predicting the performance of fuel cells using machine learning models. The workflow includes preprocessing the dataset, selecting the target variable based on a roll number mapping, splitting the dataset into training and testing sets, and comparing multiple regression models using PyCaret.

Dataset

The dataset contains various performance metrics for fuel cells. Each row represents a data point with specific features and target variables.

Dataset Requirements

Ensure the dataset is in CSV format.

The dataset must include multiple target columns: Target1, Target2, Target3, Target4, and Target5.

Ensure there are no missing values in the dataset, or handle them during preprocessing.

Workflow Steps

Step 1: Load Dataset

The dataset is loaded from a local file path using pandas.

import pandas as pd

# Load the dataset
dataset_path = "C:\\Users\\khura\\OneDrive\\Desktop\\MLLabEval\\Fuel_cell_performance_data-Full.csv"
df = pd.read_csv(dataset_path)

Step 2: Handle Missing Values

Missing values are handled by dropping rows with null values:

if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.dropna()

Step 3: Select Target Based on Roll Number

The last digit of the roll number determines the target column to be used for predictions.

roll_number = "102203677"
last_digit = int(roll_number[-1])

# Map roll number ending to target
target_map = {
    0: "Target1", 5: "Target1",
    1: "Target2", 6: "Target2",
    2: "Target3", 7: "Target3",
    3: "Target4", 8: "Target4",
    4: "Target5", 9: "Target5"
}
selected_target = target_map[last_digit]
print(f"\nSelected Target: {selected_target}")

Step 4: Split Dataset (70/30)

The dataset is split into training and testing sets using train_test_split.

from sklearn.model_selection import train_test_split

# Prepare features and target
X = df.drop(columns=[selected_target])
y = df[selected_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Step 5: Use PyCaret for Model Comparison

The dataset is processed and multiple regression models are compared using PyCaret.

from pycaret.regression import setup, compare_models

# Combine target and features for PyCaret
df_pycaret = pd.concat([y, X], axis=1)

# Initialize PyCaret setup
s = setup(data=df_pycaret, target=selected_target,
          remove_outliers=True, outliers_threshold=0.05,
          normalize=True, normalize_method='zscore',
          transformation=True, transformation_method='yeo-johnson',
          data_split_shuffle=False, verbose=False)

# Compare models
cm = compare_models()

Key Features

Automatic Target Selection: The target column is determined dynamically based on the roll number.

Data Preprocessing: Includes handling missing values, outlier removal, normalization, and transformation.

Model Comparison: Leverages PyCaret’s compare_models to find the best-performing regression model.

Requirements

Python 3.7+

Required Libraries:

pandas

scikit-learn

pycaret

Install dependencies using:

pip install pandas scikit-learn pycaret

How to Run

Ensure the dataset is available at the specified file path.

Replace the roll number in the script with your actual roll number.

Run the script to preprocess the data, split the dataset, and compare models.

View the output to determine the best-performing model.

# Outputs

Best regression model based on PyCaret’s comparison.

Metrics like Mean Squared Error (MSE) and R-squared for each model.

# Notes

Customize preprocessing steps (e.g., handling missing values) as needed.

Ensure the target column exists in the dataset before running the script.

You can adjust PyCaret setup parameters (e.g., normalization and transformation methods) to optimize performance.

# Authors

Developed by [Saurabh kr jain].

