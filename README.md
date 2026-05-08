# House Prices Regression: Predicting Property Prices with Interpretable Machine Learning

## Overview

This project builds and evaluates regression models to predict house prices using structured property data.

The goal is not only to obtain accurate predictions, but also to understand which property characteristics influence price, evaluate model errors, and communicate results in a business-oriented way.

## Business Context

Real estate pricing depends on multiple factors such as location, size, quality, age, condition, and amenities.

A regression model can help estimate property values, support pricing decisions, identify value drivers, and detect cases where the model performs poorly.

## Objectives

- Audit and understand the dataset.
- Prepare numerical and categorical features.
- Handle missing values.
- Build baseline and machine learning regression models.
- Compare Linear Regression, Ridge, Lasso, and Random Forest.
- Evaluate models using MAE, RMSE, and R².
- Analyze residuals and large prediction errors.
- Interpret the most relevant features.
- Generate business recommendations.

## Project Structure

```text
house-prices-regression/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_regression_modeling.ipynb
├── src/
│   ├── load_data.py
│   ├── audit_data.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   └── evaluate_models.py
├── reports/
│   ├── executive_summary_en.md
│   ├── resumen_ejecutivo_es.md
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

## Status

Project in progress.