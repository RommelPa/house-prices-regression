# Executive Summary — House Prices Regression

## 1. Objective

This project builds and evaluates regression models to estimate residential property sale prices based on structural, location, quality, and condition-related features.

The goal is not only to predict prices, but also to evaluate model stability, analyze prediction errors, and interpret the property characteristics associated with price.

## 2. Business Context

Residential property pricing depends on multiple factors, including location, living area, construction quality, condition, garage capacity, basement quality, and sale characteristics.

A regression model can support pricing decisions, identify value drivers, and flag properties where automated estimates may require manual review.

The model should be used as a decision-support tool, not as a fully automatic pricing authority.

## 3. Dataset Scope

The project uses the Kaggle House Prices dataset.

The training dataset contains:

| Dataset | Rows | Columns |
|---|---:|---:|
| Training data | 1,460 | 81 |
| Kaggle test data | 1,459 | 80 |

The target variable is `SalePrice`.

Because `SalePrice` is right-skewed, the model was trained using a log-transformed target:

```text
SalePriceLog = log1p(SalePrice)
```

Predictions were converted back to the original price scale for evaluation.

## 4. Methodology

The project follows a reproducible machine learning workflow:

1. Load raw training and test data.
2. Audit missing values, feature types, target distribution, and correlations.
3. Create train and validation splits.
4. Handle missing values using feature-specific rules.
5. Encode categorical variables with one-hot encoding.
6. Train multiple regression models.
7. Compare models using validation metrics and cross-validation.
8. Select a final predictive model.
9. Analyze prediction errors.
10. Interpret feature associations using Ridge coefficients.

## 5. Models Compared

The following models were evaluated:

| Model | Purpose |
|---|---|
| Median baseline | Minimum benchmark |
| Linear Regression | Simple interpretable model |
| Ridge Regression | Regularized linear model |
| Lasso Regression | Sparse regularized linear model |
| Random Forest Regressor | Nonlinear ensemble model |

## 6. Key Results

### 6.1 Single validation split

On the initial validation split, Linear Regression achieved the lowest error:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 15,074.87 | 22,906.51 | 0.9316 |
| Ridge Regression | 16,434.82 | 25,110.06 | 0.9178 |
| Lasso Regression | 16,328.30 | 25,122.65 | 0.9177 |
| Random Forest | 17,349.97 | 29,673.74 | 0.8852 |
| Median Baseline | 59,568.25 | 88,667.17 | -0.0250 |

However, relying only on one validation split would be weak because the dataset is small and contains many one-hot encoded categorical variables.

### 6.2 Cross-validation changed the model decision

Cross-validation showed that Linear Regression was unstable across folds.

Random Forest achieved the best average RMSE and better stability:

| Model | Mean RMSE | RMSE Std | Mean R² |
|---|---:|---:|---:|
| Random Forest | 30,537.30 | 6,119.20 | 0.8508 |
| Ridge Regression | 43,337.81 | 36,656.64 | 0.5981 |
| Lasso Regression | 45,883.63 | 42,185.14 | 0.5240 |
| Linear Regression | 52,687.83 | 49,190.82 | 0.3637 |
| Median Baseline | 81,317.92 | 5,639.73 | -0.0565 |

## 7. Final Model Decision

The final predictive model is **Random Forest**.

This decision is based on cross-validation stability, not only on the best result from a single validation split.

Ridge Regression is used separately for interpretation because its regularized coefficients are easier to explain than Random Forest internal structure.

## 8. Final Model Error Analysis

The final Random Forest model achieved the following validation performance:

| Metric | Value |
|---|---:|
| MAE | 17,349.97 |
| RMSE | 29,673.74 |
| R² | 0.8852 |
| Median absolute error | 9,980.23 |
| Mean absolute percentage error | 10.19% |
| Median absolute percentage error | 6.29% |

The difference between MAE and RMSE indicates that a relatively small number of large prediction errors increases the overall error.

The largest errors are concentrated in higher-value or atypical properties.

## 9. Feature Interpretation

Ridge Regression coefficients suggest that the following factors are positively associated with sale price:

- Premium neighborhoods such as StoneBr, Crawfor, NridgHt, and NoRidge.
- Overall quality.
- Above-ground living area.
- Excellent basement quality.
- Garage capacity.
- Excellent kitchen quality.
- Normal functionality.

Negative associations include:

- Lower-value neighborhoods.
- Abnormal sale conditions.
- Certain zoning categories.
- Functional issues.
- Less desirable property characteristics.

These coefficients should be interpreted as associations, not causal effects. Some effects may be influenced by rare categories and should be treated as directional signals rather than definitive business rules.

## 10. Business Recommendations

1. Use the model as a pricing support tool, not as an automatic valuation system.
2. Review high-error predictions manually, especially for luxury or atypical properties.
3. Use Random Forest for predictive performance and Ridge coefficients for stakeholder communication.
4. Combine model outputs with local market knowledge before making pricing decisions.
5. Improve future models with location granularity, renovation details, market timing, comparable sales, and macroeconomic indicators.

## 11. Limitations

- The dataset is small for a high-dimensional regression problem.
- The Kaggle test set does not include true sale prices, so validation relies on train/validation split and cross-validation.
- One-hot encoded categorical variables can make linear models unstable.
- The model does not include exact geographic coordinates, school district quality, interest rates, local demand, or comparable sales.
- Model interpretation is associative, not causal.
- Rare categorical levels may distort some coefficient-based interpretations.

## 12. Next Steps

- Tune Random Forest hyperparameters.
- Compare with Gradient Boosting or XGBoost.
- Add feature engineering for house age, remodeling age, and total living area.
- Build a prediction API in a future deployment project.
- Create a dashboard to explain pricing drivers to business users.