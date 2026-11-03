# Household Income Prediction

Python machine learning project predicting annual household income from
demographic, spending and asset data.

## What we did
- Engineered 4 new features (food ratio, age squared, crowding index, employment rate)
- Log-transformed the skewed target; handled missing occupation values (treated as MNAR)
- Compared 5 models with 5-fold cross-validation: Linear Regression, KNN,
  Random Forest, Decision Tree, Linear Regression + PCA
- Improved the best model (Random Forest): hyperparameter tuning, Stacking + Ridge, Bagging + OOB
- Explained the model with SHAP

## Results
- Final model: Bagging with Random Forest, test R² = 0.8775, OOB score = 0.8784
- Top predictors: miscellaneous goods and services, food, and housing/water expenditure

## Notebook
https://colab.research.google.com/drive/1PJ6WtiUyjwbI-A598dfa14TmGlr7xkqB?usp=sharing

## Data
"Filipino Family Income and Expenditure" dataset (41,544 households, 60 columns),
available on Kaggle: https://www.kaggle.com/datasets/grosvenpaul/family-income-and-expenditure



