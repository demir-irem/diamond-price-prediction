# Diamond Price Prediction

Predicting the price of ~54,000 diamonds from carat, cut, color, clarity and dimensions. The focus is on how **outliers** and **feature scaling** change the performance of Linear Regression and kNN.

## What I did
- **EDA:** price against each feature and correlations between numeric features
- **Outlier detection with IQR** for carat, depth, table, x, y and z, then removal, with the original data kept for comparison
- **Ordinal encoding** for cut, color and clarity (keeping their natural order)
- **Manual shuffle and 70/30 split** (no `train_test_split` / `shuffle`)
- **Linear Regression** on the original vs. outlier-removed data, plus **5-fold cross-validation** (R² and RMSE, mean ± std)
- **kNN regression** for k = 1–10, then repeated with standardized features

## Results

| Model | Data | Test MSE | R² |
|---|---|---|---|
| Linear Regression | original | 1.40 M | — |
| Linear Regression | outliers removed | 1.08 M | — |
| kNN (k = 9) | outliers removed | 0.52 M | 0.954 |
| **kNN (k = 7), standardized** | outliers removed | **0.35 M** | **0.970** |

- Removing outliers lowered the error and made cross-validation results more stable (R² std 0.003 → 0.001).
- kNN beat Linear Regression, and scaling improved it further because kNN is distance-based.

## Tech
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## Data
The `diamonds` dataset that ships with seaborn (`sns.load_dataset("diamonds")`). No download needed.

## Run
```bash
pip install -r requirements.txt
jupyter notebook diamond_price_prediction.ipynb
```

*Developed as part of a data science course (AIN214, Hacettepe University, Fall 2025).*
