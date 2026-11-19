# Hong Kong Housing Data Analysis

Group project for SDSC2102 Statistical Methods & Data Analysis at City University of Hong Kong. Course grade: A.

## Overview

This project investigates property-price drivers in Hong Kong using regression and classification. The workflow covers data cleaning, exploratory analysis, feature engineering, multicollinearity checks, model comparison, diagnostics, feature importance, and hypothesis testing.

## Dataset

The analysis reads the public HKProp dataset directly from:

https://raw.githubusercontent.com/just4jc/HKProp/master/HKProp_Dataset.csv

The source contains 1,139 property transactions from 2016–2018 with 28 attributes. The original course presentation identifies Centaline Property as the underlying source. No dataset copy is redistributed here.

## Methods

- Regression: Linear Regression and Gaussian Process Regression.
- Classification: Decision Tree and Random Forest for Low, Medium, and High price tiers.
- Preprocessing: missing-value handling, currency conversion, outlier filtering, categorical encoding, and standardisation.
- Evaluation: train/test split, five-fold cross-validation, R², RMSE, MAE, accuracy, macro precision/recall/F1, residual analysis, and confusion matrices.

## Reproduce

```bash
python -m pip install -r requirements.txt
python src/housing_analysis.py
```

The file uses `# %%` cell markers, so it can also be opened interactively in VS Code or another percent-script-compatible IDE. Gaussian Process cross-validation may take several minutes.

## Project documentation

- `docs/project-summary.md`: objectives, reported results, limitations, and future work.
- `src/housing_analysis.py`: cleaned analysis code with personal identifiers and stored notebook outputs removed.

## Notes

The reported metrics come from the original group submission. Results should be interpreted in the context of the small, historical dataset and should not be treated as current property-valuation advice.
