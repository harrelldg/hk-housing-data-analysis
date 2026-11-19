# Project Summary

## Research questions

1. Which property features most strongly predict sale price?
2. Does Gaussian Process Regression outperform Linear Regression?
3. Can tree-based classifiers predict Low, Medium, and High price categories?

## Approach

The team cleaned 1,139 transactions, removed outliers and multicollinear variables, engineered distance to the city centre, and compared regression and classification models. Saleable area, building age, and location were central variables.

## Reported findings

- Saleable area was the strongest positive driver; building age was the strongest negative driver.
- The final presentation reported Gaussian Process test R² of 0.789 versus 0.658 for Linear Regression, with RMSE of 106.4 versus 141.1.
- Random Forest and Decision Tree achieved 80.6% and 75.4% test accuracy respectively, above the 33.3% three-class random baseline.

These figures are preserved from the submitted presentation rather than newly benchmarked during portfolio preparation.

## Limitations

- Transactions are limited to 2016–2018 and do not represent current market conditions.
- Renovation quality, view, and other price-relevant attributes are absent.
- Outlier removal reduced the sample from 1,139 to 1,014 records.
- Cross-validation comparisons based on a small number of folds should be interpreted cautiously.

## Future work

Add interest-rate and policy variables, test time-aware validation and additional models, and evaluate performance on newer transactions.

## Team

Group 3: CLAUDIA Marchelyn, GUNAWAN Harrell Dave, RAHARDJO Marvel Reswanda, and WU Frederico Owen.
