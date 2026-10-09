# Breast Cancer Diagnosis: End-to-End Data Science Project

Final project for the **Virtual Data Science with Python Apprentice Internship** (YuvaIntern / Henry Harvin Education).

## Objective
Classify breast masses as malignant or benign from 30 cell-nuclei features, and turn the results into clear, risk-aware recommendations.

## Dataset
Wisconsin Diagnostic Breast Cancer (WDBC): 569 records, 30 numeric features (bundled with scikit-learn, no download needed).

## What the code does
1. Data quality checks: missing values, duplicates, IQR outliers, skewness
2. EDA: class balance, feature distributions, correlation heatmap
3. Modelling: Logistic Regression, Random Forest, Gradient Boosting (stratified 75/25 split, 5-fold CV)
4. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, permutation importance
5. Threshold analysis to favour recall on malignant cases

## Key results
- Best model: Logistic Regression (CV ROC-AUC 0.991, test accuracy 96.5%)
- Lowering the threshold from 0.5 to 0.3 raises malignant recall from 92.5% to 96.2%

## Run
```bash
pip install -r requirements.txt
python analysis.py
```
Charts are saved to `charts/` and metrics to `results.json`.

## Files
- `analysis.py`: full analysis pipeline
- `charts/`: generated figures used in the report
- `results.json`: all computed metrics

## Author
Yugank Charnotia
