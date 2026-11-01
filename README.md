# Yuvaintern Week 2: Supervised Machine Learning Models

## Objective
Train and compare supervised regression and classification models using Scikit-learn.

## Algorithms
- Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor, KNN Regressor.
- Classification: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, KNN Classifier.

## Datasets
- Scikit-learn Diabetes dataset for regression.
- Breast Cancer Wisconsin dataset (built into Scikit-learn) for binary classification.
No manual dataset download is required.

## Run in VS Code on Windows
1. Extract the ZIP and open this folder in VS Code.
2. Open Terminal > New Terminal.
3. Run `python -m venv .venv`
4. Run `.venv\\Scripts\\activate`
5. Run `pip install -r requirements.txt`
6. Run `python train_models.py`

## Metrics
Regression: MAE, RMSE, R-squared. Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC.

## Outputs
CSV metric comparisons, a text classification report, confusion matrix, and comparison charts are saved in `outputs/`.

## Reproducibility
A fixed random seed (42) is used. Classification uses a stratified train/test split. StandardScaler is fitted inside Pipelines for Logistic Regression and KNN to avoid test-data leakage.
