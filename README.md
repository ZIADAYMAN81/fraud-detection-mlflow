# Fraud Detection & Diabetes Prediction with MLflow Experiment Tracking

> Machine Learning coursework, The British University in Egypt, 2026.

A complete supervised-learning workflow tested on two very different problems:

| Dataset | Size | Positive class |
|---|---|---|
| Online payment transactions | ~300K rows | fraud ≈ **2.2%** (heavily imbalanced) |
| Pima Indians Diabetes | 768 rows | diabetic ≈ 35% |

Every experiment is logged in **MLflow** (parameters, metrics and the fitted pipeline), so all runs are compared on the same footing.

## Pipeline
1. **Data understanding.** Skewed transaction amounts were kept, because the extreme values are legitimate. Zeros in diabetes features that really mean "not measured" (e.g. blood pressure, BMI, insulin) were treated as missing.
2. **Leakage-safe preprocessing.** A `ColumnTransformer` (scaling + one-hot encoding), with imputation **fitted on training data only**.
3. **Clustering as a feature.** K-Means fitted on training features only, with the cluster ID added as an input. Silhouette scores were 0.11–0.16; k = 2 gave a 71/29 "typical/atypical" split.
4. **Models.** Logistic Regression, Decision Tree, Random Forest, soft voting and bagging.
5. **Bias–variance check.** Learning curves of train vs validation ROC-AUC.
6. **Class imbalance.** SMOTE, random oversampling and random undersampling (`imblearn` pipelines).
7. **Tracking.** `mlflow.start_run` per model, with `log_params`, `log_metric` and `sklearn.log_model`.

## Results

**Online payments (test set)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Bagging** | 0.993 | **0.950** | 0.741 | **0.833** | **0.975** |
| Random Forest | 0.962 | 0.351 | 0.874 | 0.501 | 0.970 |
| Soft Voting | 0.955 | 0.313 | 0.880 | 0.462 | 0.967 |
| Decision Tree | 0.955 | 0.312 | 0.853 | 0.457 | 0.910 |
| Logistic Regression | 0.892 | 0.152 | 0.847 | 0.257 | 0.934 |

**Diabetes (test set):** Random Forest was best (F1 0.696, ROC-AUC 0.817). Ensembles didn't help on this small dataset.

**Learning curve (Random Forest, payments):** the train/validation ROC-AUC gap shrank from 0.027 to 0.020 as the training data grew, so the model generalises well.

**Resampling (Logistic Regression, payments)**

| Method | Recall | F1 | ROC-AUC | Training rows |
|---|---|---|---|---|
| SMOTE | 0.840 | 0.268 | 0.933 | 468,932 |
| Random oversampling | 0.847 | 0.258 | 0.934 | 468,932 |
| Random undersampling | 0.849 | 0.252 | 0.934 | **10,580** |

Undersampling matched the other methods while using ~2% of the rows, which makes it a cheap option when compute is limited.

## Key takeaways
- Accuracy is misleading on imbalanced data. Precision, recall, F1 and ROC-AUC tell the real story.
- Bagging suited the large, imbalanced problem, while a single Random Forest was enough for the small one.
- *Which* resampling method mattered less than resampling at all.

## Run it
```bash
pip install -r requirements.txt
# place data/transactions.csv and data/diabetes.csv (not included)
jupyter notebook notebooks/fraud_detection_mlflow.ipynb
mlflow ui            # then open http://localhost:5000
```

## Related work
The same methodology (`ColumnTransformer`, SMOTE in an `imblearn` pipeline, `GridSearchCV` on F1, MLflow tags per stage) was applied in a team project on **insurance car-claims fraud** (~6% fraud).

## Tech stack
Python · scikit-learn · imbalanced-learn · MLflow · pandas · NumPy · matplotlib · seaborn
