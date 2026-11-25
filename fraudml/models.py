"""Model pipelines. Imputation, scaling and encoding are fitted on training data only, inside each pipeline."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def gpu_available() -> bool:
    try:
        import numpy as np
        import xgboost as xgb

        xgb.XGBClassifier(device="cuda", n_estimators=1).fit(np.zeros((4, 1)), [0, 1, 0, 1])
        return True
    except Exception:
        return False


def preprocessor(numeric, categorical) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])


def xgboost(pos_weight: float, device: str = "cpu", seed: int = 0):
    """Settings chosen by a 30-configuration random search scored on the validation month only
    (early stopping picked about 730 trees). The square root of the class ratio as `scale_pos_weight`
    worked better than the full ratio, which over-weights the rare class."""
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=750, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=1.0,
                         min_child_weight=10, reg_lambda=5.0, scale_pos_weight=pos_weight ** 0.5,
                         eval_metric="aucpr", tree_method="hist", device=device, random_state=seed)


def candidates(numeric, categorical, pos_weight: float, device: str = "cpu", seed: int = 0) -> dict:
    """The original coursework models (rebuilt) plus XGBoost, and the resampling strategies it compared."""
    def wrap(model):
        return Pipeline([("prep", preprocessor(numeric, categorical)), ("model", model)])

    def resampled(sampler, model):
        return ImbPipeline([("prep", preprocessor(numeric, categorical)), ("sample", sampler), ("model", model)])

    logreg = LogisticRegression(max_iter=3000, C=1.0)
    return {
        "Logistic Regression (class-weighted)": wrap(LogisticRegression(max_iter=3000, class_weight="balanced")),
        "Logistic Regression + SMOTE": resampled(SMOTE(random_state=seed), logreg),
        "Logistic Regression + undersampling": resampled(RandomUnderSampler(random_state=seed), logreg),
        "Random Forest": wrap(RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                                                     n_jobs=-1, random_state=seed)),
        "Bagging (decision trees)": wrap(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=2), n_estimators=100,
                                                           n_jobs=-1, random_state=seed)),
        "XGBoost": wrap(xgboost(pos_weight, device, seed)),
    }
