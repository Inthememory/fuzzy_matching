import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight


def xgb_classifier(X_train_xgb, Y_train_xgb):
    # define positive class scaling factor
    weights = compute_class_weight(
        "balanced",
        classes=np.unique(Y_train_xgb),
        y=Y_train_xgb.get_column("target").to_numpy(),
    )
    scale = weights[1] / weights[0]

    # Fix new param with optimal parameter
    xgb_model = XGBClassifier(
        learning_rate=0.2,
        n_estimators=100,
        gamma=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=8,
        min_child_weight=1,
        objective="binary:logistic",
        scale_pos_weight=scale,
    )
    xgb_model.fit(X_train_xgb, Y_train_xgb)
    return xgb_model
