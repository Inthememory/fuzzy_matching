import polars as pl
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss

from fuzzy_matching.models.preprocessing import (
    create_input_for_prediction,
    label_dataset,
    get_train_test,
)
from fuzzy_matching.models.evaluation import evaluate_model
from fuzzy_matching.models.prediction import get_predictions
from fuzzy_matching.models.fit import xgb_classifier


def launch_training(
    df_for_prediction,
    labeled_pairs,
    indicators_var,
    label_var,
    target_var="target",
):
    # Create input
    df_for_prediction_labeled = label_dataset(df_for_prediction, labeled_pairs)

    # Split into train and test set
    label_train, label_test, X_train, X_test, Y_train, Y_test = get_train_test(
        df_for_prediction_labeled, indicators_var, label_var, target_var
    )

    # Create and train model
    xgb_model = xgb_classifier(X_train, Y_train)

    # Predict
    df_prediction = pl.concat(
        [label_test, X_test, Y_test]
        + [
            pl.DataFrame(xgb_model.predict(X_test), schema={"prediction": pl.Int64}),
            pl.DataFrame(
                xgb_model.predict_proba(X_test),
                schema={"proba_0": pl.Float64, "proba_1": pl.Float64},
            ),
        ],
        how="horizontal",
    )

    # Performances
    log_loss_test, roc_auc_score_test, confusion_matrix_test = evaluate_model(
        xgb_model, X_test, Y_test, "test"
    )

    return xgb_model, df_prediction
