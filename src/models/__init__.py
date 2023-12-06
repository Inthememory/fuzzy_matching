from loguru import logger

from src.models.preprocessing import create_input_for_prediction
from src.models.evaluation import evaluate_model
from src.models.prediction import export_prediction

from loguru import logger
import polars as pl
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss

from src.models.preprocessing import label_dataset, get_train_test
from src.models.fit import xgb_classifier


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
    df_prediction = export_prediction(
        [label_test, X_test, Y_test],
        xgb_model.predict(X_test),
        xgb_model.predict_proba(X_test),
        "test",
    )

    # Performances
    log_loss_train, roc_auc_score_train, confusion_matrix_train = evaluate_model(
        xgb_model, X_train, Y_train, "train"
    )

    log_loss_test, roc_auc_score_test, confusion_matrix_test = evaluate_model(
        xgb_model, X_test, Y_test, "test"
    )

    return xgb_model
