from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss

from src.models.preprocessing import label_dataset, get_train_test
from src.models.fit import xgb_classifier


def get_model_performance(X, Y, model):
    Y_pred = model.predict(X)
    return [log_loss(Y, Y_pred), roc_auc_score(Y, Y_pred)]


def get_confusion_matrix(X, Y, model):
    Y_pred = model.predict(X)
    return confusion_matrix(Y, Y_pred)


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

    # Performances
    log_loss_train, roc_auc_score_train = get_model_performance(
        X_train, Y_train, xgb_model
    )
    train_Logger = logger.bind(
        log_loss=log_loss_train, roc_auc_score=roc_auc_score_train
    )
    train_Logger.info("Performance train set")

    log_loss_test, roc_auc_score_test = get_model_performance(
        X_train, Y_train, xgb_model
    )
    test_Logger = logger.bind(log_loss=log_loss_test, roc_auc_score=roc_auc_score_test)
    test_Logger.info("Performance test set")

    confusion_matrix_test = get_confusion_matrix(X_train, Y_train, xgb_model)
    confusion_matrix_Logger = logger.bind(confusion_matrix=confusion_matrix_test)
    confusion_matrix_Logger.info("confusion_matrix")

    return xgb_model
