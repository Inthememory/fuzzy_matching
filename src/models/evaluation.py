from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss


def evaluate_model(model, X, Y, set_name):
    Y_pred = model.predict(X)

    log_loss_value = log_loss(Y, Y_pred)
    roc_auc_score_value = roc_auc_score(Y, Y_pred)
    Logger = logger.bind(log_loss=log_loss_value, roc_auc_score=roc_auc_score_value)
    Logger.info(f"Performance {set_name} set")

    confusion_matrix_value = confusion_matrix(Y, Y_pred)
    confusion_matrix_Logger = logger.bind(confusion_matrix=confusion_matrix_value)
    confusion_matrix_Logger.info(f"confusion_matrix {set_name}")

    return log_loss_value, roc_auc_score_value, confusion_matrix_value
