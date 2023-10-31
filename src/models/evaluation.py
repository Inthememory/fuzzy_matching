from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss


def get_model_performance(X, Y, model):
    Y_pred = model.predict(X)
    return [log_loss(Y, Y_pred), roc_auc_score(Y, Y_pred)]


def get_confusion_matrix(X, Y, model):
    Y_pred = model.predict(X)
    return confusion_matrix(Y, Y_pred)
