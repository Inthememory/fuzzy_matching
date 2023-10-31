import yaml
import argparse
from loguru import logger
from utils import pickle_save, pickle_get
from src import (
    gold,
    classification,
    classification_words,
    sentence_transformer,
    syntax_ngram,
    syntax_words,
    create_input_for_prediction,
    load_pairs_labeled,
    label_dataset,
    get_train_test,
    xgb_classifier,
    get_model_performance,
    get_confusion_matrix,
    save_prediction,
)

from nltk.corpus import stopwords

STOPWORDS_LIST = stopwords.words("english") + stopwords.words("french")

DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"

if __name__ == "__main__":
    # Add argparse for the command line:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, help="List of datasets.")
    parser.add_argument(
        "--training", required=False, default=False, help="Execute training if true."
    )
    args = parser.parse_args()

    # Loading of the configuration file:
    logger.info("Loading YAML configuration file")
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Create dataset:
    logger.info("Create dataset")
    datasets = [gold(DATA_RAW_PATH, dataset, config) for dataset in args.datasets]

    # Build features
    logger.info("Build features")
    df_classification = classification(datasets, config["classification_levels"])
    df_classification_words = classification_words(
        datasets, config["classification_most_relevant_level"]
    )
    df_sentence_transformer = sentence_transformer(datasets)
    df_syntax_ngram = syntax_ngram(datasets)
    df_syntax_words = syntax_words(datasets)

    # Model
    df_input_for_prediction = create_input_for_prediction(
        df_classification,
        df_classification_words,
        df_sentence_transformer,
        df_syntax_ngram,
        df_syntax_words,
    )

    indicators_var = [
        "similarity_syntax_ngram",
        "similarity_syntax_words",
        "similarity_sentence_transformer",
        "similarity_classification",
        "similarity_classification_words",
    ]
    label_var = (["left_side", "right_side"],)
    target_var = "target"

    # Training
    if args.training:
        logger.info("Learning: Training of XGBoost Classifier")
        labeled_pairs = load_pairs_labeled(DATA_PROCESSED_PATH, "training_dataset")
        df_input_for_prediction_labeled = label_dataset(
            df_input_for_prediction, labeled_pairs
        )

        # Split the data into train and test set
        label_train, label_test, X_train, X_test, Y_train, Y_test = get_train_test(
            df_input_for_prediction_labeled,
            indicators_var,
            label_var=["left_side", "right_side"],
            target_var="target",
        )

        # Create and train model
        xgb_model = xgb_classifier(X_train, Y_train)

        # Performances
        log_loss_train, roc_auc_score_train = get_model_performance(
            X_train, Y_train, xgb_model
        )
        logger.info(
            f"Performance train set /n log_loss : {log_loss_train} /n roc_auc_score : {roc_auc_score_train}"
        )

        log_loss_test, roc_auc_score_test = get_model_performance(
            X_train, Y_train, xgb_model
        )
        logger.info(
            f"Performance test set /n log_loss : {log_loss_test} /n roc_auc_score : {roc_auc_score_test}"
        )

        confusion_matrix_test = get_confusion_matrix(X_train, Y_train, xgb_model)
        logger.info(f"confusion_matrix : {confusion_matrix_test}")

        # Save model
        pickle_get(xgb_model, f"{MODELS_PATH}xgb_1")

    else:
        xgb_model = pickle_get(f"{MODELS_PATH}xgb_1")

    # Prediction
    label = df_input_for_prediction.select(label_var)
    X = df_input_for_prediction.select(indicators_var)
    Y = df_input_for_prediction.select(target_var)
    save_prediction(label, X, Y, xgb_model, "xgb_model", MODELS_PATH)
