import polars as pl
import sys
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
    load_pairs_labeled,
    label_dataset,
    create_input_for_prediction,
    launch_training,
    get_prediction,
)

from nltk.corpus import stopwords

STOPWORDS_LIST = stopwords.words("english") + stopwords.words("french")

DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
MODEL_NAME = "xgb_model"

logger.remove(0)
logger.add(
    sys.stderr,
    format="{time} | {level} | {message} | {extra}",
)

if __name__ == "__main__":
    # Add argparse for the command line:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=str, required=True, nargs="+", help="Datasets listing."
    )
    parser.add_argument("--training", action="store_true", help="Execute training.")
    parser.add_argument(
        "--no-training", action="store_false", help="Don't execute training."
    )
    parser.set_defaults(training=True)
    args = parser.parse_args()

    # Loading of the configuration file:
    logger.info("Loading YAML configuration file")
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # Create datasets:
    datasetsLogger = logger.bind(datasets=[dataset for dataset in args.datasets])
    datasetsLogger.info("Create datasets")
    datasets = [gold(DATA_RAW_PATH, dataset, config) for dataset in args.datasets]

    # Build features
    logger.info("Build features")

    df_classification = classification(datasets, config["classification_levels"])

    df_classification_words = classification_words(
        datasets, config["classification_most_relevant_level"], STOPWORDS_LIST
    )

    df_sentence_transformer = sentence_transformer(datasets)

    df_syntax_ngram = syntax_ngram(datasets)

    df_syntax_words = syntax_words(datasets, STOPWORDS_LIST)

    # Model
    df_for_prediction = create_input_for_prediction(
        df_classification,
        df_classification_words,
        df_sentence_transformer,
        df_syntax_ngram,
        df_syntax_words,
    )
    print(df_for_prediction)

    indicators_var = [
        "similarity_syntax_ngram",
        "similarity_syntax_words",
        "similarity_sentence_transformer",
        "similarity_classification",
        "similarity_classification_words",
    ]
    label_var = ["left_side", "right_side"]
    target_var = "target"

    # Training
    if args.training:
        logger.info("Training of XGBoost Classifier")
        labeled_pairs = load_pairs_labeled(DATA_PROCESSED_PATH, "training_dataset")

        xgb_model = launch_training(
            df_for_prediction,
            labeled_pairs,
            indicators_var,
            label_var,
            target_var,
        )

        # Save model
        pickle_save(xgb_model, f"{MODELS_PATH}{MODEL_NAME}")

    xgb_model = pickle_get(f"{MODELS_PATH}{MODEL_NAME}")

    # Prediction
    predictions = get_prediction(
        df_for_prediction, indicators_var, label_var, target_var, xgb_model
    )
    predictions.write_csv(
        f"{DATA_PROCESSED_PATH}xgb_model_prediction.csv", separator=";"
    )
